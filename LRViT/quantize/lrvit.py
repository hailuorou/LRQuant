import torch
import torch.nn as nn
from models.int_vit_layer import QuantViTLayer
from quantize.int_linear import QuantLinear
from contextlib import nullcontext
import copy
import math
import utils
import pdb
import gc



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}

def LRViT(
    vits,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = vits.model
    processor = vits.processor
    dev = vits.device
    model.to(dev)

    if "vit" in args.net.lower() or 'deit' in args.net.lower():
        layers = model.vit.encoder.layer
        model.vit.embeddings.patch_embeddings = model.vit.embeddings.patch_embeddings.to(dev)
        model.vit.layernorm = model.vit.layernorm.to(dev)
        DecoderLayer = QuantViTLayer
        pairs = {
            "query":"qkv",
            "attention.output.dense":"fc1",
            "intermediate.dense":"fc2",
        }
        layer_name_prefix = "vit.encoder.layer"
    else:
        raise ValueError("Only support for vit/deit/swin now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast
    
    input_dimension = (model.config.image_size * model.config.image_size) // (model.config.patch_size * model.config.patch_size)
    inps = torch.zeros(
            (args.nsamples, input_dimension+1, model.config.hidden_size), dtype=dtype, device=dev
        )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, inp2, inp3, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                inputs = processor(images=batch, return_tensors="pt")
                model(**inputs.to(dev)) 
            except ValueError:
                pass
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    model.to('cpu')
    
    if "vit" in args.net.lower() or "deit" in args.net.lower():
        model.vit.embeddings.patch_embeddings = model.vit.embeddings.patch_embeddings.to('cpu')
        model.vit.layernorm = model.vit.layernorm.to('cpu')
    else:
        raise ValueError("Only support for vit/deit now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    quant_inps_fp = copy.deepcopy(inps)

    loss_func = torch.nn.MSELoss()
    cossim = nn.CosineSimilarity(dim=2)

    
    smooth_cos= []
    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        qlayer = DecoderLayer(vits.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        qlayer.set_quant_state(weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0))[0]
                        quant_inps_fp[j] = qlayer(quant_inps[j].unsqueeze(0))[0]

        # init smooth parameters
        qlayer.set_quant_state(weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.let = args.let
        use_shift = True 
        if args.let:
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)

                            scale = (act/torch.log2(2+act)).clamp(min=1e-5) #weight
                            r1 = torch.ones(module.weight.shape[0], 1).to(dev)
                            if use_shift:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            if args.lr_plus:
                                name_tmp = name.replace(".","_")
                                qlayer.register_parameter(f"{name_tmp}_smooth_rotate",torch.nn.Parameter(r1,requires_grad=False)) 
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
        
        qlayer.float()
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # required for AMP training
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":qlayer.let_parameters(use_shift),"lr":args.let_lr}, 
                 {"params":qlayer.lwc_parameters(),"lr":args.lwc_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            quant_zeros1 = torch.zeros_like(quant_inps)
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        qlayer.smooth_and_quant_temporary()
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,])[0]
                        loss1 =  loss_func(fp_inps[index:index+args.batch_size,], quant_out) 
                        cos1 = cossim(quant_out,fp_inps[index:index+args.batch_size,]).mean().abs()
                        loss2 = -torch.log(cos1)
                        if args.lr_plus:
                            loss1 += loss_func(quant_inps_fp[index:index+args.batch_size,], quant_out)
                            cos2 = cossim(quant_inps_fp[index:index+args.batch_size,],quant_out).mean().abs()
                            loss2 -= torch.log(cos2)
                        loss = loss1 + loss2
                        quant_zeros1[j]=quant_out
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.data)
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=qlayer.lr_parameters(use_shift))
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(vits._device) / 1024**2} ")
            qlayer.clear_temp_variable()
            del optimizer

            cos_smooth = cossim(quant_zeros1,fp_inps).mean()
            smooth_cos.append(cos_smooth.detach().cpu().item())
            logger.info(f"smooth cos: {cos_smooth}")
        # real smooth and quantization
        qlayer.smooth_and_quant_inplace()
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0))[0]
            qlayer.register_scales_and_zeros()
            qlayer.half()
            layers[i] = qlayer.to("cpu")
        else:
            qlayer.register_scales_and_zeros()
            qlayer.half()
            layers[i] = qlayer.to("cpu")
        del layer
        torch.cuda.empty_cache()
    logger.info(f"smooth cos: {smooth_cos}")
    del inps
    del quant_inps
    del fp_inps
    torch.cuda.empty_cache()
    gc.collect()                    
    return model

