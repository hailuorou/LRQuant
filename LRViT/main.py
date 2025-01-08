
from PIL import Image
import os
import sys
import random
import numpy as np
from models.ViTClass import ViTClass
from models.int_vit_layer import QuantViTLayer
import torch
import time
from datautils import get_loaders
from quantize.lrvit import LRViT
import utils
from pathlib import Path
from torch.cuda.amp import autocast

from quantize.int_linear import QuantLinear
from data_loaders import *
from torch.cuda.amp import autocast
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.backends.cudnn.benchmark = True

net_choices = [
    "vit-base-patch16-224"
]

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--real_quant", default=False, action="store_true",)
    parser.add_argument("--calib_dataset",type=str,default="ImageNet",
        choices=["ImageNet", "kinetics"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--nsamples", type=int, default=32, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr_plus",default=False, action="store_true",help="LRQuant+")
    parser.add_argument("--let",default=False, action="store_true",help="activate learnable equivalent transformation")
    parser.add_argument("--lwc",default=False, action="store_true",help="activate learnable weight clipping")
    parser.add_argument("--aug_loss", default=False, action="store_true", help="calculate additional loss with same input")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)
    print('-----------------------start----------------------------')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    args.model_family = args.net.split('-')[0]
    vits = ViTClass(args)
    vits.model.eval()
    for param in vits.model.parameters():
        param.requires_grad = False

    

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc": args.lwc,
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric":False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'

    # quantization
    if args.wbits < 16 or args.abits <16:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
            )
            torch.save(dataloader, cache_dataloader)    
        act_scales = None
        act_shifts = None
        if args.let:
            act_scales = torch.load(args.act_scales)
            act_shifts = torch.load(args.act_shifts)
        LRViT(
            vits,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
        )
        logger.info(time.time() - tick)
    if args.save_dir:
        # delete omni parameters
        for name, module in vits.model.named_modules():
            if isinstance(module, QuantLinear):
                # del module.weight_quantizer.compensation_factor
                del module.weight_quantizer.upbound_factor
                del module.weight_quantizer.lowbound_factor
                del module.act_quantizer.upbound_activation_factor
                del module.act_quantizer.lowbound_activation_factor
            if isinstance(module, QuantViTLayer):
                if args.let:
                    del module.qkv_smooth_scale
                    del module.qkv_smooth_shift
                    del module.fc2_smooth_scale
                    del module.fc2_smooth_shift
                    del module.fc1_smooth_scale
                    del module.fc1_smooth_shift           
        vits.model.save_pretrained(args.save_dir)  
    print('---------------------')
    torch.cuda.empty_cache()
    vits.model.eval()

    # load ImageNet dataset
    imagenet_dataloader = eval("{}DataLoader".format('ImageNet'))(
                        args.net,
                        data_dir=os.path.join('/root/dataset/', 'ImageNet'),
                        image_size=224,
                        batch_size=32,
                        num_workers=2,
                        split='val')
    vits.model.to(vits.device)

    # 评估模型准确性
    begin_time = time.time()
    accuracy = evaluate_model(vits.model, vits.processor, imagenet_dataloader, vits.device)
    logger.info(f"consumer time: {time.time()-begin_time}")

    # print("Model Accuracy on ImageNet:", accuracy)
    logger.info(f"Model Accuracy on ImageNet: {accuracy}")

# evaluate on image net
def evaluate_model(model, processor, dataloader, dev):
    total_samples = len(dataloader.dataset)
    correct_predictions = 0
    model.eval()
    with autocast(dtype=torch.float16):
        for data in dataloader:
            inputs = {"pixel_values": data[0].to(dev)}
            labels = data[1].to(dev)
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(predictions == labels)

    accuracy = correct_predictions.item() / total_samples
    return accuracy
        


if __name__ == "__main__":
    print(sys.argv)
    main()
