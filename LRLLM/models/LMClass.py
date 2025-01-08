import transformers
import torch
from .models_utils import BaseLM, find_layers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM,LlavaConfig
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pdb
from accelerate import init_empty_weights


class LMClass(BaseLM):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model_config = args.model
        if 'llava' in args.model:
            import sys
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            enc, model, image_processor, context_len = load_pretrained_model(
            model_path=args.model,
            model_base=None,
            model_name=get_model_name_from_path(args.model),
            device="cpu",
            **{"use_cache": False}
        )
            self.tokenizer = enc
            self.model = model
        elif 'vila' in args.model.lower():
            print('vila')
            config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
            config.use_cache = False
            kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
            self.tokenizer = AutoTokenizer.from_pretrained(
                    args.model, use_fast=False, trust_remote_code=True
                )
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model, config=config, trust_remote_code=True, **kwargs
            )
        else:
            config = AutoConfig.from_pretrained(args.model)
            self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False)
            self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16)

        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab size: ", self.vocab_size)

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():

            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
