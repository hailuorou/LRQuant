import transformers
import torch
from .models_utils import BaseViT, find_layers
from transformers import ViTImageProcessor, ViTForImageClassification, SwinForImageClassification, VivitImageProcessor, VivitForVideoClassification
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import pdb


class ViTClass(BaseViT):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size
        
        if "vivit" in args.net.lower():
            self.processor = VivitImageProcessor.from_pretrained(args.model)
        else:
            self.processor = ViTImageProcessor.from_pretrained(args.model)
        
        if "vivit" in args.net.lower():
            self.model = VivitForVideoClassification.from_pretrained(args.model)
        else:
            self.model = ViTForImageClassification.from_pretrained(args.model)

        self.model.eval()


    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

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
