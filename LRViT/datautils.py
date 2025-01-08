import os
import glob
import numpy as np
import torch
import random
from PIL import Image

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_imagenet1(nsamples, seed, model):
    print("get_imagenet")

    # file path
    folder_path = '/root/dataset/ImageNet/other'
    image_files = glob.glob(os.path.join(folder_path, '*.jpg'))

    random.shuffle(image_files)
    
    trainloader = []
    testenc = []
    for i in range(nsamples):
        image = Image.open(image_files[i])
        trainloader.append(image)
        testenc.append(image)
    return trainloader, testenc

def get_loaders(
    name, nsamples=128, seed=0, model='',
):
    if 'ImageNet' in name:
        return get_imagenet1(nsamples, seed, model)
