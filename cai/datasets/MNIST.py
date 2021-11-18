import os.path as pth
from glob import glob
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import defaultdict


class GrayToRGB( object ):

    def __init__(self):
        return

    def __call__(self, tensor):
        tensor = tensor.squeeze()
        tensor = tensor.unsqueeze(0)
        tensor = tensor.repeat(3, 1, 1)
        return tensor


def get_transforms(mode='view'):
    
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]

    if mode=='train':
        imgTransform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            GrayToRGB(),
            transforms.Normalize(mean=mean_values,
                                std=std_values)
        ])
    elif mode=='test':
        
        imgTransform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            GrayToRGB(),
            transforms.Normalize(mean=mean_values,
                                std=std_values)
        ])
    elif mode=='view':
        
        imgTransform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            GrayToRGB()
        ])

    return imgTransform



class Dset( Dataset ):

    def __init__(self, root, train=True, download=False,
                 transform=None, target_transform=None):
        
        from torchvision.datasets import MNIST
        
        self.dset = MNIST( root, train=train, download=download,
                           transform=transform, target_transform=target_transform )
        self.targets = self.dset.targets
        return

    
    def __getitem__(self, inx):
        img, lbl = self.dset[inx]
        return img, lbl, inx


    def __len__(self):
        return len(self.dset)