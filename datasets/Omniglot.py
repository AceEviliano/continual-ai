import os.path as pth
from glob import glob
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import defaultdict


def get_transforms(mode='view'):
    
    mean_values = [0.485, 0.456, 0.406]
    std_values = [0.229, 0.224, 0.225]

    if mode=='train':
        imgTransform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                std=std_values)
        ])
    elif mode=='test':
        
        imgTransform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_values,
                                std=std_values)
        ])
    elif mode=='view':
        
        imgTransform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

    return imgTransform


class Dset(Dataset):

    def __init__(self, root='.', train=True, split_size=0.5, 
                 transform=None, target_transform=None):

        self.root = root
        self.train = train
        self.split_size = split_size
        self.transform, self.target_transform = transform, target_transform
        
        datapath = glob( pth.abspath(self.root+'/omniglot/images_background/'+'/*/*/*') )
        datapath += glob( pth.abspath(self.root+'/omniglot/images_evaluation/'+'/*/*/*') )

        self.prepare_data(datapath)       
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)

        return

    
    def prepare_data(self, datapath):

        classInx, indices = 0, defaultdict(lambda:[])
        self.data, self.targets = [], []
        self.idx2class = defaultdict(lambda:'')
        self.class2idx = defaultdict(lambda:-1)

        for inx, imgpth in enumerate(datapath):
            classId = imgpth.split('/')[-3] + '-' + imgpth.split('/')[-2]
            if classId not in self.class2idx.keys():
                self.idx2class[classInx] = classId
                self.class2idx[classId] = classInx
                classInx += 1
        
        for inx, imgpth in enumerate(datapath):
            classId = imgpth.split('/')[-3] + '-' + imgpth.split('/')[-2]
            classInx = self.class2idx[classId]
            self.data.append([imgpth, classInx, inx])
            self.targets.append(classInx)

        return


    def __getitem__(self, inx):
        
        imgpth, target, instid = self.data[inx]
        
        imgpth = pth.join(imgpth)
        img = Image.open(imgpth).convert('RGB')
        target, instid = int(target), int(instid)

        if self.transform != None:
            img = self.transform(img)
            
        if self.target_transform != None:
            target = self.target_transform(target)
        else:
            target = torch.tensor(target, dtype=torch.long)
            
        return img, target, instid


    def __len__(self):
        return len(self.data)