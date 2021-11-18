import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


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

    def __init__(self, root='.', train=True,
                 transform=None, target_transform=None):
        
        super().__init__()
        
        self.root = os.path.join(root, 'CUB_200_2011')
        self.train = 1 if train else 0
        self.transform = transform
        self.target_transform = target_transform

        self.prepare_data()
        self.data = list(self.data)
        for instid, row in enumerate(self.data):
            self.data[instid] = list(row) + [instid]
        self.data = np.array(self.data)
        self.targets = self.data[:,1]
                
        return


    def prepare_data(self):

        # Load image paths & train test split info from files
        fname = os.path.join(self.root, 'images.txt')
        train_test_split = os.path.join(self.root, 'train_test_split.txt')

        # Filter entries based on train and test
        filenames = pd.read_csv( fname, sep=' ', header=None, 
                                 names=['id', 'filenames'])
        splits = pd.read_csv( train_test_split, sep=' ', header=None, 
                              names=['id', 'splits'])
        data = pd.merge(filenames, splits, on=['id'])
        self.data = data.loc[(data['splits'] == self.train), ['filenames']]

        # Labeling images
        labeler = lambda x: int(str(x).split('.')[0]) - 1
        self.data['labels'] = self.data.applymap(labeler)
        self.data = self.data.to_numpy()

        return


    def __getitem__(self, inx):
        
        imgpath, target, instid = self.data[inx]
        
        imgpath = os.path.join(self.root, 'images/'+imgpath)
        img = Image.open(imgpath).convert('RGB')
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