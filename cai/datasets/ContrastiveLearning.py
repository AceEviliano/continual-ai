import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

class ContrastiveDset( Dataset ):

    def __init__(self, dset):
        self.dset = dset
        self.indices = defaultdict(lambda:[])
        self.classes = set(dset.targets)

        for inx in range(len(dset)):
            self.indices[dset.targets[inx]].append(inx)
        return

    
    def __getitem__(self, ainx):
        aimg, albl, _, _ = self.dset[ainx]

        plbl = albl
        pinx = np.random.choice(self.indices[albl])
        pimg, plbl, _, _ = self.dset[pinx]
        
        nlbl = list(self.classes - set([plbl]))
        nlbl = np.random.choice(nlbl)
        ninx = np.random.choice(self.indices[nlbl])
        nimg, nlbl, _, _ = self.dset[ninx]

        return aimg, pimg, nimg, (albl, plbl, nlbl)


    def __len__(self):
        return len(self.dset)