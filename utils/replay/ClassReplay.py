import numpy as np
import numpy.linalg as linalg

import torch
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict

class ClassExemplarReplay( Dataset ):

    
    def __init__(self, handout, budget=1):
        
        self.budget = budget
        self.handout = handout
        self.memory = OrderedDict()
        self.mem_size = defaultdict(lambda : 0)
        return
        
        
    def add(self, sample):
        
        classid = sample['classid']
        self.strategy(sample, classid)
        return
    
    
    def get_index(self, inx):
        
        i=0
        for k,v in self.mem_size.items():
            i+=v
            if i>inx:
                i-=v
                break
        
        i = inx - i
        return k, i
    
    
    def __getitem__(self, inx):
        k, inx = self.get_index(inx)
        mem_sample = self.memory[k][inx]
        
        sample = []
        for key in self.handout:
            sample.append(mem_sample[key]) 
            
        return sample
        
    
    def __len__(self):
        return sum(self.mem_size.values())
    
    
    def strategy(self, sample, classid):
        raise NotImplementedError
    
    

class ClassRecentReplay( ClassExemplarReplay ):
    
    def __init__(self, handout, budget=1):
        
        super().__init__(handout, budget)
        return
    
    
    def strategy(self, sample, classid):
        
        if classid not in self.memory.keys():
            self.memory[classid] = []
        
        if len(self.memory[classid]) < self.budget:
            self.memory[classid] += [sample]
            self.mem_size[classid] += 1
        else:
            self.memory[classid].pop(0)
            self.memory[classid] += [sample]
            
        return
    
    
    
class ClassHardestReplay( ClassExemplarReplay ):
    
    def __init__(self, handout, budget=1, gamma=0.1):
        
        super().__init__(handout, budget)
        
        self.gamma = gamma
        self.prototypes = defaultdict(lambda : None)  
        return
    
    
    def preupdate(self, classid):
        
        features = [ ex['feature'] for ex in self.memory[classid] ]
        new_ptype = torch.mean(torch.stack(features))
        
        old_ptype = self.prototypes[classid]
        new_ptype = (self.gamma)*new_ptype + (1-self.gamma)*old_ptype
        self.prototypes[classid] = new_ptype
        
        ptype = self.prototypes[classid]
        for inx in range(len(self.memory[classid])):
            sample = self.memory[classid][inx]
            self.memory[classid][inx]['dist'] = linalg.norm(ptype - sample['feature'])
        
        return
    
    
    def postupdate(self, classid):
        
        features = [ ex['feature'] for ex in self.memory[classid] ]
        self.prototypes[classid] = torch.mean(torch.stack(features))
        
        ptype = self.prototypes[classid]
        for inx in range(len(self.memory[classid])):
            sample = self.memory[classid][inx]
            self.memory[classid][inx]['dist'] = linalg.norm(ptype - sample['feature'])
            
        return
        
    
    def strategy(self, sample, classid):
            
        if classid not in self.memory.keys():
            self.memory[classid] = [ sample ]
        else:
            self.memory[classid] += [ sample ]
            
        if self.prototypes[classid] is None:
            self.prototypes[classid] = sample['feature']
        
        self.preupdate(classid)
        
        if len(self.memory[classid]) <= self.budget:
            self.mem_size[classid] += 1
        else:
            maxinx = np.argmin([ex['dist'] for ex in self.memory[classid]])
            self.memory[classid].pop(maxinx)
            
        self.postupdate(classid)
            
        return