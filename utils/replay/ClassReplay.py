import numpy as np
import numpy.linalg as linalg

import torch
from torch.utils.data import Dataset
from collections import defaultdict, OrderedDict

class ClassExemplarReplay( Dataset ):

    
    def __init__(self, handout, budget=1):
        '''
        '''
        self.budget = budget
        self.handout = handout
        self.memory = []
        self.indices = defaultdict(list)
        return
        
        
    def add(self, sample):
        '''
        '''
        classid = sample['classid']
        self.strategy(sample, classid)
        return
    
   
    def __getitem__(self, inx):
        '''
        '''
        mem_sample = self.memory[inx]
        sample = []
        for key in self.handout:
            sample.append(mem_sample[key]) 
            
        return sample
        
    
    def __len__(self):
        '''
        '''
        return len(self.memory)
    
    
    def strategy(self, sample, classid):
        '''
        '''
        raise NotImplementedError
    
    

class ClassRecentReplay( ClassExemplarReplay ):
    
    def __init__(self, handout, budget=1):
        '''
        Replaces the oldest sample for the class with the incoming sample of
        that class.
        
        Params:
            handount: keys corresponding to the values in the sample dictionary
            that needs to be returned during __getitem__.
            budget: number of samples that can be held per class.
        '''
        super().__init__(handout, budget)
        return
    
    
    def strategy(self, sample, classid):
        
        if classid not in self.indices.keys():
            self.indices[classid] = []
        
        if len(self.indices[classid]) < self.budget:
            self.memory.append(sample)
            self.indices[classid].append(len(self.memory)-1)            
        else:
            inx = self.indices[classid][0]
            self.memory[inx] = sample
            inx = self.indices[classid].pop(0)
            self.indices[classid].append(inx)
            
        return


class ClassRandomReplay( ClassExemplarReplay ):
    
    def __init__(self, handout, budget=1, p=0.5):
        '''
        An incoming sample replaces one of the existing samples randomly with
        probability 'p' if memory buffer for the class is full.
        
        Params:
            handount: keys corresponding to the values in the sample dictionary
            that needs to be returned during __getitem__.
            budget: number of samples that can be held per class.
            p: probability for random retention of the new sample.            
        '''
        super().__init__(handout, budget)
        self.p = p
        return
    
    
    def strategy(self, sample, classid):
        
        if classid not in self.indices.keys():
            self.indices[classid] = []
        
        if len(self.indices[classid]) < self.budget:
            self.memory.append(sample)
            self.indices[classid].append(len(self.memory)-1)            
        elif np.random.random() < self.p:
            inx = np.random.choice(self.indices[classid])
            self.memory[inx] = sample
            
        return
     
  

if __name__ == '__main__':
    
    rpm = ClassRandomReplay(['img', 'classid'], budget=2)
    
    rpm.add({'img':'abc', 'classid':0})
    rpm.add({'img':'cab', 'classid':0})
    rpm.add({'img':'jkl', 'classid':1})
    rpm.add({'img':'bac', 'classid':0})
    rpm.add({'img':'lkj', 'classid':1})
    rpm.add({'img':'kjl', 'classid':1})
    
    print(rpm.__dict__)
    
    
    