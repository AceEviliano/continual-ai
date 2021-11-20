import copy
from collections import defaultdict


class Logger():

    def __init__(self, callbacks=None):
        '''
        Initializes a logger object with callbacks to be performed during a
        __step__, __epoch__ and __task__.
        '''
        self.task, self.T = {}, ''
        self.epoch, self.E = [], 0
        self.step, self.S  = defaultdict(list), defaultdict(lambda: 0)

        if callbacks == None:
            self.callbacks = {
                'step':  [(lambda phase:None)],
                'epoch': [(lambda: None)],
                'task':  [(lambda: None)]
            }
        else:
            self.callbacks = callbacks

        return

    def add_step(self, phase, data=None):
        '''
        To be called at the end of step. Step for logger object is defined as the
        event phase when a batch from the data is processed by the model. Callbacks
        pertaining to step phase will be called post phase information update.
        '''
        self.step[phase].append( copy.deepcopy(data) )

        for callback in self.callbacks['step']:
            callback(self, phase=phase)

        self.S[phase] += 1
        return

    def add_epoch(self, data=None):
        '''
        To be called at the end of epoch. Epoch for a logger is defined as the 
        event phase when the entire data is processed once by the model. Callbacks
        pertaining to epoch phase will be called. 
        '''
        self.epoch.append({
            'step': copy.deepcopy(self.step),
            'data': copy.deepcopy(data)
        })    

        for callback in self.callbacks['epoch']:
            callback(self)

        self.E += 1
        self.step, self.S = defaultdict(list), defaultdict(lambda: 0)
        return

    def add_task(self, taskname, data=None):
        '''
        To be called at the end of task. Task for a logger is defined as the
        event phase when the entire data is processed and learning task ends 
        for the model. Callbacks pertaining to Task phase will be called. 
        '''
        self.T = taskname
        self.task[self.T] = {
            'epoch': copy.deepcopy(self.epoch),
            'data' : copy.deepcopy(data) 
        }

        for callback in self.callbacks['task']:
            callback(self)

        self.epoch, self.E = [], 0
        self.step, self.S = defaultdict(list), defaultdict(lambda: 0)
        return
