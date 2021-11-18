from collections import defaultdict

class Logger():

    def __init__(self, callbacks=None):

        self.E = 0
        self.S = defaultdict(lambda:0)
        self.task = {}
        self.curr_task = None

        if callbacks == None:
            self.callbacks = {
                'step' : [ (lambda phase, data: None) ],
                'epoch': [ (lambda data: None) ],
                'task' : [ (lambda task, data: None) ]
            }
        else:
            self.callbacks = callbacks
            
        return


    def add_step(self, phase, data=None):
        
        self.S[phase] += 1
        for callback in self.callbacks['step']:
            callback(self, phase=phase, data=data)

        return

    
    def add_epoch(self, data=None):

        self.E += 1 
        self.S = defaultdict(lambda:0)
        
        for callback in self.callbacks['epoch']:
            callback(self, data=data)

        return

    
    def add_task(self, taskname, data=None):
        import copy

        self.E = 0
        self.S = defaultdict(lambda:0)
        self.task[taskname] = {'data': [] }
        self.curr_task = taskname
        
        for callback in self.callbacks['task']:
            callback(self, taskname=taskname, data=data)

        return