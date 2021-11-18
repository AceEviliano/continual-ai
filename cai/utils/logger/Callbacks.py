import numpy as np
import sklearn.metrics as Metrics
from collections import defaultdict
import torch

class AddPreds():

    def __call__(self, log, phase, data):
        preds = data['preds'].cpu().detach().numpy()
        trues = data['trues'].cpu().detach().numpy()

        task = log.curr_task
        if (log.E+1) > len(log.task[task]['data']):
            log.task[task]['data'].append({ 
                                        'train': { 'preds': [],
                                                    'trues': [] },
                                        'test' : { 'preds': [],
                                                    'trues': [] } 
                                    })
        
        log.task[task]['data'][log.E][phase]['preds'].append(preds)
        log.task[task]['data'][log.E][phase]['trues'].append(trues)
        return



class Print():

    def __init__(self, frequency=1):
        self.frequency = frequency
        self.C = {
            'BF': '\033[1m',
            'END': '\033[0m',
        }
        return

    def __call__(self, log, phase=None, taskname=None, data=None):
        if phase != None:
            if log.S[phase]%self.frequency == 0:
                print(f'{self.C["BF"]}Completed phase:{self.C["END"]} {phase} {self.C["BF"]}step:{self.C["END"]} {log.S[phase]}')
        elif taskname != None:
                print(f'\n\n {self.C["BF"]}Task:{self.C["END"]} {log.curr_task}\n\n')
        else:
            if log.E%self.frequency == 0:
                print(f'\n{self.C["BF"]}Completed epoch:{self.C["END"]} {log.E}\n')
        return



class PrintMetrics():

    def __init__(self, metrics=None, frequency=1):

        self.frequency = frequency
        if metrics == None:
            self.metrics = { 'accuracy': (Metrics.accuracy_score,{}) }
        else:
            self.metrics = metrics
        return


    def __call__(self, log, phase=None, data=None):      

        s = ''
        task = log.curr_task
        if phase!=None:
            preds = log.task[task]['data'][log.E][phase]['preds']
            trues = log.task[task]['data'][log.E][phase]['trues']
        else:
            preds = log.task[task]['data'][log.E-1]['test']['preds']
            trues = log.task[task]['data'][log.E-1]['test']['trues']

        if len(preds) == 1:
            preds, trues = preds[0], trues[0]
        else:
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)

        printout = ''
        for k, (metric, kwargs) in self.metrics.items():
            v = metric(preds, trues, **kwargs)
            printout += f'{k}: {v:.2f}\t'

        if phase != None:
            if log.S[phase]%self.frequency == 0:
                print(printout)
        else:
            if log.E%self.frequency == 0:
                print(printout)

        return



class SaveLog():

    def __init__(self, savepath='untitled.log', frequency=1, 
                 reduced=False, metrics=None, phase=['train', 'test']):

        self.savepath = savepath
        self.frequency = frequency
        self.reduced = reduced
        self.metrics = metrics
        self.phase = phase
        assert (reduced and metrics!=None) or (not reduced), "provide metrics for reduced"

        return


    def __call__(self, log, data):

        import copy
        if log.E-1 == 0:
            self.save_dict = copy.deepcopy(log.task)
            self.save_dict[log.curr_task]['data'] = []
                       
        task = log.curr_task
        for p in self.phase:
            for keys in ['preds', 'trues']:
                v = log.task[task]['data'][log.E-1][p][keys]
                if type(v[0]) != int:
                    v = np.concatenate(v).tolist()
                log.task[task]['data'][log.E-1][p][keys] = v
    
        
        if self.reduced:
            m = defaultdict(dict)
            for p in self.phase:
                preds = log.task[task]['data'][log.E-1][p]['preds']
                trues = log.task[task]['data'][log.E-1][p]['trues']
                for k, (metric, kwargs) in self.metrics.items():
                    m[p][k] = metric(preds, trues, **kwargs)

            self.save_dict[task]['data'].append(m)


        if log.E%self.frequency != 0:
            return
        
        import json
        with open(self.savepath, 'w') as fp:
            if self.reduced:
                json.dump(self.save_dict, fp, indent=4)
            else:
                json.dump(log.task, fp, indent=4)

        return


class SaveModel():

    def __init__(self, metric=(Metrics.accuracy_score, {}) ):
        self.metric = metric
        self.max_score = 0.
        return

    def __call__(self, log, data=None):
        
        task = log.curr_task
        savepath = data['savepath']

        preds = log.task[task]['data'][log.E-1]['test']['preds']
        trues = log.task[task]['data'][log.E-1]['test']['trues']
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        metric, kwargs = self.metric
        new_score = metric(preds, trues, **kwargs)

        if self.max_score < new_score:
            self.max_score = new_score
            data['score'] = new_score
            torch.save(data, savepath)
            print(f'\033[1mSaving model \033[0m score : {new_score:.3f}')

        return



class Timer():

    def __init__(self):
        self.start = 0.
    
    def __call__(self, log, data=None):
        import time
        self.end = time.time()
        if log.E > 1:
            print(f'Epoch time: {self.end-self.start:.2f}(s)')
        self.start = self.end
        return