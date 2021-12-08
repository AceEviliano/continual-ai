import numpy as np
import sklearn.metrics as Metrics
from collections import defaultdict
import torch


class Print():

    def __init__(self, frequency=1):
        '''
        Displays info regarding which phase of the training is going.
        A generic callback that can be added as any phase callback.
        Does not take any pay load during the call.

        Init Prams:
            frequency: Parameter to control the output update frequency.
        '''
        self.frequency = frequency
        self.C = {
            'BF': '\033[1m',
            'END': '\033[0m',
        }
        return

    def __call__(self, log, phase=None):
        if phase != None:
            if log.S[phase] % self.frequency == 0:
                print(
                    f'{self.C["BF"]}Completed phase:{self.C["END"]} {phase} {self.C["BF"]}step:{self.C["END"]} {log.S[phase]}')
        else:
            if log.E % self.frequency == 0:
                print(
                    f'\n{self.C["BF"]}Completed epoch:{self.C["END"]} {log.E}\n')
        return


class PrintMetrics():

    def __init__(self, metrics=None, frequency=1):
        '''
        '''
        self.frequency = frequency
        if metrics == None:
            self.metrics = {'accuracy': (Metrics.accuracy_score, {})}
        else:
            self.metrics = metrics
        return

    def __call__(self, log, phase=None):

        if phase == None:
            phase = 'test'
            iterator = log.epoch[log.E-1]['step']
        else:
            iterator = log.step
            
        preds, trues = [], []
        for d in iterator[phase]:
            preds.append(d['preds'])
            trues.append(d['trues'])
            
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        
        for name, m in self.metrics.items():
            metric, kwargs = m
            score = metric(preds, trues, **kwargs)
            print(f'{name}: {score}')
            
        return


class SaveLog():

    def __init__(self, savepath='untitled.log', frequency=1,
                 reduced=False, metrics=None, phase=['train', 'test']):

        self.savepath = savepath
        self.frequency = frequency
        self.reduced = reduced
        self.metrics = metrics
        self.phase = phase
        assert (reduced and metrics != None) or (
            not reduced), "provide metrics for reduced"

        return

    def __call__(self, log, data):
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

        if log.E % self.frequency != 0:
            return

        import json
        with open(self.savepath, 'w') as fp:
            if self.reduced:
                json.dump(self.save_dict, fp, indent=4)
            else:
                json.dump(log.task, fp, indent=4)

        return


class SaveModel():

    def __init__(self, metric=None):
        '''
        Saves the model in pytorch format ('.pt').
        An Epoch phase Callback. Takes a data payload at call.

        Init Params:
            metric: metric used to choose the optimal model. Savemodel callbacks
            with different metrics can be used to save different optimal models
            for that score.

        Callback Params:
            data: {
                'savepath': 'dir/to/save/model',
                'k1': v1, ...
            }
            A dictionary with atleast 'savepath' key passed at end of every epoch. 
            Saves only the optimal model based on the initialized metric. The metric
            is evaluated on the test preds.
        '''
        if metric == None:
            self.metric = (Metrics.accuracy_score, {})
        else:
            self.metric = metric
        self.max_score = 0.
        return

    def __call__(self, log):

        data  = log.epoch[-1]['data']
        savepath = log.epoch[-1]['data']['savepath']
        
        preds = 0
        trues = 0
        
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
        '''
        Displays time taken per epoch. 
        A Epoch phase callback, does not take any data payload.
        [Needs to updated as a generic callback].
        '''
        self.start = 0.

    def __call__(self, log, data=None):
        import time
        self.end = time.time()
        if log.E > 0:
            print(f'Epoch time: {self.end-self.start:.2f}(s)')
        self.start = self.end
        return
