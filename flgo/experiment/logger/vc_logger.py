import os
import torch
from flgo.experiment.logger import BasicLogger
import numpy as np


class VCLogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        self.set_es_key('local_test_f1')
        self.turn_es_direction()

    def show_current_output(self, yes_key=['train', 'test', 'val'], no_key=['dist']):
        for key, val in self.output.items():
            a = [(yk in key) for yk in yes_key]
            nf = [(nk not in key) for nk in no_key]
            if np.all(nf) and np.any(a) and type(val) is list:
                try:
                    content = self.temp.format(key, val[-1])
                except:
                    content = "{}:".format(key)+str(val[-1])
                self.info(content)
            elif np.all(nf) and np.any(a):
                try:
                    content = self.temp.format(key, val)
                except:
                    content = "{}:".format(key)+str(val)
                self.info(content)  
                    
    def log_once(self, *args, **kwargs):
        # local performance
        cvals = []
        ctests = []
        model_name = self.option['m']
        for c in self.clients:
            model = c.model if (hasattr(c, 'model') and c.model is not None) else self.server.model
            cvals.append(c.test(model, 'val'))
            ctests.append(c.test(model, 'test'))

        cval_dict = {}

        if len(cvals)>0:
            for met_name in cvals[0].keys():
                if met_name not in cval_dict.keys(): cval_dict[met_name] = []
                for cid in range(len(cvals)):
                    cval_dict[met_name].append(cvals[cid][met_name])

                if type(cvals[cid][met_name]) != list:
                    self.output['val_'+met_name].append(float(np.array(cval_dict[met_name]).mean()))
                else:
                    tmp_class_accs = np.zeros((len(cvals),len(cval_dict[met_name][0]))) 
                    for i, value in enumerate(cval_dict[met_name]):
                        tmp_class_accs[i] = np.array(value)
                    self.output['val_'+met_name].append([tmp_class_accs.mean(axis=0)])

        ctest_dict = {}
        if len(ctests)>0:
            for met_name in ctests[0].keys():
                if met_name not in ctest_dict.keys(): ctest_dict[met_name] = []
                for cid in range(len(ctests)):
                    ctest_dict[met_name].append(ctests[cid][met_name])

                if type(cvals[cid][met_name]) != list:
                    self.output['test_'+met_name].append(float(np.array(ctest_dict[met_name]).mean()))
                else:
                    tmp_class_accs = np.zeros((len(cvals),len(ctest_dict[met_name][0]))) 
                    for i, value in enumerate(ctest_dict[met_name]):
                        tmp_class_accs[i] = np.array(value)
                    self.output['test_'+met_name].append([tmp_class_accs.mean(axis=0)])
            
            if self.output['val_f1_macro'][-1] > self.best_val:
                path = self.option['path']
                path = path + '/model_state/' + self.option['a']
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(self.server.model.state_dict(), path + f'/server_{model_name}.pt')
                self.best_val = self.output['val_f1_macro'][-1]
                self.best_test = self.output['test_f1_macro'][-1]
                self.output['best_val_f1'] = self.best_val
                self.output['best_test_f1'] = self.best_test

        self.show_current_output()