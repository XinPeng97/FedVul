import os
import torch
from flgo.experiment.logger import BasicLogger
import numpy as np

class VDLogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        self.set_es_key('local_test_f1')
        self.turn_es_direction()
    
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
                self.output['val_'+met_name].append(float(np.array(cval_dict[met_name]).mean()))

        ctest_dict = {}
        if len(ctests)>0:
            for met_name in ctests[0].keys():
                if met_name not in ctest_dict.keys(): ctest_dict[met_name] = []
                for cid in range(len(ctests)):
                    ctest_dict[met_name].append(ctests[cid][met_name])
                self.output['test_'+met_name].append(float(np.array(ctest_dict[met_name]).mean()))

            if self.output['val_f1_score'][-1] > self.best_val:
                path = self.option['path']
                path = path + '/model_state/' + self.option['a']
                if not os.path.exists(path):
                    os.makedirs(path)
                torch.save(self.server.model.state_dict(), path + f'/server_{model_name}.pt')
                self.best_val = self.output['val_f1_score'][-1]
                self.best_test = self.output['test_f1_score'][-1]
                self.output['best_val_f1'] = self.best_val
                self.output['best_test_f1'] = self.best_test

        self.show_current_output()