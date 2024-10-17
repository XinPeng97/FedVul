"""
This is a non-official implementation of personalized FL method FedProto (https://ojs.aaai.org/index.php/AAAI/article/view/20819).
The original implementation is at https://github.com/yuetan031/FedProto
"""
import collections
import copy
import random
import torch
import torch.utils.data.dataset
import torch.nn as nn
from flgo.algorithm.fedbase import BasicServer, BasicClient
import flgo.utils.fmodule as fmodule
import numpy as np



class Server(BasicServer):
    def initialize(self):
        self.init_algo_para({'alpha':0.1,'beta':0.1})

        self.num_classes = len(collections.Counter([d['target'] for d in self.test_data]))
        self.sample_option = 'full'
        with torch.no_grad():
            dataloader = self.calculator.get_dataloader(self.test_data, 1)
            for batch_id, batch_data in enumerate(dataloader):
                x = batch_data[0].to(self.device)
                self.model.to(self.device)
                h = self.model.encoder(x)
                self.dim = h.shape[-1]

                break
        for c in self.clients:
            c.num_classes = self.num_classes
            c.dim = self.dim

        self.c = torch.zeros((self.num_classes, self.dim))
    def pack(self, client_id, mtype=0):
        return {
            "model": copy.deepcopy(self.model),
            'c': copy.deepcopy(self.c),
        }
    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        models, cs, sizes_label = res['model'], res['c'], res['sizes_label']
        self.model, self.c = self.aggregate(models, cs, sizes_label)
        return len(models) > 0
    
    def aggregate(self, models: list, cs:list, sizes_label:list):
        if len(models) == 0: return self.model
        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)
        p = [1.0 * local_data_vols[cid] / total_data_vol for cid in self.received_clients]
        sump = sum(p)
        p = [pk / sump for pk in p]
        
        # c
        if len(cs)==0: return self.c
        num_samples = np.sum(sizes_label, axis=0) 
        num_samples = num_samples.reshape(self.num_classes, 1)

        new_c = torch.zeros((self.num_classes, self.dim))
        
        for ci, i, si in zip(cs, self.received_clients, sizes_label):
            ci = ci.to('cpu')
            si=si.reshape(self.num_classes, 1)
            new_c += ci*si/num_samples

        return fmodule._model_sum([model_k * pk for model_k, pk in zip(models, p)]), new_c

class Client(BasicClient):
    def initialize(self):
        label_counter = collections.Counter([d['target'] for d in self.train_data])
        self.sizes_label = np.zeros(self.num_classes) # (10 * 0)
        for lb in range(self.num_classes):
            if lb in label_counter.keys(): 
                self.sizes_label[lb] = label_counter[lb]
        self.probs_label = self.sizes_label/self.sizes_label.sum()

        self.label_list_np = np.arange(self.num_classes)

    def reply(self, svr_pkg):
        model, cg = self.unpack(svr_pkg)
        cg = cg.to(self.device)
        model = model.to(self.device)
        self.train(model, cg)
        return self.pack()

    def unpack(self, svr_pkg):
        return svr_pkg['model'], svr_pkg['c']

    def pack(self):
        # c = {}
        label_all = torch.arange(self.num_classes, device=self.device)
        
        c = torch.zeros((self.num_classes, self.dim), device=self.device)
        self.model.to(self.device)
        with torch.no_grad():
            dataloader = self.calculator.get_dataloader(self.train_data, self.batch_size)
            for batch_id, batch_data in enumerate(dataloader):
                batch_data = self.calculator.to_device(batch_data)
                protos = self.model.encoder(batch_data[0]).detach()
                labels = batch_data[-1]

                label_batch_one = torch.unique(labels)
                for j in label_batch_one:
                    matrix = protos[torch.isin(labels, j)]
                    query = label_all == j
                    indices = torch.where(query)[0]
                    c[indices.item()] = matrix.sum(dim=0).data

            for j in range(len(self.sizes_label)):
                if self.sizes_label[j]==0: continue
                c[j]/=self.sizes_label[j]

        self.model = self.model.to('cpu')
        c = c.to('cpu')

        return {'model': self.model, 'c': c, 'sizes_label': self.sizes_label}

    @fmodule.with_multi_gpus
    def train(self, model, cg): 
        src_model = copy.deepcopy(model)
        src_model.freeze_grad()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, momentum=self.momentum, weight_decay=self.weight_decay)
        for iter in range(self.num_steps):
            model.zero_grad()
            batch_data = self.calculator.to_device(self.get_batch_data())
            protos = model.encoder(batch_data[0]) # 维度是512
            labels = batch_data[-1]
            outputs = model.head(protos)
            loss_ce = model.pre_label_loss(outputs, labels)

            loss_reg = 0
            for pm, ps in zip(model.parameters(), src_model.parameters()):
                loss_reg += torch.sum((pm - ps)**2)


            protos_pos = copy.deepcopy(protos.data) # [batch, hidden]
            protos_neg = copy.deepcopy(protos.data)

            if torch.all(cg != 0.):
                protos_pos[:,:] = cg[labels,:].data[:,:]

                length = len(labels)
                if self.num_classes == 2:
                    
                    different_list = 1 - labels
                    different_list = different_list.cpu().numpy()
                else:
                    label_train = labels.cpu().numpy()
                    array_a = np.random.randint(low=1, high=self.num_classes-1, size=(len(label_train))) 
                    different_list = (label_train+array_a)%(self.num_classes)

                protos_neg[:,:] = cg[different_list,:].data[:,:]

                loss_con = model.con_loss(protos, protos_pos, protos_neg)
            else:
                loss_con = 0.
            
            loss = loss_ce + self.alpha * loss_reg + self.beta * loss_con

            loss.backward()
            optimizer.step()

        self.model = copy.deepcopy(model).to(torch.device('cpu'))
        self.model.freeze_grad()        
        return

