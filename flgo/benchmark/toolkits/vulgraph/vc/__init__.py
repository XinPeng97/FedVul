import torch
from torch_geometric.data import Data, DataLoader
from flgo.benchmark.toolkits.graph.node_classification import BuiltinClassGenerator as NodeClassGenerator
import flgo.benchmark.base
import os
try:
    import ujson as json
except:
    import json
from flgo.benchmark.base import BasicTaskPipe, BasicTaskCalculator
import random
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np

FromDatasetGenerator = flgo.benchmark.base.FromDatasetGenerator

class FromDatasetPipe(flgo.benchmark.base.FromDatasetPipe):
    class TaskDataset(torch.utils.data.Subset):
        def __init__(self, dataset, indices, pin_memory=False):
            super().__init__(dataset, indices)
            self.dataset = dataset
            self.indices = indices
            self.pin_memory = pin_memory

        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names,
                   'server_data': list(range(len(generator.test_data))),
                   }
        for cid in range(len(client_names)):
            feddata[client_names[cid]] = {'data': generator.local_datas[cid], }
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        train_data = self.train_data
        test_data = self.test_data
        val_data = self.val_data
        # rearrange data for server
        if val_data is None:
            server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        else:
            server_data_test = test_data
            server_data_val = val_data
        task_data = {'server': {'test': test_data, 'val': val_data}}
        # rearrange data for clients
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'], running_time_option['pin_memory'])
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['train_holdout']>0 and running_time_option['local_test']:
                cdata_val, cdata_test = self.split_dataset(cdata_val, 0.5)
            else:
                cdata_test = None
            # 这里是在每个训练集中划分验证集,占比为训练集中
            # 的0.1,然后再将验证集划分测试集,称为局部测试
            # task_data[cname] = {'train':cdata_train, 'val':cdata_val, 'test': cdata_test}
            task_data[cname] = {'train':cdata_train, 'val':val_data, 'test': self.test_data}
        return task_data

class BuiltinClassGenerator(NodeClassGenerator):
    def load_data(self):
        print('error!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        default_init_para = {'root': self.rawdata_path, 'download':self.download, 'train':True, 'transform':self.transform}
        default_init_para.update(self.additional_option)
        if 'kwargs' not in self.builtin_class.__init__.__annotations__:
            pop_key = [k for k in default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            for k in pop_key: default_init_para.pop(k)
        self.dataset = self.builtin_class(**default_init_para)
        self.num_samples = len(self.dataset)
        k = int(self.test_rate*self.num_samples)
        all_idxs = list(range(self.num_samples))
        random.shuffle(all_idxs)
        self.test_idxs = all_idxs[:k]
        self.train_idxs = all_idxs[k:]
        self.train_data = self.dataset[self.train_idxs]
        self.train_data = [(d, self.dataset[d].cwe_vc) for d in self.train_idxs]
        self.test_data = self.dataset[self.test_idxs]

    def partition(self):
        self.local_datas = self.partitioner(self.train_data)
        self.num_clients = len(self.local_datas)
        self.local_datas = [[self.train_data[idx][0] for idx in local_data] for local_data in self.local_datas]

class BuiltinClassPipe(BasicTaskPipe):
    def __init__(self, task_path, buildin_class, transform=None, pre_transform=None):
        super(BuiltinClassPipe, self).__init__(task_path)
        self.builtin_class = buildin_class
        self.pre_transform = pre_transform
        self.transform = transform

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names,
                   'server_data': list(range(len(generator.test_data))),
                   'rawdata_path': generator.rawdata_path,
                   'additional_option':generator.additional_option,
                   'test_idxs':generator.test_idxs,
                   'download': generator.download_data,
                   }
        for cid in range(len(client_names)):
            feddata[client_names[cid]] = {'data': generator.local_datas[cid], }
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        default_init_para = {'root': self.feddata['rawdata_path'], 'download':self.feddata['download'], 'train':True, 'transform':self.transform, 'pre_transform':self.pre_transform}
        default_init_para.update(self.feddata['additional_option'])
        if 'kwargs' not in self.builtin_class.__init__.__annotations__:
            pop_key = [k for k in default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            for k in pop_key: default_init_para.pop(k)
        # load the datasets
        dataset = self.builtin_class(**default_init_para)
        k = int(len(self.feddata['test_idxs'])*running_time_option['test_holdout'])
        val_idxs = self.feddata['test_idxs'][:k]
        test_idxs = self.feddata['test_idxs'][k:]
        server_data_test = dataset[test_idxs]
        server_data_val = dataset[val_idxs]
        task_data = {'server': {'test': server_data_test if len(test_idxs)>0 else None, 'val': server_data_val if len(val_idxs)>0 else None}}
        # rearrange data for clients
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata_all = self.feddata[cname]['data']
            ck = int(running_time_option['train_holdout']*len(cdata_all))
            cdata_val = cdata_all[:ck]
            cdata_train = cdata_all[ck:]
            if running_time_option['train_holdout'] > 0 and running_time_option['local_test']:
                ck2 = int(0.5*ck)
                cdata_test = cdata_val[ck2:]
                cdata_val = cdata_val[:ck2]
            else:
                cdata_test = []
            task_data[cname] = {'train': dataset[cdata_train] if len(cdata_train)>0 else None, 'val': dataset[cdata_val] if len(cdata_val)>0 else None, 'test': dataset[cdata_test] if len(cdata_test)>0 else None}
        return task_data

class GeneralCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='sgd'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.DataLoader = DataLoader

    def compute_loss(self, model, data):
        """
        Args: model: the model to train
                 data: the training dataset
        Returns: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        tdata = self.to_device(data)
        loss = model.compute_loss(tdata)
        # outputs = model(tdata)
        # loss = self.criterion(outputs, tdata.cwe_vc)
        # print('loss:', loss)
        
        return {'loss': loss}

    @torch.no_grad()
    def fedproto_test(self, model, cg, dataset, batch_size=64, num_workers=0, pin_memory=False):
        """
        Metric = [mean_accuracy, mean_loss]
        Args:
            dataset:
                 batch_size:
        Returns: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size == -1: batch_size = len(dataset)
        data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        total_loss = 0.0
        num_correct = 0
        loss_mse = torch.nn.MSELoss()
        all_predictions, all_targets = [], []
        cg = self.to_device(cg)
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.to_device(batch_data)
            proto = model.encoder(batch_data)
            a_large_num = 100
            num_class = model.head.out_features
        
            dist = a_large_num * torch.ones(size=(len(batch_data), num_class))  # initialize a distance matrix
            dist = self.to_device(dist)
            for i in range(len(batch_data)):
                for j in range(num_class):
                    d = loss_mse(proto[i, :], cg[j, :])
                    dist[i, j] = d
            _, pred_labels = torch.min(dist, 1)
            pred_labels = pred_labels.view(-1)

            batch_mean_loss = 0.0
            all_predictions.extend(pred_labels.detach().cpu().numpy().tolist())
            all_targets.extend(batch_data.cwe_vc.detach().cpu().numpy().tolist())

            total_loss += batch_mean_loss * len(batch_data.cwe_vc)
        acc = accuracy_score(all_targets, all_predictions) * 100
        f1_macro = f1_score(all_targets, all_predictions, average='macro') * 100
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted') * 100
        cm = confusion_matrix(all_targets, all_predictions)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        class_accs = []
        for i, cla_acc in enumerate(class_accuracies):
            class_accs.append(round((cla_acc*100), 2))

        return {'accuracy': round(acc, 2), 
                'f1_macro': round(f1_macro, 2),
                'f1_weighted': round(f1_weighted, 2),
                'loss': total_loss / len(dataset),
                'class_accs':class_accs
                }

    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0, pin_memory=False):
        """
        Metric = [mean_accuracy, mean_loss]
        Args:
            dataset:
                 batch_size:
        Returns: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size == -1: batch_size = len(dataset)
        data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        total_loss = 0.0
        num_correct = 0

        all_predictions, all_targets = [], []
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.to_device(batch_data)
            logits = model(batch_data)
            # print(logits.size(), batch_data.cwe_vc.size())
            batch_mean_loss = self.criterion(logits, batch_data.cwe_vc).item()

            predictions = logits.detach().cpu()
            all_predictions.extend(
                np.argmax(predictions.numpy(), axis=-1).tolist())
            all_targets.extend(batch_data.cwe_vc.detach().cpu().numpy().tolist())

            # y_pred = outputs.data.max(1, keepdim=True)[1]
            # correct = y_pred.eq(batch_data.cwe_vc.data.view_as(y_pred)).long().cpu().sum()
            # num_correct += correct.item()

            total_loss += batch_mean_loss * len(batch_data.cwe_vc)
        acc = accuracy_score(all_targets, all_predictions) * 100
        # pre = precision_score(all_targets, all_predictions) * 100
        # recall = recall_score(all_targets, all_predictions) * 100
        f1_macro = f1_score(all_targets, all_predictions, average='macro') * 100
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted') * 100
        # f1_micro = f1_score(all_targets, all_predictions, average='micro')
        cm = confusion_matrix(all_targets, all_predictions)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        class_accs = []
        for i, cla_acc in enumerate(class_accuracies):
            class_accs.append(round((cla_acc*100), 2))

        return {'accuracy': round(acc, 2), 
                'f1_macro': round(f1_macro, 2),
                'f1_weighted': round(f1_weighted, 2),
                'loss': total_loss / len(dataset),
                'class_accs':class_accs
                }


    def to_device(self, data):
        return data.to(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False, *args, **kwargs):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)