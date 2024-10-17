from typing import Any
from collections.abc import Callable
import flgo.benchmark.base
from torch.utils.data import random_split, Subset
# from torchtext.data.functional import to_map_style_dataset
from flgo.benchmark.base import BasicTaskCalculator, BasicTaskGenerator, BasicTaskPipe
import os
try:
    import ujson as json
except:
    import json
import torch

FromDatasetGenerator = flgo.benchmark.base.FromDatasetGenerator


class FromDatasetPipe(flgo.benchmark.base.FromDatasetPipe):
    TaskDataset = Subset

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'rawdata_path': generator.rawdata_path, 'additional_option': generator.additional_option, 'train_additional_option':generator.train_additional_option, 'test_additional_option':generator.test_additional_option, }
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid],}
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            json.dump(feddata, outf)
        return
    
    def load_data(self, running_time_option) -> dict:
        train_data = self.train_data
        test_data = self.test_data
        val_data = self.val_data

        task_data = {'server': {'test': test_data, 'val': val_data}}
        for cid, cname in enumerate(self.feddata['client_names']):
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'])
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            task_data[cname] = {'train':cdata_train, 'val':val_data, 'test': self.test_data}
        return task_data  
          

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
class GeneralCalculator(BasicTaskCalculator):
    r"""
    Calculator for the dataset in torchvision.datasets.

    Args:
        device (torch.device): device
        optimizer_name (str): the name of the optimizer
    """
    def __init__(self, device, optimizer_name='sgd'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.DataLoader = torch.utils.data.DataLoader
        self.collate_fn = None
        
    def compute_loss(self, model, data):
        """
        Args:
            model: the model to train
            data: the training dataset
        Returns: dict of train-one-step's result, which should at least contains the key 'loss'
        """

        text, label = self.to_device(data)
        loss = model.compute_loss(text, label)
        
        # outputs = model(text)
        # loss = self.criterion(outputs, label)
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
        # cg = self.to_device(cg)
        cg = cg.to(self.device)
        for batch_id, batch_data in enumerate(data_loader):
            num_batch = batch_data[0]['input_ids'].size()[0]
            batch_data = self.to_device(batch_data)
            proto = model.encoder(batch_data[0])
            a_large_num = 100
            num_class = model.head.out_features
        
            dist = a_large_num * torch.ones(size=(num_batch, num_class))  # initialize a distance matrix
            # dist = self.to_device(dist)
            dist = dist.to(self.device)
            for i in range(num_batch):
                for j in range(num_class):
                    d = loss_mse(proto[i, :], cg[j, :])
                    dist[i, j] = d
            _, pred_labels = torch.min(dist, 1)
            pred_labels = pred_labels.view(-1)

            batch_mean_loss = 0.0
            all_predictions.extend(pred_labels.detach().cpu().numpy().tolist())
            all_targets.extend(batch_data[1].detach().cpu().numpy().tolist())

            total_loss += batch_mean_loss * len(batch_data[1])
        acc = accuracy_score(all_targets, all_predictions) * 100
        pre = precision_score(all_targets, all_predictions) * 100
        recall = recall_score(all_targets, all_predictions) * 100
        f1 = f1_score(all_targets, all_predictions) * 100
        return {'accuracy': acc, 
                'precision': pre,
                'recall': recall,
                'f1_score': f1,
                'loss': total_loss / len(dataset)}
    
    @torch.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0, pin_memory=False):
        """
        Metric = [mean_accuracy, mean_loss]

        Args:
            model:
            dataset:
            batch_size:
        Returns: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
        total_loss = 0.0
        num_correct = 0
        all_predictions, all_targets = [], []
        for batch_id, batch_data in enumerate(data_loader):
            batch_data = self.to_device(batch_data)
            outputs = model(batch_data[0])
            batch_mean_loss = self.criterion(outputs, batch_data[1]).item()
            # y_pred = outputs.data.max(1, keepdim=True)[1]
            # correct = y_pred.eq(batch_data[-1].data.view_as(y_pred)).long().cpu().sum()

            predictions = outputs.detach().cpu()
            all_predictions.extend(
                np.argmax(predictions.numpy(), axis=-1).tolist())
            all_targets.extend(batch_data[1].detach().cpu().numpy().tolist())

            # num_correct += (outputs.argmax(1)==batch_data[1]).sum().item()
            total_loss += batch_mean_loss * len(batch_data[0])

        acc = accuracy_score(all_targets, all_predictions) * 100
        pre = precision_score(all_targets, all_predictions) * 100
        recall = recall_score(all_targets, all_predictions) * 100
        f1 = f1_score(all_targets, all_predictions) * 100

        return {'accuracy': acc, 
                'precision': pre,
                'recall': recall,
                'f1_score': f1,
                'loss': total_loss / len(dataset)}
    def to_device(self, data):
        res = []
        for i in range(len(data)):
            if isinstance(data[i], torch.Tensor):
                di = data[i].to(self.device)
            elif isinstance(data[i], list):
                di = [d.to(self.device) for d in data[i]]
            else:
                raise TypeError('data should be either of type list or torch.Tensor')
            res.append(di)
        return tuple(res)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False, drop_last=False, *args, **kwargs):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        # print(self.collate_fn)
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last, collate_fn=self.collate_fn)
