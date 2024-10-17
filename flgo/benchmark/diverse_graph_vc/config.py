import os
import random
import flgo.benchmark
import torch.nn as nn
import torch_geometric
import numpy as np
from utils.GraphDatasetPYG import VLGraphDataset
from configs.option.data.base import config
from sklearn.model_selection import train_test_split

all_data = VLGraphDataset(root=f'./data/Diversevul')


data_index_all = np.arange(len(all_data))
all_idxs = data_index_all
train_ratio, val_ratio, test_ratio = config['train_val_test'][0], config['train_val_test'][1], config['train_val_test'][2]
train_idxs, temp_index = train_test_split(all_idxs, test_size=(1 - train_ratio))
val_idxs, test_idxs = train_test_split(temp_index, test_size=(test_ratio / (val_ratio + test_ratio)))


train_data = all_data[train_idxs]
val_data = all_data[val_idxs]
test_data = all_data[test_idxs]


class GCN(nn.Module):
    def __init__(self, *arg):
        super(GCN, self).__init__()
        pass
    def forward(self, data):
        return data

def get_model():
    return GCN()