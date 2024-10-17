import pickle
import random
import re
import time
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore", category=Warning)



class VLGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pro=False):
        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):

        return []

    @property
    def processed_file_names(self):
        return [f'{self.process_dataset_name}_graph.pt']

    def download(self):
        pass

    def process(self):

        with open(f'./data/{self.process_dataset_name}/pyg_list/data.bin', 'rb') as f:
            data_list = pickle.load(f)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list) 

        torch.save((data, slices), self.processed_paths[0])
