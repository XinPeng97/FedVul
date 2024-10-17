from torch.utils.data import Dataset
import json
import numpy as np
class VLTokenDataset(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for idx, line in enumerate(data):
                Data[idx] = line
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, list) or isinstance(idx, np.array):
            return [self.data[i] for i in idx]
        return self.data[idx]