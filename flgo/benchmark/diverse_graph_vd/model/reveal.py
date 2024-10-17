from flgo.utils.fmodule import FModule
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv,GCNConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
import importlib
import copy
import os
name = name=os.path.basename(__file__).split(".")[0]
default_option_t = importlib.import_module(f'configs.option.model.vd_div.base').config
default_option = copy.deepcopy(default_option_t)
if os.path.exists(f'./configs/option/model/vd_div/{name}.py'):
    option_t = importlib.import_module(f'configs.option.model.vd_div.{name}').config
    option = copy.deepcopy(option_t)
    for op_key in option:
        if op_key in default_option.keys():
            default_option[op_key] = option[op_key]
        else:
            default_option[op_key] = option[op_key]
option = copy.deepcopy(default_option)

class Encoder(FModule):
    def __init__(self):
        super().__init__()
        input_dim, hidden_dim, out_dim = option['num_node_features'], option['node_hidden_dim'], option['out_dim']
        
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=6)
        self.extract_feature=nn.Linear(in_features=hidden_dim, out_features=out_dim)
        

    def forward(self, data):
        x, edge_index, edge_attr, batch, y = data.x, data.edge_index, data.edge_attr, data.batch, data.y

        outputs = self.ggnn(x, edge_index)
        
        pooled = global_mean_pool(outputs, batch)
        
        emb = self.extract_feature(pooled)
        return emb

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

        self.head = nn.Linear(option['out_dim'], option['num_class'])

        self.f_k = nn.Bilinear(option['out_dim'], option['out_dim'], 1)

        weight = torch.FloatTensor(option['loss_weight'])
        self.ce = torch.nn.CrossEntropyLoss(weight = weight)


    def compute_loss(self, data):
        emb = self.encoder(data)
        out = self.head(emb)        
        loss = self.ce(out, data.y)
        return loss
    def forward(self, data):
        emb = self.encoder(data)
        out = self.head(emb)
        return out

    def pre_label_loss(self, pre, label):
        loss = self.ce(pre, label)
        return loss
    
    def con_loss(self, protos, protos_pos, protos_neg):
        sc_1 = self.f_k(protos, protos_pos)
        sc_2 = self.f_k(protos, protos_neg)        
        logits = torch.cat((sc_1, sc_2), 1)
        loss = F.cross_entropy(logits, torch.zeros(logits.size(0), device=logits.device).long())
        return loss



def init_local_module(object, option):
    pass

def init_global_module(object, option):
    if 'Server' in object.__class__.__name__:
        object.model = Model().to(object.device)