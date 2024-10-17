import pickle
with open(f'./data/Diversevul/class_weights.pkl', 'rb') as f:
    loaded_list = pickle.load(f)
    
config = {
'num_node_features': 100,                  
'node_hidden_dim': 200,
'out_dim': 200,
'num_class': 38,
'loss_weight': loaded_list,
}