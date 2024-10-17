import yaml
import os
import importlib
import copy

def record_basic_info(logging, args): 
    logging.info("===============================================") 
    max_len = 0
    for key, value in args.__dict__.items(): 
        if max_len < len(key):
            max_len = len(key)
    for key, value in args.__dict__.items(): 
        logging.info(key+(max_len-len(key))*' '+": "+str(value)) 

# load_option_congigs

def load_algor_configs(name = 'base'):
    default_option_t = importlib.import_module(f'configs.option.algorithm.base').config
    default_option = copy.deepcopy(default_option_t)
    if os.path.exists(f'./configs/option/algorithm/{name}.py'):
        option_t = importlib.import_module(f'configs.option.algorithm.{name}').config
        option = copy.deepcopy(option_t)
        for op_key in option:
            if op_key in default_option.keys():
                default_option[op_key] = option[op_key]
            else:
                default_option[op_key] = option[op_key]
    return default_option


def load_model_configs(name = 'base', task = 'vc_big'):
    default_option_t = importlib.import_module(f'configs.option.model.{task}.base').config
    default_option = copy.deepcopy(default_option_t)
    if os.path.exists(f'./configs/option/model/{task}/{name}.py'):
        option_t = importlib.import_module(f'configs.option.model.{task}.{name}').config
        option = copy.deepcopy(option_t)
        for op_key in option:
            if op_key in default_option.keys():
                default_option[op_key] = option[op_key]
            else:
                default_option[op_key] = option[op_key]
    return default_option

def load_tune_configs(name = 'base'):
    default_option_t = importlib.import_module(f'configs.option.algorithm.base').config
    default_option = copy.deepcopy(default_option_t)
    if os.path.exists(f'./configs/option/tune/{name}.py'):
        option_t = importlib.import_module(f'configs.option.tune.{name}').config
        option = copy.deepcopy(option_t)
        for op_key in option:
            if op_key in default_option.keys():
                default_option[op_key] = option[op_key]
            else:
                default_option[op_key] = option[op_key]
    return default_option