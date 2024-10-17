import flgo
import os
import torch.multiprocessing
from flgo.utils.misc import *
from flgo.utils.fflow import read_option_from_command
from flgo.experiment.logger.vc_logger import VCLogger
from flgo.experiment.analyzer import show
import importlib
import flgo.experiment.device_scheduler as ds
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    args = read_option_from_command()

    benchmark_name = args['b']
    dataset_name = benchmark_name.split('_')[0]
    vd_or_vc = benchmark_name.split('_')[-1]
    task_name = args['t']
    algo_name = args['a']
    model_name = args['m']

    bmkname = f'flgo/benchmark/{dataset_name}_{model_name}_{vd_or_vc}'

    bmkname = bmkname.replace('/', '.')
    task = f'./save/{dataset_name}_{vd_or_vc}/{task_name}-{model_name}'
    task_config = {
        'benchmark':{'name': bmkname},
        'partitioner':{'name': 'DiversityPartitioner','para':{'num_clients':10, 'diversity':1.0, "index_func": lambda X:[xi['cwe_vc'] for xi in X]}}
    } 
   
    if not os.path.exists(task): flgo.gen_task(task_config, task_path=task)


    option = {}
    option['path'] = task

    algo = importlib.import_module(f'flgo.algorithm.{algo_name}')
    tmp_algo_option = load_algor_configs(algo_name)
    algor_option = copy.deepcopy(tmp_algo_option)
    
    model = importlib.import_module(f'flgo.benchmark.{benchmark_name}.model.{model_name}')
    task_tmp = vd_or_vc+'_'+benchmark_name[:3]
    tmp_model_option = load_model_configs(model_name, task = task_tmp)
    model_option = copy.deepcopy(tmp_model_option)
    
    option.update(algor_option)
    option.update(model_option) 

    torch.multiprocessing.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
  
    runner = flgo.init(task, algo, option, model=model, Logger=VCLogger)
    runner.run()
