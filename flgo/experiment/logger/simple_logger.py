from flgo.experiment.logger import BasicLogger
import numpy as np
import flgo.simulator.base as ss

class SimpleLogger(BasicLogger):
    r"""Simple Logger. Only evaluating model performance on testing dataset and validation dataset."""
    def initialize(self):
        """This method is used to record the stastic variables that won't change across rounds (e.g. local_movielens_recommendation data size)"""
        for c in self.participants:
            self.output['client_datavol'].append(len(c.train_data))

    """This logger only records metrics on validation dataset"""
    def log_once(self, *args, **kwargs):
        self.info('Current_time:{}'.format(self.clock.current_time))
        self.output['time'].append(self.clock.current_time)
        test_metric = self.coordinator.test() # 中心服务器  测试集的ACC
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        val_metrics = self.coordinator.global_test(flag='val') # 验证集是每个客户端都去验证
        # val_metrics = self.coordinator.global_test(flag='test')
        local_data_vols = [c.datavol for c in self.participants] # 每个客户端分配到样本的数量
        total_data_vol = sum(local_data_vols) # 数据总量
        for met_name, met_val in val_metrics.items():
            self.output['val_'+met_name+'_dist'].append(met_val)
            self.output['val_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.output['mean_val_' + met_name].append(np.mean(met_val))
            self.output['std_val_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()