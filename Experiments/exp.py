from Data_Provider.Site_Provider import get_site_dataloader
from Data_Provider.Meteo_Provider import get_meteo_dataloader
from Utils.Tools import EarlyStopping, adjust_learning_rate
from Utils.MetricsCalculator import Metrics_Calculator, BatchWelford_R, BatchNME_Calculator
from Utils.Dir import get_project_root

from Models.AutoFormer import Model as AutoFormer
from Models.TimeMixer import Model as TimeMixer
from Models.TimesNet import Model as TimesNet
from Models.LSTM_Family import Model as LSTM
from Models.TriPFormer import Model as TriPFormer
from Models.MoHE import Model as MoHETransformer
from Models.SwitchTransformer import Model as SwitchTransformer

import torch
import json
import copy
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import collections

from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from pathlib import Path
from datetime import date, datetime, timedelta
from torch.optim import lr_scheduler


warnings.filterwarnings("ignore")  # forbit all warning information
project_dir = get_project_root()


class ExpMetrics:
    def __init__(self, target, epochs=0):
        self.calculator = Metrics_Calculator(target)
        self.target = target
        self.epochs = epochs
        if epochs < 1:
            self.datas = {
                'mse_loss': torch.zeros(1),
                'mse_per_feature': torch.zeros(len(target)),

                'mae_loss': torch.zeros(1),
                'mae_per_feature': torch.zeros(len(target)),

                'rmse_loss': torch.zeros(1),
                'rmse_per_feature': torch.zeros(len(target)),

                'ps_loss': torch.zeros(1),
                'ps_per_feature': torch.zeros(len(target)),

                'tops_10': torch.zeros(len(target)),
                'tops_30': torch.zeros(len(target)),
            }
        else:
            self.datas = {
                'mse_loss': torch.zeros(epochs),
                'mse_per_feature': torch.zeros(epochs, len(target)),

                'mae_loss': torch.zeros(epochs),
                'mae_per_feature': torch.zeros(epochs, len(target)),

                'rmse_loss': torch.zeros(epochs),
                'rmse_per_feature': torch.zeros(epochs, len(target)),

                'ps_loss': torch.zeros(epochs),
                'ps_per_feature': torch.zeros(epochs, len(target)),

                'tops_10': torch.zeros(epochs, len(target)),
                'tops_30': torch.zeros(epochs, len(target)),
            }

    def update_data(self, value, **kwargs):
        if value is None:
            return
        for key, val in value.items():
            if key not in self.datas.keys():
                self.add_key(key_id=key)
            if self.epochs >= 1 and 'index' in kwargs:
                self.datas[key][kwargs['index']] = val
            else:
                self.datas[key] = val

    def show(self, index=None):
        if index is None:
            for item in self.datas.keys():
                print(f'{item}: {self.datas[item]}')
        else:
            for item in self.datas.keys():
                print(f'{item}: {self.datas[item][index]}')

    def save_log(self, log_save_pth):
        if self.epochs >= 1:
            with open(log_save_pth, 'w') as f:
                for epoch in range(self.epochs):
                    f.write('---***************************---\n')
                    for item in self.datas:
                        if len(self.datas[item].shape) == 2:
                            for i, pollutant in enumerate(self.target):
                                f.write(f'epoch: {epoch}, item: {pollutant}_{item}, value: {self.datas[item][epoch, i].item()}\n')
                        else:
                            f.write(f'epoch: {epoch}, item: {item}, value: {self.datas[item][epoch].item()} \n')
                    f.write('---***************************---\n\n')
        else:
            pass

    def plot(self, fig_save_pth):
        Path(fig_save_pth).mkdir(parents=True, exist_ok=True)
        for item in self.datas.keys():
            if len(self.datas[item].shape) == 2:
                for i, pollutant in enumerate(self.target):
                    fig_name = f'{pollutant}_{item}.png'
                    item_save_pth = Path(fig_save_pth) / fig_name
                    target_array = self.datas[item][:, i]
                    plt.figure(figsize=(10, 6))
                    plt.plot(target_array)
                    plt.xlabel('epochs')
                    plt.ylabel(f'{item}')
                    plt.title(f'{pollutant}_{item}')
                    plt.grid(True)
                    plt.savefig(item_save_pth)
                    plt.close()

            else:
                fig_name = f'{item}.png'
                item_save_pth = Path(fig_save_pth) / fig_name
                target_array = self.datas[item]
                plt.figure(figsize=(10, 6))
                plt.plot(target_array)
                plt.xlabel('epochs')
                plt.ylabel(f'{item}')
                plt.title(f'{item}')
                plt.grid(True)
                plt.savefig(item_save_pth)
                plt.close()

    def calculate(self, predict, gt, index=0):
        result = self.calculator(predict, gt)
        if result is None:
            return 0
        else:
            if self.epochs >= 1:
                self.update_data(result, index=index)
            else:
                self.update_data(result)
            return 1

    def get_data(self, index):
        result = {}
        for item, val in self.datas.items():
            result[item] = val[index]
        return result

    def __deepcopy__(self, memo):
        new_metrics = ExpMetrics(copy.deepcopy(self.target, memo), self.epochs)
        new_datas = {}
        for item in self.datas.keys():
            new_datas[item] = self.datas[item].clone()
        new_metrics.datas = new_datas
        return new_metrics

    # 重载 + 运算符
    def __add__(self, other):
        result_metrics = ExpMetrics(self.target, self.epochs)
        for item in self.datas.keys():
            result_metrics.datas[item] = self.datas[item] + other.datas[item]
        return result_metrics

    # 重载 / 运算符
    def __truediv__(self, _fractor):
        result_metrics = ExpMetrics(self.target, self.epochs)
        for item in self.datas.keys():
            result_metrics.datas[item] = self.datas[item] / _fractor
        return result_metrics

    # 重载反向加法
    def __radd__(self, other):
        if other == 0:  # sum() 从 0 开始累加
            return self
        else:
            return self.__add__(other)

    # 为data添加新的键
    def add_key(self, key_id):
        self.datas[key_id] = torch.empty(0)

    # 求均值
    @ property
    def mean(self):
        result = {}
        for item, val in self.datas.items():
            result[item] = torch.mean(val, dim=0).detach().cpu()
        return result

    @ property
    def cpu(self):
        result = {}
        for item, val in self.datas.items():
            result[item] = val.detach().cpu()
        return result


class Exp:
    def __init__(self, args):
        # 1. 获取实验设置
        self.args = args

        # 2. 获取支持的模型
        self.supported_models = {

        }

        # 3. 解包实验配置
        self.target = self.args.target  # 目标污染物
        self.device = self._acquire_device()  # 使用的计算硬件

        self.model_type = self.args.model_type  # 模型类型
        self.model_supported = {
            'AutoFormer': AutoFormer,
            'TimeMixer': TimeMixer,
            'TimesNet': TimesNet,
            'LSTM': LSTM,
            'TriPFormer': TriPFormer,
            'MoHETransformer': MoHETransformer,
            'SwitchTransformer': SwitchTransformer
        }
        self.model = self.model_supported[self.args.model_type](self.args).float() # 初始化一个模型
        self.model_path = self.args.model_path  # 使用已存在的模型，允许该项为None

        self.train_epochs = self.args.train_epochs  # 训练主干网络的默认epoch数
        self.lr = self.args.learning_rate  # 训练主干网络的学习率
        self.mission = self.args.mission  # 本次实验的任务类型{train, test, predict(仅限使用meteo数据集)}
        self.dataset_type = self.args.dataset_type  # 采用的数据源
        self.checkpoints_path = project_dir / 'checkpoints'  # 每个epoch得到的模型存放的路径
        self.seq_len = self.args.seq_len  # pl
        self.pred_len = self.args.pred_len  # fl
        self.d_model = self.args.d_model if hasattr(self.args, 'd_model') else 0  # 多头注意力每个头的维度
        self.d_ff = self.args.d_ff if hasattr(self.args, 'd_ff') else 0 # ffn模块中间层的维度
        self.loss_func = self.args.loss_func  # 损失函数
        self.dataset_norm = self.args.dataset_norm  # 是否使用数据集归一化

        if self.args.use_multi_gpu and self.args.use_gpu:
            self.model = nn.DataParallel(self.model, device_ids=self.args.device_ids)
            self.model = self.model.to(self.args.device_ids[0])  # 主设备设为 cuda:0
        else:
            self.model = self.model.to(self.device)

        # 4. 注册实验名称
        target_str = '_'.join(args.target)
        date_str = date.today().strftime("%Y-%m-%d") if args.date == "None" else args.date
        self.setting = "{}_{}_pl{}_fl{}_{}_{}".format(
            target_str,
            self.model_type,
            self.seq_len,
            self.pred_len,
            self.loss_func,
            date_str
        )

        # 5. 指标
        self.calculator = Metrics_Calculator(self.target)

        # 5. 获取数据集
        self._get_data()

    def _load_model(self, model_path: Path):
        model = None
        try:
            model = torch.load(model_path, weights_only=False)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Failed to load model: {e}")
            return None  # 提前返回

        # 如果是 state_dict
        if isinstance(model, collections.OrderedDict):
            new_model = self.model_supported[self.model_type](self.args).float()
            model_dict = new_model.state_dict()

            # 找到匹配的参数
            pretrained_dict = {k: v for k, v in model.items() if k in model_dict and v.shape == model_dict[k].shape}

            # 更新参数
            model_dict.update(pretrained_dict)

            # 找到未被加载的参数
            missing_keys = [k for k in model_dict.keys() if k not in pretrained_dict]

            if len(missing_keys) > 0:
                warnings.warn(
                    f"{len(missing_keys)} parameters in model were not initialized from checkpoint:\n"
                    + "\n".join(missing_keys)
                )

            new_model.load_state_dict(model_dict, strict=False)
            model = new_model.to(self.device)

        # 设置模型目标
        print(f'成功加载模型：{model_path}')
        model.target = self.args.target
        return model

    # 同时加载多个模型
    def load_multi_model(self, model_paths, model_types, model_args):
        load_model = []
        for i in range(len(model_paths)):
            try:
                model = torch.load(model_paths[i], weights_only=False)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"Failed to load model: {e}")
                return None  # 提前返回
            new_model = self.model_supported[model_types[i]].TimeMixer(model_args[i]).float()
            model_dict = new_model.state_dict()

            # 普通模型：加载匹配的参数
            pretrained_dict = {k: v for k, v in model.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            new_model.load_state_dict(model_dict, strict=False)

            load_model.append(new_model)
            print(f'成功加载模型：{model_paths[i]}')
        return load_model

    def _get_data(self):
        # 根据实验任务获取数据集
        self.datasets = {}
        self.dataloaders = {}
        print(self.mission)
        if self.mission == 'train':
            self.datasets["train_dataset"], self.dataloaders['train_dataloader'],\
            self.datasets["vali_dataset"], self.dataloaders["vali_dataloader"] = get_site_dataloader(self.args) \
                if self.dataset_type == "site" else get_meteo_dataloader(self.args)
        elif self.mission == 'test' or self.mission == 'test_p2p' or self.mission == 'test_importance':
            self.datasets['test_dataset'], self.dataloaders['test_dataloader'] = get_site_dataloader(self.args) \
                if self.dataset_type == "site" else get_meteo_dataloader(self.args)
        elif self.mission == 'predict':
            self.datasets['predict_dataset'], self.dataloaders['predict_dataloader'] = get_site_dataloader(self.args) \
                if self.dataset_type == "site" else get_meteo_dataloader(self.args)
        else:
            raise ValueError('Mission should be "train", "test", "test_p2p" or "predict".')

    def _acquire_device(self):
        if self.args.use_gpu:
            available_gpus = []
            for i in range(torch.cuda.device_count()):
                try:
                    # 检查每个GPU的可用性
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                    # 简单测试
                    x = torch.tensor([1.0], device=f'cuda:{i}')
                    _ = x * 2
                    available_gpus.append(i)
                except:
                    print(f'GPU {i} 不可用')

            try:
                # 设置CUDA可见设备
                if not self.args.use_multi_gpu:
                    # 单GPU模式
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.gpu)
                    device = torch.device(f'cuda:{self.args.gpu}')
                    print(f'使用GPU: cuda:{self.args.gpu}')
                else:
                    # 多GPU模式
                    if hasattr(self.args, 'devices') and self.args.devices:
                        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.devices
                        # 在多GPU模式下，通常使用第一个可见的GPU作为主设备
                        device = torch.device('cuda:0')
                        print(f'使用多GPU: {self.args.devices}, 主设备: cuda:0')
                    else:
                        # 如果没有指定具体设备，使用所有可用GPU
                        num_gpus = torch.cuda.device_count()
                        visible_devices = ','.join(str(i) for i in range(num_gpus))
                        os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
                        device = torch.device('cuda:0')
                        print(f'使用所有GPU: {visible_devices}, 主设备: cuda:0')

                # 验证GPU是否真的可用
                test_tensor = torch.tensor([1.0]).to(device)
                return device

            except (RuntimeError, AssertionError) as e:
                print(f'GPU设置失败: {e}，回退到CPU')
                os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 清除GPU设置
                return torch.device('cpu')
        else:
            print('使用CPU')
            return torch.device('cpu')

    def _select_optimizer(self, learning_rate):
        model_optim = optim.Adam(self.model.parameters(), lr=learning_rate)
        return model_optim

    def print_params(self):
        total_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"{name} | {param.numel():,}")
                total_params += param.numel()
        print(f'total_params: {total_params}')

    def backward(self, _metrics, _optim, _scaler, _loss_func):
        if self.args.use_amp:
            if self.args.loss_func == 'mse':
                _scaler.scale(_metrics['mse_loss']).backward()
            elif self.args.loss_func == 'rmse':
                _scaler.scale(_metrics['rmse_loss']).backward()
            elif self.args.loss_func == 'ps':
                _scaler.scale(_metrics['ps_loss']).backward()
            else:
                raise ValueError
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1000)
            _scaler.step(_optim)
            _scaler.update()
        else:
            if self.args.loss_func == 'mse':
                _metrics['mse_loss'].backward()
            elif self.args.loss_func == 'rmse':
                _metrics['rmse_loss'].backward()
            elif self.args.loss_func == 'ps':
                _metrics['ps_loss'].backward()
            else:
                raise _metrics.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1000)
            _optim.step()

    def train(self):
        print(f'start training: {self.dataset_type}_{self.setting}')
        checkpoint_save_pth = self.checkpoints_path / self.model_type / self.setting  # 保存checkpoint的路径
        Path(checkpoint_save_pth).mkdir(parents=True, exist_ok=True)
        train_dataset, train_dataloader = self.datasets['train_dataset'], self.dataloaders['train_dataloader']

        train_steps = len(train_dataloader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        train_metrics, batch_metrics, current_metrics = ExpMetrics(self.target, self.train_epochs), ExpMetrics(self.target, len(train_dataloader)), ExpMetrics(self.target)

        # 获取待测试模型
        if self.model_path is not None:
            self.model = self._load_model(self.model_path)

        self.model = self.model.to(self.device)

        # 开始训练
        model_optim = self._select_optimizer(self.lr)

        # scheduler = lr_scheduler.OneCycleLR(
        #     optimizer=model_optim,
        #     steps_per_epoch=train_steps,
        #     pct_start=self.args.pct_start,
        #     epochs=self.train_epochs,
        #     max_lr=self.lr
        # )

        # total_params = sum(p.numel() for p in self.model.parameters())
        # print(f'Total parameters: {total_params}')

        if self.args.dataset_norm:
            with open(str(project_dir / 'normalization_params.json'), 'r') as f:
                data = json.load(f)
                means = data['air_means']
                stds = data['air_stds']
            air_cols = train_dataset.air_cols
            indices = [air_cols.index(target) for target in self.args.target]
            target_means = torch.tensor([means[i] for i in indices], requires_grad=False).to(self.device)
            target_stds = torch.tensor([stds[i] for i in indices], requires_grad=False).to(self.device)
            target_means = target_means.unsqueeze(0).unsqueeze(-1)
            target_stds = target_stds.unsqueeze(0).unsqueeze(-1)

        for current_epoch in range(self.train_epochs):
            self.model.train()
            epoch_start_time = time.time()  # 当前epoch的开始时间

            for i, batch in enumerate(tqdm(train_dataloader)):
                model_optim.zero_grad()  # 清除上一次反向传播中的梯度
                batch_x, batch_label, batch_y, batch_x_stamp, batch_label_stamp, batch_air, batch_air_label, batch_static = batch
                batch_x = batch_x.float().to(self.device)
                batch_label = batch_label.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y.permute(0, 2, 1)
                batch_x_stamp = batch_x_stamp.float().to(self.device)
                batch_label_stamp = batch_label_stamp.float().to(self.device)
                batch_air = batch_air.float().to(self.device)
                batch_air_label = batch_air_label.float().to(self.device)
                batch_static = batch_static.float().to(self.device)

                # 获取模型预测值
                batch = batch_x, batch_label, batch_y, batch_x_stamp, batch_label_stamp, batch_air, batch_air_label, batch_static
                predict = self.model(batch)

                not_nan = current_metrics.calculate(predict, batch_y)
                predict = predict * target_stds + target_means if self.args.dataset_norm else predict
                batch_y = batch_y * target_stds + target_means if self.args.dataset_norm else batch_y
                denormalized = current_metrics.calculate(predict, batch_y)
                batch_metrics.update_data(current_metrics.cpu, index=i)

                # 反向传播
                if not_nan:
                    self.backward(
                            _metrics=current_metrics.datas,
                            _optim=model_optim,
                            _scaler=None,
                            _loss_func=self.loss_func
                    )
            adjust_learning_rate(model_optim, current_epoch + 1, self.args)

            print("Epoch: {} cost time: {}".format(current_epoch + 1, time.time() - epoch_start_time))
            # 记录当前epoch的评均train_metrics
            train_metrics.update_data(batch_metrics.mean, index=current_epoch)

            # 获取验证集
            vali_metrics = self.vali()

            print(f"Epoch: {current_epoch + 1}, Steps: {train_steps}")
            print('train_metrics:')
            train_metrics.show(current_epoch)

            torch.save(self.model.state_dict(), checkpoint_save_pth / f'checkpoint_{current_epoch}.pth')

            if self.args.loss_func == 'mse':
                early_stopping(vali_metrics.datas['mse_loss'], self.model, checkpoint_save_pth)
            elif self.args.loss_func == 'rmse':
                early_stopping(vali_metrics.datas['rmse_loss'], self.model, checkpoint_save_pth)
            elif self.args.loss_func == 'ps':
                early_stopping(vali_metrics.datas['ps_loss'], self.model, checkpoint_save_pth)
            else:
                raise ValueError

            if early_stopping.early_stop:
                print("Early stopping")
                break

        return train_metrics

    def vali(self):
        # 1. 获取验证集
        vali_dataset, vali_dataloader = self.datasets['vali_dataset'], self.dataloaders['vali_dataloader']

        # 2. 存放验证集的指标
        vali_metrics, batch_metrics = ExpMetrics(self.target), ExpMetrics(self.target, len(vali_dataloader))

        # 3. 开始验证
        self.model.eval()

        if self.dataset_norm:
            statistics_name = f'normalization_params.json'
            with open(str(project_dir / statistics_name), 'r') as f:
                data = json.load(f)
                means = data['air_means']
                stds = data['air_stds']
            air_cols = vali_dataset.air_cols
            indices = [air_cols.index(target) for target in self.args.target]
            target_means = torch.tensor([means[i] for i in indices], requires_grad=False).to(self.device)
            target_stds = torch.tensor([stds[i] for i in indices], requires_grad=False).to(self.device)
            target_means = target_means.unsqueeze(0).unsqueeze(-1)
            target_stds = target_stds.unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            for i, batch in enumerate(vali_dataloader):
                batch_x, batch_label, batch_y, batch_x_stamp, batch_label_stamp, batch_air, batch_air_label, batch_static = batch
                batch_x = batch_x.float().to(self.device)
                batch_label = batch_label.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y.permute(0, 2, 1)
                batch_x_stamp = batch_x_stamp.float().to(self.device)
                batch_label_stamp = batch_label_stamp.float().to(self.device)
                batch_air = batch_air.float().to(self.device)
                batch_air_label = batch_air_label.float().to(self.device)
                batch_static = batch_static.float().to(self.device)

                # 获取模型预测值
                batch = batch_x, batch_label, batch_y, batch_x_stamp, batch_label_stamp, batch_air, batch_air_label, batch_static
                predict = self.model(batch)

                # 获取真值
                if self.dataset_norm:
                    batch_y = batch_y * target_stds + target_means
                    predict = predict * target_stds + target_means

                # 计算当前批量的精度指标
                batch_metrics.calculate(predict, batch_y, i)

            vali_metrics.update_data(batch_metrics.mean)
            return vali_metrics

    def test(self):
        print(f'start testing: {self.dataset_type}_{self.setting}')
        # 测试结果保存路径
        result_path = project_dir / f'exp_{self.dataset_type}' / 'test_result' / self.model_type / self.setting
        result_path.mkdir(parents=True, exist_ok=True)

        # 测试集
        test_dataset, test_dataloader = self.datasets['test_dataset'], self.dataloaders['test_dataloader']

        # 测试集指标
        test_metrics, batch_metrics = ExpMetrics(self.target), ExpMetrics(self.target, len(test_dataloader))
        BR = BatchWelford_R(feature_num=len(self.target), device=self.device)
        BN = BatchNME_Calculator(feature_num=len(self.target), device=self.device)

        # 获取待测试模型
        if self.model_path is not None:
            self.model = self._load_model(self.model_path)
        else:
            self.model = self._load_model(project_dir / 'checkpoints' / self.model_type / self.setting / 'checkpoint.pth')
        self.model = self.model.to(self.device)

        # 开始测试
        self.model.eval()

        if self.dataset_norm:
            with open(str(project_dir / 'normalization_params.json'), 'r') as f:
                data = json.load(f)
                means = data['air_means']
                stds = data['air_stds']
            air_cols = test_dataset.air_cols
            indices = [air_cols.index(target) for target in self.args.target]
            target_means = torch.tensor([means[i] for i in indices], requires_grad=False).to(self.device)
            target_stds = torch.tensor([stds[i] for i in indices], requires_grad=False).to(self.device)
            target_means = target_means.unsqueeze(0).unsqueeze(-1)
            target_stds = target_stds.unsqueeze(0).unsqueeze(-1)

        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_dataloader)):
                batch_x, batch_label, batch_y, batch_x_stamp, batch_label_stamp, batch_air, batch_air_label, batch_static = batch
                batch_x = batch_x.float().to(self.device)
                batch_label = batch_label.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y.permute(0, 2, 1)
                batch_x_stamp = batch_x_stamp.float().to(self.device)
                batch_label_stamp = batch_label_stamp.float().to(self.device)
                batch_air = batch_air.float().to(self.device)
                batch_air_label = batch_air_label.float().to(self.device)
                batch_static = batch_static.float().to(self.device)

                # 获取模型预测值
                batch = batch_x, batch_label, batch_y, batch_x_stamp, batch_label_stamp, batch_air, batch_air_label, batch_static
                predict = self.model(batch)

                # 整体进行反归一化
                if self.dataset_norm:
                    batch_y = batch_y * target_stds + target_means
                    predict = predict * target_stds + target_means

                # 绘制图像
                # for j, pollutant in enumerate(self.target):
                #     pollutant_pth = result_path / pollutant
                #     pollutant_pth.mkdir(parents=True, exist_ok=True)
                #
                #     former = batch_x[0, j].cpu().numpy()
                #     predict_np = predict[0, j].cpu().numpy()
                #     gt_np = batch_y[0, j].cpu().numpy()
                #
                #     predict_all = np.concatenate([former, predict_np])
                #     gt_all = np.concatenate([former, gt_np])
                #     time_step = np.arange(predict_all.size)
                #
                #     plt.figure(figsize=(15, 6))
                #     plt.plot(time_step, predict_all, label='Predict', color='red')
                #     plt.plot(time_step, gt_all, label='GT', color='blue')
                #     plt.xlabel('Time Steps')
                #     plt.title(f'{pollutant}')
                #     plt.savefig(pollutant_pth / f'{i}.png')
                #     plt.close()

                # 计算当前batch的metrics
                batch_metrics.calculate(predict, batch_y, i)
                BR.update_batch(predict, batch_y)
                BN.update_batch(predict, batch_y)

            test_metrics.update_data(batch_metrics.mean)
            test_metrics.update_data({
                'rmse_loss': torch.sqrt(test_metrics.datas['mse_loss']),
                'rmse_per_feature': torch.sqrt(test_metrics.datas['mse_per_feature']),
                'R': BR.corr.to('cpu'),
                'R2': BR.R2_score.to('cpu'),
                'NME': BN.nme.to('cpu'),
                'NMB': BN.nmb.to('cpu')
            })

            test_metrics.save_log(result_path/'test_metrics.txt')
            test_metrics.show()

            return test_metrics

    def train_classification(self):
        print(f'start training: {self.dataset_type}_{self.setting}')
        checkpoint_save_pth = self.checkpoints_path / self.model_type / self.setting  # 保存checkpoint的路径
        Path(checkpoint_save_pth).mkdir(parents=True, exist_ok=True)
        train_dataset, train_dataloader = self.datasets['train_dataset'], self.dataloaders['train_dataloader']

        train_steps = len(train_dataloader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 获取待测试模型
        if self.model_path is not None:
            self.model = self._load_model(self.model_path)

        self.model = self.model.to(self.device)

        # 一次性冻结所有需要冻结的参数
        freeze_patterns = ['TriBlock.MMAList', 'output_projection']

        for name, param in self.model.named_parameters():
            if any(pattern in name for pattern in freeze_patterns):
                param.requires_grad = False
        print("已冻结预测头，保留分类头")

        # 开始训练
        model_optim = self._select_optimizer(self.lr)

        for current_epoch in range(self.train_epochs):
            self.model.train()
            epoch_start_time = time.time()  # 当前epoch的开始时间
            all_loss = []

            for i, batch in enumerate(tqdm(train_dataloader)):
                model_optim.zero_grad()  # 清除上一次反向传播中的梯度
                batch_x, batch_label, batch_y, batch_x_stamp, batch_label_stamp, batch_air, batch_air_label, batch_static = batch
                batch_x = batch_x.float().to(self.device)
                batch_label = batch_label.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y.permute(0, 2, 1)
                batch_x_stamp = batch_x_stamp.float().to(self.device)
                batch_label_stamp = batch_label_stamp.float().to(self.device)
                batch_air = batch_air.float().to(self.device)
                batch_air_label = batch_air_label.float().to(self.device)
                batch_static = batch_static.float().to(self.device)

                # 获取模型预测值
                batch = batch_x, batch_label, batch_y, batch_x_stamp, batch_label_stamp, batch_air, batch_air_label, batch_static
                classification = torch.log(self.model.classify(batch)+1e-8)
                batch_size, num_pollutants = classification.shape[0], classification.shape[1]
                base = torch.tensor(np.arange(num_pollutants)).long()
                labels = base.repeat(64, 1)
                labels = labels.to(classification.device)

                classification = classification.reshape(batch_size*num_pollutants, -1)
                labels = labels.reshape(batch_size*num_pollutants)
                loss = nn.functional.nll_loss(classification, labels)
                all_loss.append(loss.item())

                # 反向传播
                model_optim.zero_grad()  # 1. 清空梯度
                loss.backward()  # 2. 反向传播计算梯度
                model_optim.step()  # 3. 更新参数
            adjust_learning_rate(model_optim, current_epoch + 1, self.args)

            print("Epoch: {} cost time: {}".format(current_epoch + 1, time.time() - epoch_start_time))
            # 记录当前epoch的评均train_metrics
            epoch_avg_loss = sum(all_loss) / len(all_loss)

            # 获取验证集
            vali_loss = self.vali_classification()

            print(f"Epoch: {current_epoch + 1}, Steps: {train_steps}")
            print('train_metrics:')
            print("Loss: ", epoch_avg_loss)
            torch.save(self.model.state_dict(), checkpoint_save_pth / f'checkpoint_{current_epoch}.pth')

            early_stopping(vali_loss, self.model, checkpoint_save_pth)

            if early_stopping.early_stop:
                print("Early stopping")
                break
        return

    def vali_classification(self):
        # 1. 获取验证集
        vali_dataset, vali_dataloader = self.datasets['vali_dataset'], self.dataloaders['vali_dataloader']

        # 2. 存放验证集的指标
        vali_metrics, batch_metrics = ExpMetrics(self.target), ExpMetrics(self.target, len(vali_dataloader))

        # 3. 开始验证
        self.model.eval()

        all_loss = []

        with torch.no_grad():
            for i, batch in enumerate(vali_dataloader):
                batch_x, batch_label, batch_y, batch_x_stamp, batch_label_stamp, batch_air, batch_air_label, batch_static = batch
                batch_x = batch_x.float().to(self.device)
                batch_label = batch_label.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y.permute(0, 2, 1)
                batch_x_stamp = batch_x_stamp.float().to(self.device)
                batch_label_stamp = batch_label_stamp.float().to(self.device)
                batch_air = batch_air.float().to(self.device)
                batch_air_label = batch_air_label.float().to(self.device)
                batch_static = batch_static.float().to(self.device)

                # 获取模型预测值
                batch = batch_x, batch_label, batch_y, batch_x_stamp, batch_label_stamp, batch_air, batch_air_label, batch_static
                classification = torch.log(self.model.classify(batch)+1e-8)
                batch_size, num_pollutants = classification.shape[0], classification.shape[1]
                base = torch.tensor(np.arange(num_pollutants)).long()
                labels = base.repeat(64, 1)
                labels = labels.to(classification.device)

                classification = classification.reshape(batch_size*num_pollutants, -1)
                labels = labels.reshape(batch_size*num_pollutants)
                loss = nn.functional.nll_loss(classification, labels)
                all_loss.append(loss.item())

            vali_loss = sum(all_loss) / len(all_loss)

        return vali_loss

    def train_predictor(self):
        print(f'start training: {self.dataset_type}_{self.setting}')
        checkpoint_save_pth = self.checkpoints_path / self.model_type / self.setting  # 保存checkpoint的路径
        Path(checkpoint_save_pth).mkdir(parents=True, exist_ok=True)
        train_dataset, train_dataloader = self.datasets['train_dataset'], self.dataloaders['train_dataloader']

        train_steps = len(train_dataloader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        train_metrics, batch_metrics, current_metrics = ExpMetrics(self.target, self.train_epochs), ExpMetrics(self.target, len(train_dataloader)), ExpMetrics(self.target)

        # 获取待测试模型
        if self.model_path is not None:
            self.model = self._load_model(self.model_path)

        self.model = self.model.to(self.device)

        # 开始训练
        model_optim = self._select_optimizer(self.lr)

        # 一次性冻结所有需要冻结的参数
        freeze_patterns = ['PRE', 'Classifier']

        for name, param in self.model.named_parameters():
            if any(pattern in name for pattern in freeze_patterns):
                param.requires_grad = False
        print("已冻结分类头，保留预测头")

        if self.args.dataset_norm:
            with open(str(project_dir / 'normalization_params.json'), 'r') as f:
                data = json.load(f)
                means = data['air_means']
                stds = data['air_stds']
            air_cols = train_dataset.air_cols
            indices = [air_cols.index(target) for target in self.args.target]
            target_means = torch.tensor([means[i] for i in indices], requires_grad=False).to(self.device)
            target_stds = torch.tensor([stds[i] for i in indices], requires_grad=False).to(self.device)
            target_means = target_means.unsqueeze(0).unsqueeze(-1)
            target_stds = target_stds.unsqueeze(0).unsqueeze(-1)

        for current_epoch in range(self.train_epochs):
            self.model.train()
            epoch_start_time = time.time()  # 当前epoch的开始时间

            for i, batch in enumerate(tqdm(train_dataloader)):
                model_optim.zero_grad()  # 清除上一次反向传播中的梯度
                batch_x, batch_label, batch_y, batch_x_stamp, batch_label_stamp, batch_air, batch_air_label, batch_static = batch
                batch_x = batch_x.float().to(self.device)
                batch_label = batch_label.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y.permute(0, 2, 1)
                batch_x_stamp = batch_x_stamp.float().to(self.device)
                batch_label_stamp = batch_label_stamp.float().to(self.device)
                batch_air = batch_air.float().to(self.device)
                batch_air_label = batch_air_label.float().to(self.device)
                batch_static = batch_static.float().to(self.device)

                # 获取模型预测值
                batch = batch_x, batch_label, batch_y, batch_x_stamp, batch_label_stamp, batch_air, batch_air_label, batch_static
                predict = self.model(batch)

                not_nan = current_metrics.calculate(predict, batch_y)
                predict = predict * target_stds + target_means if self.args.dataset_norm else predict
                batch_y = batch_y * target_stds + target_means if self.args.dataset_norm else batch_y
                denormalized = current_metrics.calculate(predict, batch_y)
                batch_metrics.update_data(current_metrics.cpu, index=i)

                # 反向传播
                if not_nan:
                    self.backward(
                            _metrics=current_metrics.datas,
                            _optim=model_optim,
                            _scaler=None,
                            _loss_func=self.loss_func
                    )
            adjust_learning_rate(model_optim, current_epoch + 1, self.args)

            print("Epoch: {} cost time: {}".format(current_epoch + 1, time.time() - epoch_start_time))
            # 记录当前epoch的评均train_metrics
            train_metrics.update_data(batch_metrics.mean, index=current_epoch)

            # 获取验证集
            vali_metrics = self.vali()

            print(f"Epoch: {current_epoch + 1}, Steps: {train_steps}")
            print('train_metrics:')
            train_metrics.show(current_epoch)

            torch.save(self.model.state_dict(), checkpoint_save_pth / f'checkpoint_{current_epoch}.pth')

            if self.args.loss_func == 'mse':
                early_stopping(vali_metrics.datas['mse_loss'], self.model, checkpoint_save_pth)
            elif self.args.loss_func == 'rmse':
                early_stopping(vali_metrics.datas['rmse_loss'], self.model, checkpoint_save_pth)
            elif self.args.loss_func == 'ps':
                early_stopping(vali_metrics.datas['ps_loss'], self.model, checkpoint_save_pth)
            else:
                raise ValueError

            if early_stopping.early_stop:
                print("Early stopping")
                break

        return train_metrics
