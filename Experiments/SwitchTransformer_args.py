import argparse

import torch

from Experiments.exp import Exp
from Utils.Dir import *

project_dir = get_project_root()

def list_from_string(s):
    """将逗号分隔的字符串转换为列表"""
    return s.split(',')

def int_list(s):
    return [int(item) for item in s.split(',')]

def get_site_args():
    parser = argparse.ArgumentParser(description='Time Series Forecasting')

    # 通用部分
    parser.add_argument('--csv_path', type=str, default=project_dir / "Dataset/train.csv")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--target', type=list_from_string, default=['NO2', 'PM2.5', 'O3'])
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--pred_len', type=int, default=168)
    parser.add_argument('--label_len', type=int, default=168)
    parser.add_argument('--dataset_norm', type=bool, default=True)
    parser.add_argument('--lradj', type=str, default='type3')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument("--zero_fix", type=bool, default=True)
    parser.add_argument('--label', type=bool, default=False)

    # model
    parser.add_argument("--n_vars", type=int, default=1)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_experts", type=int, default=3)
    parser.add_argument("--depth", type=int, default=4)

    # exp configs
    parser.add_argument('--mission', type=str, default='train')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--device_ids', type=list_from_string, default=['0', '1'])
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--use_amp', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default='SwitchTransformer')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--dataset_type', type=str, default='site')
    parser.add_argument('--checkpoints_path', type=Path, default=project_dir / 'checkpoints')
    parser.add_argument('--loss_func', type=str, default='rmse')
    parser.add_argument('--date', type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_site_args()
    for item in args.target:
        args.target = [item]
        args.mission = 'train'
        args.csv_path = project_dir / "Dataset/train.csv"

        args.learning_rate = 1e-4
        exp_finetune = Exp(args)
        exp_finetune.train()
        args.mission = 'test'
        args.csv_path = project_dir / "Dataset/test.csv"
        args.model_path = project_dir / f'checkpoints/SwitchTransformer/{item}_SwitchTransformer_pl336_fl168_rmse_Shenzhen_None/checkpoint.pth'
        exp_test = Exp(args)
        test_metrics = exp_test.test()
        logs = {}
        for pollutant in args.target:
            logs[f'{pollutant}_logs'] = {}
        for i, pollutant in enumerate(args.target):
            logs[f'{pollutant}_logs']['R'] = test_metrics.datas['R'][i].item()
            logs[f'{pollutant}_logs']['RMSE'] = test_metrics.datas['rmse_per_feature'][i].item()
        exp_test.test_per_site(logs)
        args.model_path = None
