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
    parser.add_argument('--batch_size', type=int, default = 64)
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--target', type=list_from_string, default=['NO2', 'PM2.5', 'O3'])
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--pred_len', type=int, default=168)
    parser.add_argument('--label_len', type=int, default=168)
    parser.add_argument('--dataset_norm', type=bool, default=True)
    parser.add_argument('--lradj', type=str, default='type3')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument("--zero_fix", type=bool, default=True)
    parser.add_argument('--label', type=bool, default=False)

    # model
    parser.add_argument('--conv_stride', type=int, default=1)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--conv_layers', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=4)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--input_size', type=tuple, default=(1, 64, 64))
    parser.add_argument('--output_size', type=tuple, default=(64, 64, 32))
    parser.add_argument('--kernel_size', type=tuple, default=(3, 5, 3))
    parser.add_argument('--dilation', type=tuple, default=(1, 2, 4))
    parser.add_argument('--static_dim', type=int, default=None)
    parser.add_argument('--enc_in', type=int, default=3)

    # exp configs
    parser.add_argument('--mission', type=str, default='train')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--device_ids', type=list_from_string, default=['0', '1'])
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--use_amp', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default='LSTM')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-3)
    parser.add_argument('--dataset_type', type=str, default='site')
    parser.add_argument('--checkpoints_path', type=Path, default=project_dir / 'checkpoints')
    parser.add_argument('--loss_func', type=str, default='rmse')
    parser.add_argument('--date', type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_site_args()
    # args.mission = 'train'
    # args.csv_path = project_dir / "Dataset/2022.csv"
    # exp_train = Exp(args)
    # exp_train.train()
    args.mission = 'test'
    args.csv_path = project_dir / "Dataset/2023.csv"
    args.model_path = project_dir / f'checkpoints/LSTM/NO2_PM2.5_O3_LSTM_pl336_fl168_rmse_generate_None/checkpoint.pth'
    exp_test = Exp(args)
    exp_test.test()
