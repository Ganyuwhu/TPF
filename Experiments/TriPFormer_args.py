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
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument("--zero_fix", type=bool, default=True)
    parser.add_argument('--label', type=bool, default=False)

    # model
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--patch_len", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--static_dim", type=int, default=26)
    parser.add_argument("--padding", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dims", type=tuple, default=(1, 2))
    parser.add_argument("--time_dim", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--embed_type", type=str, default='fixed')
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--attn_type", type=str, default='DS')
    parser.add_argument("--mask_flag", type=bool, default=True)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--output_attention", type=bool, default=False)
    parser.add_argument("--last_dim", type=int, default=512)
    parser.add_argument("--n_vars", type=int, default=3)
    parser.add_argument("--init_model", type=int, default=512)
    parser.add_argument("--no_air", type=bool, default=False)
    parser.add_argument("--no_static", type=bool, default=False)
    parser.add_argument("--ms_type", type=str, default='self')
    parser.add_argument("--n_layers", type=int, default=3)

    # exp configs
    parser.add_argument('--mission', type=str, default='train')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--device_ids', type=list_from_string, default=['0', '1'])
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--use_amp', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default='TriPFormer')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--dataset_type', type=str, default='site')
    parser.add_argument('--checkpoints_path', type=Path, default=project_dir / 'checkpoints')
    parser.add_argument('--loss_func', type=str, default='rmse')
    parser.add_argument('--date', type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_site_args()
    exp = Exp(args)
    exp.train_classification()
    args.model_path = project_dir / f'checkpoints/TriPFormer/NO2_PM2.5_O3_TriPFormer_pl336_fl168_rmse_None/checkpoint.pth'
    exp.train_predictor()
    exp.train()
    args.mission = 'test'
    args.csv_path = project_dir / "Dataset/test.csv"
    args.model_path = project_dir / f'checkpoints/TriPFormer/NO2_PM2.5_O3_TriPFormer_pl336_fl168_rmse_None/checkpoint.pth'
    exp_test = Exp(args)
    exp_test.test()
