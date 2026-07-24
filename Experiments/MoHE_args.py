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
    parser.add_argument('--multi_modal', type=bool, default=True, help='Enable multi-modal mode')
    parser.add_argument('--is_causal', type=bool, default=False, help='Use causal attention')
    parser.add_argument('--n_layer', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--d_model', type=int, default=384, help='Model hidden dimension')
    parser.add_argument('--block_size', type=int, default=672, help='Block size for attention')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--n_kv_heads', type=int, default=3, help='Number of key/value heads (GQA)')
    parser.add_argument('--d_ff', type=int, default=768, help='Feed-forward network dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--drop_path', type=float, default=0.3, help='Stochastic depth drop rate')
    parser.add_argument('--norm_type', type=str, default='rms', choices=['rms', 'layer'], help='Normalization type')
    parser.add_argument('--diff_attn', type=bool, default=False, help='Use differential attention')
    parser.add_argument('--ffn_type', type=str, default='dwconv', choices=['mlp', 'conv', 'dwconv', 'fan'],
                        help='FFN type')
    parser.add_argument('--glu', type=bool, default=False, help='Use GLU in FFN')
    parser.add_argument('--n_experts', type=int, default=8, help='Number of experts in MoE')
    parser.add_argument('--top_k_experts', type=int, default=2, help='Top-K experts to route')
    parser.add_argument('--experts_type', type=str, default='fan', choices=['mlp', 'fan'], help='Experts type')
    parser.add_argument('--exp_route_dropout', type=float, default=0.1, help='Expert router dropout rate')
    parser.add_argument('--exp_route_temperature', type=float, default=1.0, help='Expert router temperature')
    parser.add_argument('--bias', type=bool, default=False, help='Use bias in linear layers')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    parser.add_argument('--use_qk_norm', type=bool, default=False, help='Use query-key normalization')
    parser.add_argument('--headwise_attn_gate', type=bool, default=False, help='Use headwise attention gate')
    parser.add_argument('--c_att_mode', type=str, default='full', choices=['full', 'partial'],
                        help='Cross attention mode')
    parser.add_argument('--n_vars', type=int, default=3, help='Number of variables')
    parser.add_argument('--cross_vars', type=int, default=5, help='Number of cross variables')

    # exp configs
    parser.add_argument('--mission', type=str, default='train')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--device_ids', type=list_from_string, default=['0', '1'])
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--use_amp', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default='MoHETransformer')
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
    # exp_classification = Exp(args)
    # exp_classification.train_classification()
    #
    # args.model_path = project_dir / f'checkpoints/TriPFormer/NO2_PM2.5_O3_TriPFormer_pl336_fl168_rmse_None/checkpoint.pth'
    # exp_predictor = Exp(args)
    # exp_predictor.train_predictor()

    args.learning_rate = 1e-4
    exp_finetune = Exp(args)
    exp_finetune.train()

    args.mission = 'test'
    args.csv_path = project_dir / "Dataset/test.csv"
    args.model_path = project_dir / f'checkpoints/MoHETransformer/NO2_PM2.5_O3_MoHETransformer_pl336_fl168_rmse_None/checkpoint.pth'
    exp_test = Exp(args)
    exp_test.test()
