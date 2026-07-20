import math

import torch
import torch.nn as nn
import torch.nn.functional as f

"""
    predict: [batch_size, num_features, predict_len];
    gt: [batch_size, num_features, predict_len]
"""

STANDARD_TABLES = {
    'SO2': [0, 150, 500, 650, 800, float('inf')],
    'CO': [0, 5, 10, 35, 60, 90, float('inf')],
    'NO2': [0, 100, 200, 700, 1200, 2340, float('inf')],
    'O3': [0, 100, 160, 215, 265, 800, float('inf')],
    'PM25': [0, 35, 75, 115, 150, 250, float('inf')],
    'API': [0, 50, 100, 150, 200, 250, 300, 400, 500]
}


class PS_Loss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.3, gamma=0.3, dim=1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.dim = dim

    def forward(self, predict, gt):
        predict_feature = predict.unsqueeze(dim=self.dim)
        gt_feature = gt.unsqueeze(dim=self.dim)
        pearson_loss = 1 - pearson_corr(predict_feature, gt_feature)
        mae_loss_fun = nn.L1Loss()
        mae_loss = mae_loss_fun(predict_feature, gt_feature) # 用softmax计算归一化分布
        p_predict = f.softmax(predict_feature, dim=-1) + 1e-6
        p_gt = f.softmax(gt_feature, dim=-1) + 1e-6
        kl_loss = f.kl_div(
            input=p_predict.log(), # 输入需取log
            target=p_gt, # 目标分布
            reduction='batchmean', # 对batch和patch取平均
            log_target=False # target未取log
        )

        pearson_loss, kl_loss, mae_loss = pearson_loss.squeeze(), kl_loss.squeeze(), mae_loss.squeeze()

        losses_item = torch.stack([pearson_loss, kl_loss, mae_loss])
        self.alpha, self.beta, self.gamma = losses_item / (losses_item.sum() + 1e-5)
        result = self.alpha * pearson_loss + self.beta * kl_loss + self.gamma * mae_loss
        return result


class BatchWelford_R:
    def __init__(self, feature_num=1, device='cpu'):
        self.feature_num = int(feature_num)
        self.n = 0
        self.mean_x = torch.zeros(self.feature_num).to(device)
        self.mean_y = torch.zeros(self.feature_num).to(device)
        self.S_xx = torch.zeros(self.feature_num).to(device)
        self.S_yy = torch.zeros(self.feature_num).to(device)
        self.S_xy = torch.zeros(self.feature_num).to(device)

    def update_batch(self, x, y):
        cx = x.clone().to(self.mean_x.device)
        cy = y.clone().to(self.mean_y.device)

        feature_num = x.shape[1]
        cx = cx.permute(1, 0, 2).contiguous().reshape(feature_num, -1)
        cy = cy.permute(1, 0, 2).contiguous().reshape(feature_num, -1)

        total_num = cx.shape[-1]

        mean_xb = torch.mean(cx, dim=1)
        mean_yb = torch.mean(cy, dim=1)

        dx = mean_xb - self.mean_x
        dy = mean_yb - self.mean_y

        # batch内统计
        S_xx = ((cx - mean_xb.unsqueeze(-1)) ** 2).sum(dim=-1)
        S_yy = ((cy - mean_yb.unsqueeze(-1)) ** 2).sum(dim=-1)
        S_xy = ((cx - mean_xb.unsqueeze(-1)) * (cy - mean_yb.unsqueeze(-1))).sum(dim=-1)

        n_new = self.n + total_num

        # 合并
        self.S_xx = self.S_xx + S_xx + dx * dx * self.n * total_num / n_new
        self.S_yy = self.S_yy + S_yy + dy * dy * self.n * total_num / n_new
        self.S_xy = self.S_xy + S_xy + dx * dy * self.n * total_num / n_new
        self.mean_x = self.mean_x + dx * total_num / n_new
        self.mean_y = self.mean_y + dy * total_num / n_new
        self.n = n_new

    @ property
    def corr(self):
        return (self.S_xy / (torch.sqrt(self.S_xx * self.S_yy) + 1e-5)).to('cpu')

    @ property
    def R2_score(self):
        return self.corr ** 2


class BatchNME_Calculator:
    def __init__(self, feature_num=1, device='cpu'):
        self.feature_num = feature_num
        self.abs_delta = torch.zeros(feature_num).to(device)
        self.delta = torch.zeros(feature_num).to(device)
        self.y = torch.zeros(feature_num).to(device)

    def update_batch(self, x, y, predict_first=True):
        cx = x.clone()
        cy = y.clone()

        feature_num = cx.shape[1]
        cx = cx.permute(1, 0, 2).contiguous().reshape(feature_num, -1).to(self.delta.device)
        cy = cy.permute(1, 0, 2).contiguous().reshape(feature_num, -1).to(self.delta.device)

        batch_delta = torch.sum(cx - cy, dim=-1) if predict_first else torch.sum(cy - cx, dim=-1)
        batch_abs_delta = torch.sum(torch.abs(cx - cy), dim=-1)
        batch_y = torch.sum(cy, dim=-1) if predict_first else torch.sum(cx, dim=-1)

        self.delta = self.delta + batch_delta
        self.abs_delta = self.abs_delta + batch_abs_delta
        self.y = self.y + batch_y

    @property
    def nmb(self):
        return self.delta / (self.y + 1e-5)

    @ property
    def nme(self):
        return self.abs_delta / (self.y + 1e-5)


def criterion(loss_type='mse'):
    losses = {
        'mae': nn.L1Loss(),
        'mae_per_feature': nn.L1Loss(reduction='none'),
        'mse': nn.MSELoss(),
        'mse_per_feature': nn.MSELoss(reduction='none'),
        'ps': PS_Loss()
    }
    return losses.get(loss_type, nn.MSELoss())


def pearson_corr(x, y, dim=0, eps=1e-5):
    x = x - x.mean(dim=dim, keepdim=True)
    y = y - y.mean(dim=dim, keepdim=True)

    num = (x * y).sum(dim=dim)
    den = torch.sqrt((x ** 2).sum(dim=dim)) * torch.sqrt((y ** 2).sum(dim=dim))
    return num / (den + eps)


def r2_score(x, y, dim=0, eps=1e-5):
    ss_res = ((y - x) ** 2).sum(dim=dim)
    ss_tot = ((y - y.mean(dim=dim, keepdim=True)) ** 2).sum(dim=dim)
    return 1 - ss_res / (ss_tot + eps)


def get_tops(x, y, top_rate, dim=0, eps=1e-5):
    delta = (x - y).abs()
    thr = top_rate / 100.0
    return ((delta / (y + eps)) < thr).float().mean(dim=dim)


def get_nmb(x, y, dim=0, eps=1e-5):
    return (x - y).sum(dim=dim) / (y.sum(dim=dim) + eps)


def get_mape(x, y, dim=0, eps=1e-5):
    return torch.abs((x - y) / (y + eps)).mean(dim=dim)


def get_pollution_level(
        potency: torch.Tensor,
        pollutant: str,
        output_type: str = 'show'
):

    standard_list = torch.tensor(STANDARD_TABLES[pollutant], device=potency.device)
    api_list = torch.tensor(STANDARD_TABLES['API'], device=potency.device)
    ranked_potency = torch.searchsorted(standard_list.contiguous(), potency.contiguous())
    r = ranked_potency
    api_low, api_high = api_list[r - 1], api_list[r]
    std_low, std_high = standard_list[r - 1], standard_list[r]
    score = (api_high - api_low) / (std_high - std_low) * (potency - std_low) + api_low

    if output_type == 'show':
        return score
    elif output_type == 'predict':
        interval_array = torch.stack([0.75 * score, 1.25 * score], dim=1)
        return interval_array
    else:
        raise ValueError


def get_accuracy(potency_api: torch.Tensor, gt_api: torch.Tensor, pollutant: str):
    """
    计算预测区间准确率（基于污染物标准区间）

    参数:
        potency: 预测区间张量，形状为 [N, 2]（每行代表一个区间 [low, high]）
        gt: 真实值张量，形状为 [N]
        pollutant: 污染物类型（'SO2'/'CO'/'NO2'/'O3'/'PM25'/'API'）

    返回:
        准确率（0~1之间的标量）
    """
    potency_level = torch.searchsorted(torch.tensor(STANDARD_TABLES[pollutant], device=potency_api.device), potency_api)
    gt_level = torch.searchsorted(torch.tensor(STANDARD_TABLES[pollutant], device=potency_api.device), gt_api)

    if len(potency_level.shape) > 2:
        correct = (gt_level >= potency_level[:, 0, :]) & (gt_level <= potency_level[:, 1, :])
    else:
        correct = (gt_level >= potency_level[:, 0]) & (gt_level <= potency_level[:, 1])
    correct_rate = correct.sum() / gt_api.numel()

    return correct_rate


def accuracy_calculator(predict, gt, pollutant):
    potency_api = get_pollution_level(predict, pollutant, output_type='predict')
    acc = get_accuracy(potency_api, gt, pollutant)
    return acc


# 在最后一个维度上做滑动平均
def Moving_Avg(tensor, window_length):
    tensor = torch.tensor(tensor) if not isinstance(tensor, torch.Tensor) else tensor
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(1)

    # 构建卷积核
    kernel = torch.ones(1, 1, window_length, device=tensor.device, dtype=tensor.dtype) / window_length

    ma = f.conv1d(tensor, kernel, stride=window_length)

    if len(tensor.shape) == 1:
        ma = ma.squeeze(0).squeeze(0)
    if len(tensor.shape) == 2:
        ma = ma.squeeze(1)

    return ma


# 在最后一个维度上做周期最大滑动平均
def Moving_Max_Avg(tensor, window_length, period):
    assert (period > window_length)

    # 构造卷积核
    tensor = torch.tensor(tensor) if not isinstance(tensor, torch.Tensor) else tensor
    kernel = torch.ones(1, 1, window_length, device=tensor.device, dtype=tensor.dtype) / window_length

    # 变形
    shape = tensor.shape
    nums = shape[-1] // period
    tensor_reshape = tensor[:nums*period].contiguous().view((nums, period))

    # 滑动平均
    ma = f.conv1d(tensor_reshape.unsqueeze(1), kernel).squeeze(1)
    max_ma = ma.max(dim=-1).values

    return max_ma


# 获取分类评价指标
def Categorical(predict, gt, pollutant_name, level):
    """
    :param predict: 预测值
    :param gt: 真值
    :param pollutant_name: 污染物名称
    :param level: 限定等级
    :return: 分类评价字典
    """
    pollute_level = {
        'NO2_level_1': 100,
        'NO2_level_2': 200,
        'O3_level_1': 100,
        'O3_level_2': 160,
        'PM25_level_1': 35,
        'PM25_level_2': 75,
    }

    # 计算命中数，漏报数，虚警数以及正确否定数
    standard = pollute_level[f'{pollutant_name}_level_{level}']

    H = torch.sum((predict >= standard) & (gt >= standard)).item()
    M = torch.sum((predict < standard) & (gt >= standard)).item()
    F = torch.sum((predict >= standard) & (gt < standard)).item()
    C = torch.sum((predict < standard) & (gt < standard)).item()

    HIT = H / (H + M) if (H + M) != 0 else 1
    FAR = F / (H + F) if (H + F) != 0 else 1
    CSI = H / (H + M + F) if (H + M + F) != 0 else 1
    POC = (H + C) / (H + M + F + C) if (H + M + F + C) != 0 else 1

    categorical_metrics = {
        'H': H,
        'M': M,
        'F': F,
        'C': C,
        'Total': H+M+F+C,
        'HIT': HIT,
        'FAR': FAR,
        'CSI': CSI,
        'POC': POC
    }

    return categorical_metrics


class Metrics_Calculator:
    def __init__(self, target):
        # target
        self.target = target

        # loss functions
        self.mse_loss = criterion('mse')
        self.mse_per_feature = criterion('mse_per_feature')

        self.mae_loss = criterion('mae')
        self.mae_per_feature = criterion('mae_per_feature')

        self.ps_loss = criterion('ps')

    def calculate(self, _predict, _gt):
        _metric = {}

        _mask = (_gt > 0)  # 清除真值中的非正项
        _mask = _mask.detach()

        valid_predict, valid_gt = _predict[_mask], _gt[_mask]
        valid_predict_feature, valid_gt_feature = [_predict[:, i][_mask[:, i]] for i in range(_predict.shape[1])], [_gt[:, i][_mask[:, i]] for i in range(_gt.shape[1])]

        mse_loss = self.mse_loss(valid_predict, valid_gt)
        mse_per_feature = torch.stack([self.mse_loss(valid_predict_feature[i], valid_gt_feature[i]) for i in range(len(valid_predict_feature))], dim=0)

        mae_loss = self.mae_loss(valid_predict, valid_gt)
        mae_per_feature = torch.stack([self.mae_loss(valid_predict_feature[i], valid_gt_feature[i]) for i in range(len(valid_predict_feature))], dim=0)

        tops_10 = torch.stack([get_tops(valid_predict_feature[i], valid_gt_feature[i], 10) for i in range(len(valid_predict_feature))], dim=0)
        tops_30 = torch.stack([get_tops(valid_predict_feature[i], valid_gt_feature[i], 30) for i in range(len(valid_predict_feature))], dim=0)
        ps = torch.stack([self.ps_loss(valid_predict_feature[i], valid_gt_feature[i]) for i in range(len(valid_predict_feature))], dim=0)

        _metric['mse_loss'] = mse_loss
        _metric['mse_per_feature'] = mse_per_feature

        _metric['mae_loss'] = mae_loss
        _metric['mae_per_feature'] = mae_per_feature

        _metric['rmse_loss'] = torch.sqrt(mse_loss)
        _metric['rmse_per_feature'] = torch.sqrt(mse_per_feature)

        _metric['ps_loss'] = ps.mean()
        _metric['ps_per_feature'] = ps

        _metric['tops_10'] = tops_10
        _metric['tops_30'] = tops_30

        metric_valid = self.self_check(_metric)
        if metric_valid:
            return _metric
        else:
            return None

    def self_check(self, metric):
        for key in metric.keys():
            if metric[key].isnan().any():
                return False
        return True

    def __call__(self, _predict, _gt):
        return self.calculate(_predict, _gt)


if __name__ == '__main__':
    # 测试案例
    predict = torch.randn((8400)) * 150
    gt = predict + torch.randn((8400)) * 10

    # 滑动平均
    predict_ma = Moving_Max_Avg(predict, 8, 24)
    gt_ma = Moving_Max_Avg(gt, 8, 24)

    cate_metrics = Categorical(predict_ma, gt_ma, 'O3', 1)
    print(predict_ma, gt_ma, cate_metrics)
