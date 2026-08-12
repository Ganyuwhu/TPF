import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import linregress, gaussian_kde
from sklearn.linear_model import LinearRegression
from matplotlib.colors import LinearSegmentedColormap


def list_from_string(s):
    return s.split(',')

def int_list(s):
    return [int(item) for item in s.split(',')]


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=True):
        super(Transpose, self).__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path / 'checkpoint.pth')
        self.val_loss_min = val_loss


# 绘制两个np数组的散点图
def scatter_plot(predict, gt, pollutant_name, save_pth, logs):
    gt = np.asarray(gt).flatten()
    pred = np.asarray(predict).flatten()

    # 去除无效值
    mask = (~np.isnan(gt)) & (~np.isnan(pred)) & (gt >= 0) & (pred >= 0)
    gt = gt[mask]
    pred = pred[mask]

    xy = np.vstack([gt, pred])
    density = gaussian_kde(xy)(xy)

    # 为了视觉效果：按密度排序（高密度后画）
    idx = density.argsort()
    gt, pred, density = gt[idx], pred[idx], density[idx]
    density_norm = (density - density.min()) / (density.max() - density.min())

    model = LinearRegression()
    model.fit(gt.reshape(-1, 1), pred)
    a = model.coef_[0]
    b = model.intercept_
    x_line = np.linspace(gt.min(), gt.max(), 100)
    y_line = model.predict(x_line.reshape(-1, 1))

    plt.figure(figsize=(6, 6))

    sc = plt.scatter(
        gt, pred,
        c=density_norm,
        s=10,
        cmap='turbo',
        alpha=0.8
    )

    plt.plot(x_line, y_line, color='black', lw=2)
    plt.plot(x_line, x_line, '--', color='gray', lw=1)

    plt.xlabel(f'Predict {pollutant_name}', fontsize=16)
    plt.ylabel(f'GT {pollutant_name}', fontsize=16)
    # plt.title(logs['title'], fontsize=16)

    plt.tight_layout()

    # 填充文本
    plt.text(0.05, 0.95, s=f'R = {logs["R"]: .2f}', transform=plt.gca().transAxes, fontsize=16) if 'R' in logs else 1
    plt.text(0.05, 0.9, s=f'RMSE = {logs["RMSE"]: .2f}', transform=plt.gca().transAxes, fontsize=16) if 'RMSE' in logs else 1
    plt.text(0.05, 0.85, s=f'y = {a: .2f}x + {b: .2f}', transform=plt.gca().transAxes, fontsize=16)

    plt.savefig(save_pth, dpi=600)

