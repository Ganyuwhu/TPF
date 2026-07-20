import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as f
from pathlib import Path
from datetime import datetime, timedelta

plt.rcParams['font.family'] = 'Arial'   # 英文论文常用
plt.rcParams['font.size'] = 16


def draw_importance():
    labels = ["wd", "ws", "T", "P", "RH", "SO$_2$", "NO$_2$", "CO", "O$_3$", "PM$_{10}$", "PM$_{2.5}$"]
    n = len(labels)

    r_no2 = [
        0.079750092, 0.309261301, 0.128445424, 0.174384417, 0.175486953, 0.132671812, 0, 0.037486218, 0.069092245,
        0.170341786, 0.085262771
    ]
    r_pm25 = [
        0.039365847, 0.064490125, 0.072954454, 0.034932151, 0.087733441, 0.096600833, 0.045411796, 0.000671772,
        0.031438936, 0.036544404, 0
    ]
    r_o3 = [
        0.013005051, 0.018055556, 0.086111111, 0.024494949, 0.024747475, 0.030050505, 0.023989899, -0.018813131,
        0, 0.015656566, -0.005808081
    ]

    rmse_no2 = [
        0.043211263, 0.31969252, 0.119917084, 0.146096044, 0.220763517, 0.07617896, 0, 0.033149076, 0.051761962,
        0.102927967, 0.081128001
    ]

    rmse_pm25 = [
        0.05368335, 0.072228783, 0.095391649, 0.06184334, 0.131014019, 0.165516093, 0.063387531, 0.001468495,
        0.048202985, 0.050579829, 0
    ]

    rmse_o3 = [
        0.044292201, 0.079108591, 0.271239205, 0.068383337, 0.232879229, 0.075381264, 0.083135155, -0.039583169, 0,
        0.055453185, -0.023308922
    ]

    # y轴位置
    y = np.arange(n)

    # 每个bar的偏移
    h = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

    # ---------------- 左图 (R) ----------------
    ax = axes[0]

    ax.barh(y - h, r_pm25, height=h, label="PM$_{2.5}$", color="#d73027")
    ax.barh(y, r_no2, height=h, label="NO$_{2}$", color="#fc8d59")
    ax.barh(y + h, r_o3, height=h, label="O$_{3}$", color="#fee090")

    ax.axvline(0, color='black')

    ax.set(yticks=y, yticklabels=labels)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_title("(a) R")
    ax.set_xlabel("Correlation Coefficient")

    ax.grid(axis='x', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.3)

    ax.legend()

    # ---------------- 右图 (RMSE) ----------------
    ax = axes[1]

    # ⚠️ 注意：颜色和左图一致（你要求的）
    ax.barh(y - h, rmse_pm25, height=h, color="#d73027")
    ax.barh(y, rmse_no2, height=h, color="#fc8d59")
    ax.barh(y + h, rmse_o3, height=h, color="#fee090")

    ax.axvline(0, color='black')

    ax.set_title("(b) RMSE")
    ax.set_xlabel("Correlation Coefficient")

    ax.grid(axis='x', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.3)

    plt.subplots_adjust(wspace=0.2)
    plt.show()


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


def compare(npy_list, model_list, r_list, rmse_list):
    data = [np.load(f).squeeze() for f in npy_list]
    plt.figure(figsize=(20, 6))

    start_time = datetime.strptime("202301160000", "%Y%m%d%H%M")

    time_axis = [
        start_time + timedelta(hours=24*i)
        for i in range(len(data[0]))
    ]

    plt.plot(time_axis, data[0], label="GT")
    for i, d in enumerate(data[1:]):
        plt.plot(time_axis, d, label=model_list[i])

    r_text = ["R = "]
    rmse_text = ["RMSE = "]
    for r, rmse in zip(r_list, rmse_list):
        r_text.append(str(r))
        r_text.append(" / ")

        rmse_text.append(str(rmse))
        rmse_text.append(" / ")

    ax = plt.gca()
    lines = ax.get_lines()
    colors = [line.get_color() for line in lines]

    x, y = 0.585, 0.98
    for i in reversed(range(len(r_text))):
        t = ax.text(
            x, y,
            r_text[i],
            color="black" if i % 2 == 0 else colors[i//2 + 1],  # 注意：第0条是基准，所以用 i+1
            transform=ax.transAxes,
            ha='right',
            va='top'
        )

        # 关键：获取文本宽度，往左移动
        renderer = plt.gcf().canvas.get_renderer()
        bbox = t.get_window_extent(renderer=renderer)
        width = bbox.width / plt.gcf().dpi / plt.gcf().get_size_inches()[0]

        x -= (width + 0.005)

    x, y = 0.585, 0.93
    for i in reversed(range(len(rmse_text))):
        t = ax.text(
            x, y,
            rmse_text[i],
            color="black" if i % 2 == 0 else colors[i//2 + 1],  # 注意：第0条是基准，所以用 i+1
            transform=ax.transAxes,
            ha='right',
            va='top'
        )

        # 关键：获取文本宽度，往左移动
        renderer = plt.gcf().canvas.get_renderer()
        bbox = t.get_window_extent(renderer=renderer)
        width = bbox.width / plt.gcf().dpi / plt.gcf().get_size_inches()[0]

        x -= (width + 0.005)


    plt.legend(loc='upper left')
    degree = "\u00B0"
    plt.title(f"PM$_{{2.5}}$ in Minzhi(114.261{degree}E, 22.620{degree}N)")
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.show()


if __name__ == "__main__":
    npy_list = [
        rf"C:\Users\gzr\Desktop\study\paper\0APS\图片\Workflow\test_p2p_result\minzhi\PM25.npy",
        rf"C:\Users\gzr\Desktop\study\paper\0APS\图片\Workflow\test_p2p_result\Test\NO2_PM25_O3_Test_pl336_fl168_dm128_dff1024_rmse_Attn_cross_2026-03-11-static_false\Test_PM25_民治_24avg.npy",
        rf"C:\Users\gzr\Desktop\study\paper\0APS\图片\Workflow\test_p2p_result\TimeMixer\NO2_PM25_O3_TimeMixer_pl336_fl168_dm64_dff128_rmse_Attn_cross_2026-02-27\TimeMixer_PM25_民治_24avg.npy",
        rf"C:\Users\gzr\Desktop\study\paper\0APS\图片\Workflow\test_p2p_result\TCN_BiLSTM\NO2_PM25_O3_TCN_BiLSTM_pl336_fl168_dm128_dff1024_rmse_Attn_cross_2026-01-23\TCN_BiLSTM_PM25_民治_24avg.npy",
    ]

    r_list = [0.916, 0.88, 0.694]
    rmse_list = [3.895, 4.707, 6.991]
    model_list = ["TriPFormer", "TimeMixer", "STL-TCN-BiLSTM"]

    compare(npy_list, model_list, r_list, rmse_list)
