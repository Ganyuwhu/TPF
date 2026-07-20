# 分析不同气象因素之间的统计关系
import pandas as pd
import numpy as np
from pathlib import Path
from Data_Provider.Site_Provider import wash_data
import seaborn as sns
import matplotlib.pyplot as plt
from Utils.Dir import get_project_root
from sklearn.feature_selection import mutual_info_regression


pollutant_cols = ["NO2", "PM2.5", "O3"]

air_cols = [
    "SO2",
    "NO2",
    "CO",
    "O3",
    "PM10",
    "PM2.5",
    "wd",
    "ws",
    "T",
    "P",
    "RH"
]

def corr_analysis(csv_path: Path):
    df_raw = wash_data(csv_path, zero_fix=True)
    df_air = df_raw[air_cols]
    corr_matrix = df_air.corr()

    corr_target = corr_matrix[pollutant_cols]

    plt.rcParams["font.family"] = "Arial"
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        corr_target,
        annot=True,  # 显示数值
        fmt=".2f",  # 保留两位小数
        cmap="coolwarm",  # 颜色
        center=0
    )

    plt.title(r"Correlation between NO$_2$, PM$_{2.5}$ and O$_3$ and other variables")
    plt.show()
    plt.savefig("Correlation.png")
    plt.close()
    print("done")

def lag_correlation(x, y, max_lag=24, mode="abs"):
    lags = range(-max_lag, max_lag + 1)
    corrs = []

    for lag in lags:
        if lag > 0:
            corr = x[:-lag].corr(y[lag:])
        elif lag < 0:
            corr = x[-lag:].corr(y[:lag])
        else:
            corr = x.corr(y)

        corrs.append(corr)
    corrs = np.array(corrs)
    lags = np.array(list(lags))
    valid_idx = ~np.isnan(corrs)
    lags = lags[valid_idx]
    corrs = corrs[valid_idx]

    # 选择最佳 lag
    if mode == "abs":
        best_idx = np.argmax(np.abs(corrs))
    elif mode == "pos":
        best_idx = np.argmax(corrs)
    elif mode == "neg":
        best_idx = np.argmin(corrs)
    else:
        raise ValueError("mode must be 'abs', 'pos', or 'neg'")

    best_lag = lags[best_idx]
    best_corr = corrs[best_idx]

    return lags, corrs, best_lag, best_corr

def delay_analysis(csv_path: Path):
    df_raw = wash_data(csv_path, zero_fix=True)
    df_air = df_raw[air_cols]
    for c1 in pollutant_cols:
        for c2 in air_cols:
            _, _, best_lag, best_corr= lag_correlation(df_air[c1], df_air[c2], 24)
            print(f'{c1} and {c2}: {best_lag} - {best_corr}')

def cossimilarity_analysis(csv_path: Path, calculate_length=336):
    df_raw = wash_data(csv_path, zero_fix=True)
    df_air = df_raw[air_cols]
    for c1 in pollutant_cols:
        for c2 in air_cols:
            c1_np = df_air[c1][:calculate_length].to_numpy()
            c2_np = df_air[c2][:calculate_length].to_numpy()
            cos_similarity = np.dot(c1_np, c2_np.T) / (np.linalg.norm(c1_np) * np.linalg.norm(c2_np))
            print(f'{c1} and {c2}: {cos_similarity}')

def mi_analysis(csv_path: Path):
    df_raw = wash_data(csv_path, zero_fix=True)
    df_air = df_raw[air_cols]
    for c1 in pollutant_cols:
        for c2 in air_cols:
            c1_np = df_air[c1].to_numpy()
            c2_np = df_air[c2].to_numpy()

            c1_np = (c1_np - c1_np.mean()) / c1_np.std()
            c2_np = (c2_np - c2_np.mean()) / c2_np.std()

            mi = mutual_info_regression(c1_np.reshape(-1, 1), c2_np)
            print(f'{c1} and {c2}: {mi}')


if __name__ == "__main__":
    data_path = get_project_root() / "Dataset/train.csv"
    corr_analysis(data_path)
