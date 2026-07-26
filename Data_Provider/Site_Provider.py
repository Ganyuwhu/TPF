import pandas as pd
import numpy as np
import random
import json
from pathlib import Path

import torch
from tqdm import tqdm
from typing import Optional

from Utils.Dir import *
from Utils.TimeFeatures import time_features
from torch.utils.data import Dataset, DataLoader

project_dir = get_project_root()

air_cols = [
    "NO2",
    "O3",
    "PM2.5",
    "wd",
    "ws",
    "T",
    "P",
    "RH"
]

static_cols = [
    "Bare",
    "Building",
    "Forest",
    "Grass",
    "OISA",
    "Road",
    "Water",
    "elevation",
    "slope",
    "road_density",
    "building_height",
    "D2S",
    "poi_交通设施",
    "poi_休闲娱乐",
    "poi_公司企业",
    "poi_医疗健康",
    "poi_商务住宅",
    "poi_旅游景点",
    "poi_汽车相关",
    "poi_生活服务",
    "poi_科教文化",
    "poi_购物消费",
    "poi_运动健身",
    "poi_酒店住宿",
    "poi_金融机构",
    "poi_餐饮美食"
]

static_mapping = {
    'NO2': ["Bare",
            "Building",
            "Forest",
            "Grass",
            "OISA",
            "Road",
            "Water",
            "elevation",
            "slope",
            "road_density",
            "building_height",
            "D2S",
            "poi_交通设施",
            "poi_休闲娱乐",
            "poi_公司企业",
            "poi_医疗健康",
            "poi_商务住宅",
            "poi_旅游景点",
            "poi_汽车相关",
            "poi_生活服务",
            "poi_科教文化",
            "poi_购物消费",
            "poi_运动健身",
            "poi_酒店住宿",
            "poi_金融机构",
            "poi_餐饮美食"
            ],
    'PM2.5': ["Bare",
              "Building",
              "Forest",
              "Grass",
              "OISA",
              "Road",
              "Water",
              "elevation",
              "slope",
              "road_density",
              "building_height",
              "D2S",
              "poi_交通设施",
              "poi_休闲娱乐",
              "poi_公司企业",
              "poi_医疗健康",
              "poi_商务住宅",
              "poi_旅游景点",
              "poi_汽车相关",
              "poi_生活服务",
              "poi_科教文化",
              "poi_购物消费",
              "poi_运动健身",
              "poi_酒店住宿",
              "poi_金融机构",
              "poi_餐饮美食"
              ],
    'O3': ["Forest",
           "Grass",
           "elevation",
           "slope",
           "road_density",
           "D2S",
           "poi_公司企业"
           ]
}


def wash_data(csv_path: Path, zero_fix: bool = False):
    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw.rename(columns={"PM25": "PM2.5"})

    df_raw.loc[(df_raw["P"] < 800) | (df_raw["P"] > 1100), "P"] = np.nan
    df_raw["P"] = df_raw["P"].fillna(df_raw["P"].median())

    df_raw[air_cols] = df_raw[air_cols].clip(lower=0)

    df_raw = df_raw.ffill()

    if zero_fix:
        df_raw[air_cols] = df_raw[air_cols].mask(df_raw[air_cols] <= 0, np.nan)
    df_raw[air_cols] = df_raw[air_cols].interpolate(method="linear")

    df_raw["wd"] = np.sin(df_raw["wd"] * np.pi / 180)

    return df_raw


class SiteDataset(Dataset):
    def __init__(self,
                 csv_path: Path,
                 seq_len: int = 336,
                 label_len: int = 168,
                 pred_len: int = 168,
                 freq: str = 'h',
                 zero_fix: bool = True,
                 sites: Optional[list] = None,
                 dataset_norm: bool = True,
                 target: Optional[list] = None,
                 mission: str = "train",
                 label: bool = False
                 ):
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.freq = freq
        self.zero_fix = zero_fix
        self.sites = sites
        self.normalize = dataset_norm
        self.target = target
        self.mission = mission

        self.site_samples = []

        self.read_data()
        self.cum_samples = np.cumsum(self.site_samples)
        self.air_cols = air_cols
        self.static_cols = static_cols
        self.static_mapping = static_mapping
        self.static_index = list(range(len(static_cols))) if len(self.target) != 1 else [static_cols.index(item) for item in static_mapping[self.target[0]]]
        self.label = label

    def read_data(self):
        df_raw = wash_data(self.csv_path, self.zero_fix)

        all_sites = df_raw["站点"].unique()
        self.sites = all_sites if self.sites is None else self.sites
        self.target = ["NO2", "O3", "PM2.5"] if self.target is None else self.target

        for site in self.sites:
            self.site_samples.append((df_raw["站点"] == site).sum() - self.label_len - self.pred_len - self.seq_len + 1)

        if not Path.exists(project_dir / f'Dataset/Site_pt/{self.mission}_sites.json'):
            self.to_pt(df_raw, sites=self.sites)

        if not Path.exists(project_dir / 'normalization_params.json'):
            if self.mission == "train":
                air_means = df_raw[air_cols].mean(skipna=True)
                air_stds = df_raw[air_cols].std(skipna=True)
                df_raw[air_cols] = (df_raw[air_cols] - air_means) / air_stds
                static_maxs = df_raw[static_cols].max(skipna=True)
                static_mins = df_raw[static_cols].min(skipna=True)
                df_raw[static_cols] = (df_raw[static_cols] - static_mins) / (static_maxs - static_mins)
                normalization_params = {
                    "air_means": air_means.to_list(),
                    "air_stds": air_stds.to_list(),
                    "static_maxs": static_maxs.to_list(),
                    "static_mins": static_mins.to_list()
                }
                with open(project_dir / 'normalization_params.json', 'w', encoding='utf-8') as f:
                    json.dump(normalization_params, f, ensure_ascii=False, indent=2)

        # if self.normalize:
        #     if not Path.exists(project_dir / 'normalization_params.json'):
        #         if self.mission == "train":
        #             air_means = df_raw[air_cols].mean(skipna=True)
        #             air_stds = df_raw[air_cols].std(skipna=True)
        #             df_raw[air_cols] = (df_raw[air_cols] - air_means) / air_stds
        #             static_maxs = df_raw[static_cols].max(skipna=True)
        #             static_mins = df_raw[static_cols].min(skipna=True)
        #             df_raw[static_cols] = (df_raw[static_cols] - static_mins) /  (static_maxs - static_mins)
        #             normalization_params = {
        #                 "air_means": air_means.to_list(),
        #                 "air_stds": air_stds.to_list(),
        #                 "static_maxs": static_maxs.to_list(),
        #                 "static_mins": static_mins.to_list()
        #             }
        #             with open(project_dir / 'normalization_params.json', 'w', encoding='utf-8') as f:
        #                 json.dump(normalization_params, f, ensure_ascii=False, indent=2)
        #         else:
        #             pass
        #     else:
        #         normalization_params = json.load(open(project_dir / 'normalization_params.json', 'r', encoding='utf-8'))
        #         air_means = pd.Series(normalization_params["air_means"], index=air_cols)
        #         air_stds = pd.Series(normalization_params["air_stds"], index=air_cols)
        #         static_maxs = pd.Series(normalization_params["static_maxs"], index=static_cols)
        #         static_mins = pd.Series(normalization_params["static_mins"], index=static_cols)
        #         df_raw[air_cols] = (df_raw[air_cols] - air_means) / air_stds
        #         df_raw[static_cols] = (df_raw[static_cols] - static_mins) /  (static_maxs - static_mins)

    def __len__(self):
        return sum(self.site_samples)

    def to_pt(self, df_raw, date_column='date', site_column='站点', sites=None, freq='h'):
        default_pt_dir = project_dir / f"Dataset/Site_pt/{self.mission}"
        default_pt_dir.mkdir(parents=True, exist_ok=True)
        all_cols = air_cols + static_cols
        index_map = {}
        for site in sites:
            df_site = df_raw[df_raw[site_column] == site]
            # 保存目标变量数据
            data = df_site[all_cols].to_numpy(dtype=np.float32)
            data_tensor = torch.tensor(data)  # shape: [T, D]
            data_path = default_pt_dir / f"{site}_data.pt"
            torch.save(data_tensor, data_path)

            # 保存时间特征
            date_series = pd.to_datetime(df_site[date_column].values)
            time_feat = time_features(date_series, freq=freq)  # shape assumed to be [1, D, T, 1] or similar
            time_feat_tensor = torch.tensor(time_feat, dtype=torch.float32).permute(1, 0)
            stamp_path = default_pt_dir / f"{site}_stamp.pt"
            torch.save(time_feat_tensor, stamp_path)

            # 索引信息更新
            index_map[site] = {
                "data_path": str(data_path),
                "stamp_path": str(stamp_path)
            }

        with open(str(project_dir / f"Dataset/Site_pt/{self.mission}_sites.json"), "w", encoding="utf-8") as f:
            json.dump(index_map, f, ensure_ascii=False, indent=2)

        print("已按站点将数据分隔......")
        print(f"- {len(sites)} 个站点 .pt 文件")
        print(f"- 索引文件：{project_dir / f'Dataset/Site_pt/{self.mission}_sites.json'}")

    def __getitem__(self, index):
        site_index = np.searchsorted(self.cum_samples, index, side='right')
        if site_index > 0:
            local_index = index - self.cum_samples[site_index - 1]
        else:
            local_index = index

        # 1. Raw data
        site = self.sites[site_index]
        site_pt = project_dir / f"Dataset/Site_pt/{self.mission}/{site}_data.pt"
        stamp_pt = project_dir / f"Dataset/Site_pt/{self.mission}/{site}_stamp.pt"
        site_data = torch.load(site_pt)[local_index: local_index + self.seq_len + self.label_len + self.pred_len]
        time_stamp = torch.load(stamp_pt)[local_index: local_index + self.seq_len + self.label_len + self.pred_len]

        # 2. Get index
        all_cols = self.air_cols + self.static_cols
        target_idx = [all_cols.index(col) for col in self.target]
        air_idx = [all_cols.index(col) for col in self.air_cols if col not in self.target]
        static_idx = [all_cols.index(col) for col in self.static_cols]

        # 3. Turn data to numpy
        target_data = site_data[:, target_idx]
        air_data = site_data[:, air_idx]
        static_data = site_data[:, static_idx]

        if self.label:
            x = target_data[:self.seq_len]
            label = target_data[self.seq_len: self.seq_len + self.label_len]
            y = target_data[self.seq_len + self.label_len:]
            x_time_stamp = time_stamp[:self.seq_len]
            label_time_stamp = time_stamp[self.seq_len: self.seq_len + self.label_len]
            air = air_data[:self.seq_len]
            air_label = air_data[self.seq_len: self.seq_len + self.label_len]
            static = static_data[0]
        else:
            x = target_data[self.label_len:self.label_len+self.seq_len]
            label = target_data[:self.label_len]
            y = target_data[self.label_len+self.seq_len:]
            x_time_stamp = time_stamp[self.label_len:self.label_len+self.seq_len]
            label_time_stamp = time_stamp[:self.label_len]
            air = air_data[self.label_len:self.label_len+self.seq_len]
            air_label = air_data[:self.label_len]
            static = static_data[0]

        # 5. normalization
        if self.normalize:
            normalization_params = json.load(open(project_dir / 'normalization_params.json', 'r', encoding='utf-8'))
            air_means = torch.tensor(normalization_params["air_means"])
            air_stds = torch.tensor(normalization_params["air_stds"])
            static_maxs = torch.tensor(normalization_params["static_maxs"])
            static_mins = torch.tensor(normalization_params["static_mins"])

            x_indices = [air_cols.index(col) for col in self.target]
            air_indices = [air_cols.index(col) for col in self.air_cols if col not in self.target]

            x_means = air_means[x_indices]
            x_stds = air_stds[x_indices]
            air_means = air_means[air_indices]
            air_stds = air_stds[air_indices]

            x = (x - x_means) / x_stds
            label = (label - x_means) / x_stds
            y = (y - x_means) / x_stds
            air = (air - air_means) / air_stds
            air_label = (air_label - air_means) / air_stds
            static = (static - static_mins) / (static_maxs - static_mins)
            static = static[self.static_index]

        """
            x: [seq_len, n_vars]
            label: [label_len, n_vars]
            y: [pred_len, n_vars]
            x_time_stamp: [seq_len, 4]
            label_time_stamp: [label_len, 4]
            air: [seq_len, air_vars]
            air_label: [label_len, air_vars]
            static: [static_vars, ]            
        """

        return x, label, y, x_time_stamp, label_time_stamp, air, air_label, static


def get_site_dataloader(args):
    if args.mission == "test" or args.mission == "test_p2p":
        site_path = get_project_root() / "Logs/test.txt"
        sites = site_path.read_text(encoding='utf-8').splitlines()
        batch_size = args.batch_size

        test_dataset = SiteDataset(csv_path=args.csv_path, seq_len=args.seq_len, label_len=args.label_len,
                                   pred_len=args.pred_len, freq=args.freq, zero_fix=args.zero_fix, sites=sites,
                                   dataset_norm=args.dataset_norm, target=args.target, mission=args.mission, label=args.label)

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True
        )
        return test_dataset, test_dataloader

    elif args.mission == "train":  # A validation set will also be created
        site_path = get_project_root() / "Logs/train.txt"
        sites = site_path.read_text(encoding='utf-8').splitlines()

        # 1. Select 80% sites for training
        sites_copy = sites.copy()
        random.shuffle(sites_copy)
        split_idx = int(len(sites_copy) * 0.8)

        train_sites = sites_copy[:split_idx]
        val_sites = sites_copy[split_idx:]

        # 2. build datasets
        batch_size = args.batch_size

        train_dataset = SiteDataset(csv_path=args.csv_path, seq_len=args.seq_len, label_len=args.label_len,
                                    pred_len=args.pred_len, freq=args.freq, zero_fix=args.zero_fix, sites=train_sites,
                                    dataset_norm=args.dataset_norm, target=args.target, mission=args.mission)

        val_dataset = SiteDataset(csv_path=args.csv_path, seq_len=args.seq_len, label_len=args.label_len,
                                  pred_len=args.pred_len, freq=args.freq, zero_fix=args.zero_fix, sites=val_sites,
                                  dataset_norm=args.dataset_norm, target=args.target, mission=args.mission)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            pin_memory=True
        )

        return train_dataset, train_dataloader, val_dataset, val_dataloader

    else:
        raise (ValueError("Invalid mission type"))


if __name__ == "__main__":
    ds = SiteDataset(csv_path=get_site_data_dir() / "test.csv", target=["NO2", "PM2.5", "O3"], mission="test")
    x, label, y, x_time_stamp, label_time_stamp, air, air_label, static = ds[0]
    print(x.shape, label.shape, y.shape, x_time_stamp.shape, label_time_stamp.shape, air.shape, air_label.shape, static.shape)
