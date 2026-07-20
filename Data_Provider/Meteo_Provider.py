import glob
import json
from datetime import datetime, timedelta
from typing import Any, Optional, Union
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from osgeo import gdal  # GDAL 是一个开源的地理空间数据处理库
from tqdm import tqdm
import numpy as np
from matplotlib.colors import ListedColormap

from Utils.TimeFeatures import time_features
from Utils.Dir import get_project_root

project_root = get_project_root()


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

read_list = [
    "wd",
    "ws",
    "T",
    "P",
    "RH",
    "NO2",
    "O3",
    "PM2.5"
]

air_indices = [read_list.index(col) for col in air_cols]

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


# 读取给定时间的文件
def get_file_path(dt, path):
    dt_str = dt.strftime('%Y%m%d%H%M')  # 'YYYYMMDDHHMM'

    # 获取包含dt_str的文件
    path_obj = Path(path)
    for file_path in path_obj.iterdir():
        if dt_str in file_path.name and file_path.suffix.lower() == '.tif':
            return file_path

    return None


# 读取气象数据并获取有效数据的mask
def read_meteorology_data(file_path: Path):
    if file_path and file_path.exists():
        try:
            ds = gdal.Open(str(file_path))
            data = ds.ReadAsArray()  # shape: [channels, height, width] 或 [height, width]
            if data.ndim == 2:  # 当读取的array的形状为[height, width]时，在前面扩充一个维度
                data = np.expand_dims(data, axis=0)  # [1, height, width]

            return data

        except Exception as e:
            print(f"读取文件出错 {file_path}: {e}")
            return None
    else:
        return None


def get_mask(array, axis=0):
    """
    :param array: 通常为形状为[num_vars, 460, 920]的数组，也有可能为[460, 920]
    :return: mask: 形状为[460, 920]的数组
    """
    if isinstance(array, np.ndarray):
        invalid_datas = (
                (array == -9999) |
                np.isnan(array) |
                np.isinf(array)
        )

        mask = ~np.any(invalid_datas, axis=axis)

    elif isinstance(array, torch.Tensor):
        invalid_datas = (
                (array == -9999) |
                torch.isnan(array) |
                torch.isinf(array)
        )

        mask = ~torch.any(invalid_datas, dim=axis)

    else:
        raise TypeError('只允许从数组和张量中获取掩码')

    return mask


class MeteoDataset(Dataset):
    def __init__(self, target, seq_len, label_len, pred_len, dataset_norm, dataset_stride, start_time, end_time, sample_rate, label, **kwargs):
        super().__init__()
        self.target = target
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.dataset_norm = dataset_norm
        self.dataset_stride = dataset_stride
        self.start_time = datetime.strptime(start_time, "%Y%m%d%H%M")
        self.end_time = datetime.strptime(end_time, "%Y%m%d%H%M")
        self.sample_rate = sample_rate
        self.air_cols = air_cols
        self.static_cols = static_cols
        self.air_indices = air_indices
        self.static_mapping = static_mapping
        self.width = 920
        self.height = 460
        self.label = label

        self.valid_mask = None

        # 判断end_time是否越界
        valid_end_time = self.start_time + timedelta(hours=seq_len+label_len+pred_len)
        if valid_end_time > self.end_time:
            print(f"invalid end_time, fix to: {valid_end_time}")
            self.end_time = valid_end_time

        self.all_times = []
        current = self.start_time
        while current < self.end_time:
            self.all_times.append(current)
            current += timedelta(hours=1)

        # 创建时间戳
        date_series = pd.date_range(start=self.start_time, end=self.end_time-timedelta(hours=1), freq='h')
        time_feat = time_features(date_series, freq='h')  # shape assumed to be [1, D, T, 1] or similar
        self.time_stamp = torch.tensor(time_feat, dtype=torch.float32).permute(1, 0)

        self.pollutant_indices = [air_cols.index(pollutant) for pollutant in target]
        self.air_datas, self.static_datas, self.pollutant_datas, self.valid_index = self.__read__data__()

    def get_raw_data(self):
        raw_path = project_root / 'Dataset/Meteo/raw_datas'

        data_path: dict[str, Path] = {
            'air_datas': raw_path / 'air_datas',
            'pollutant_datas': {
                'NO2_datas': raw_path / 'pollutant_datas' / 'trainingLC_Pseudo_PatchTST_regression_PolRegLCPse_ftMS_sl336_ll0_pl336_dm512_nh16_el3_dl1_df512_fc1_ebtimeF_dtTrue_test_0_o1_NO2_pretrain1',
                'O3_datas': raw_path / 'pollutant_datas' / 'trainingLC_Pseudo_PatchTST_regression_PolRegLCPse_ftMS_sl336_ll0_pl336_dm512_nh16_el3_dl1_df512_fc1_ebtimeF_dtTrue_test_0_o1_O3_pretrain1',
                'PM25_datas': raw_path / 'pollutant_datas' / 'trainingLC_Pseudo_PatchTST_regression_PolRegLCPse_ftMS_sl336_ll0_pl336_dm512_nh16_el3_dl1_df512_fc1_ebtimeF_dtTrue_test_0_o1_PM25_pretrain1'
            },
            'static_datas': {
                'LULC': glob.glob(str(raw_path / 'static_datas' / '*_Fra_b100.tif')),  # 土地覆盖数据
                'DEM': raw_path / 'static_datas' / 'DEM_sz_4326.tif',  # 数字高程数据
                'road_destiny': raw_path / 'static_datas' / 'roadDensity_100.tif',  # 道路交通数据
                'building_height': raw_path / 'static_datas' / 'height_raster_100.tif',  # 建筑高度数据
                'poi': raw_path / 'static_datas' / 'poi',  # 污染物点位数据
                'D2S': raw_path / 'static_datas' / 'D2Sea_sz_4326.tif'  # 离海距离
            }
        }

        num_samples = len(self.all_times)

        # 读取一个样本文件，获取图像的部分信息
        sample_file = get_file_path(self.all_times[0], data_path['air_datas'])
        if sample_file and sample_file.exists():
            ds = gdal.Open(str(sample_file))  # ds是一个gdal.Dataset对象
            height = ds.RasterYSize  # 获取高度
            width = ds.RasterXSize  # 获取宽度
            air_vars = ds.RasterCount + 3
        else:
            raise FileNotFoundError('未找到对应文件')

        # 4. 获取气象原始数据
        air_mask = np.ones((height, width), dtype=np.bool_)  # 标注可用点

        raw_air_data = np.zeros((num_samples, air_vars, height, width), dtype=np.float32)
        print(f'read files, start time: {self.start_time}, end time: {self.end_time}, time stride: {self.dataset_stride}')
        for index, _time in enumerate(tqdm(self.all_times)):
            current_air_file = get_file_path(_time, data_path['air_datas'])
            current_NO2_file = get_file_path(_time, data_path['pollutant_datas']['NO2_datas'])
            current_O3_file = get_file_path(_time, data_path['pollutant_datas']['O3_datas'])
            current_PM25_file = get_file_path(_time, data_path['pollutant_datas']['PM25_datas'])

            current_air_data = gdal.Open(str(current_air_file)).ReadAsArray()
            # 将风向转化为三角函数形式
            current_air_data[0] = np.sin(current_air_data[0])

            current_NO2_data = np.expand_dims(gdal.Open(str(current_NO2_file)).ReadAsArray(), axis=0)
            current_O3_data = np.expand_dims(gdal.Open(str(current_O3_file)).ReadAsArray(), axis=0)
            current_PM25_data = np.expand_dims(gdal.Open(str(current_PM25_file)).ReadAsArray(), axis=0)

            current_NO2_data[current_NO2_data < 0] = 0
            current_O3_data[current_O3_data < 0] = 0
            current_PM25_data[current_PM25_data < 0] = 0

            raw_air_data[index] = np.concatenate((current_air_data, current_NO2_data, current_O3_data, current_PM25_data), axis=0)
            current_mask = get_mask(current_air_data) * get_mask(current_NO2_data) * get_mask(current_O3_data) * get_mask(current_PM25_data)

            air_mask = air_mask * current_mask

        # 5. 读取下垫面数据
        static_features = []
        # 5.1 读取LULC数据
        lulc_files = data_path['static_datas']['LULC']
        for file in lulc_files:
            dataset = gdal.Open(str(file))
            data = dataset.GetRasterBand(1).ReadAsArray()  # .flatten()
            data = np.where(data == 255, -9999, data)
            static_features.append(data)

        # 5.2 读取DEM数据
        dem_path = data_path['static_datas']['DEM']
        dataset = gdal.Open(str(dem_path))
        dem_band_names = ['elevation', 'slope']
        for i, band_name in enumerate(dem_band_names, start=1):
            data = dataset.GetRasterBand(i).ReadAsArray()  # .flatten()
            static_features.append(data)

        # 5.3 读取路网密度
        road_density_path = data_path['static_datas']['road_destiny']
        dataset = gdal.Open(str(road_density_path))
        data = dataset.GetRasterBand(1).ReadAsArray()  # .flatten()
        static_features.append(data)

        # 5.4 读取建筑高度
        building_height_path = data_path['static_datas']['building_height']
        dataset = gdal.Open(str(building_height_path))
        data = dataset.GetRasterBand(1).ReadAsArray()  # .flatten()
        static_features.append(data)

        # 5.5 读取离海距离
        building_height_path = data_path['static_datas']['D2S']
        dataset = gdal.Open(str(building_height_path))
        data = dataset.GetRasterBand(1).ReadAsArray()  # .flatten()
        static_features.append(data)

        # 5.6 读取POI密度
        poi_list = [poi for poi in self.static_cols if poi[:3] == 'poi']
        poi_path = data_path['static_datas']['poi']
        for poi_name in poi_list:
            poi_name = poi_name[4:]
            for file in Path(poi_path).iterdir():
                if poi_name in str(file) and (file.suffix == '.tif'):
                    dataset = gdal.Open(str(file))
                    data = dataset.GetRasterBand(1).ReadAsArray()  # .flatten()
                    static_features.append(data)

        raw_static_data = np.stack(static_features, axis=-1)
        static_mask = get_mask(raw_static_data, -1)

        # 6. 获取可行域
        mask = air_mask * static_mask
        return raw_air_data, raw_static_data, mask

    def __read__data__(self):
        # 0. 获取统计量
        # 保存归一化所需的数据
        statistic_path = project_root / 'normalization_params.json'
        with open(str(statistic_path), 'r') as f:
            statistic_dict = json.load(f)

        if self.dataset_norm:
            air_cols = self.air_cols
            static_cols = self.static_cols

            air_means = torch.tensor([statistic_dict['air_means'][i] for i, col in enumerate(air_cols)])
            air_stds = torch.tensor([statistic_dict['air_stds'][i] for i, col in enumerate(air_cols)])

            static_max = torch.tensor([statistic_dict['static_maxs'][i] for i, col in enumerate(static_cols)])
            static_min = torch.tensor([statistic_dict['static_mins'][i] for i, col in enumerate(static_cols)])

        # 1. 读取原始数据
        raw_air_data, raw_static_data, mask = self.get_raw_data()
        if not Path(project_root / 'Dataset/Meteo/raw_datas/mask.npy').exists():
            np.save(str(project_root / 'Dataset/Meteo/raw_datas/mask.npy'), mask)

        seq_len, air_vars, height, width = raw_air_data.shape
        _, _, static_vars = raw_static_data.shape

        mask = torch.tensor(mask)
        raw_air_data_tensor = torch.tensor(raw_air_data) * mask
        raw_static_data_tensor = torch.tensor(raw_static_data).permute(2, 0, 1).contiguous() * mask

        # 2. 展平
        flatten_mask = mask.contiguous().view(height * width)
        flatten_air_data_tensor = raw_air_data_tensor.contiguous().view(seq_len, air_vars, height * width)
        flatten_static_data_tensor = raw_static_data_tensor.contiguous().view(static_vars, height * width)

        # 3. 填充
        padding_tensor = torch.zeros(air_vars, height * width)
        expand_static_data_tensor = torch.cat([padding_tensor, flatten_static_data_tensor], dim=0)

        # 4. 获取可行域索引
        valid_index = torch.where(flatten_mask == 1)[0]
        valid_air_data = flatten_air_data_tensor[:, :, valid_index]
        valid_static_data = flatten_static_data_tensor[:, valid_index]

        # 如果有需要，则进行归一化操作
        if self.dataset_norm is True:
            air_means = air_means.unsqueeze(0).unsqueeze(-1)
            air_stds = air_stds.unsqueeze(0).unsqueeze(-1)

            static_max = static_max.unsqueeze(-1)
            static_min = static_min.unsqueeze(-1)

            valid_air_data = (valid_air_data - air_means) / air_stds
            valid_static_data = (valid_static_data - static_min) / (static_max - static_min)

        # 5. 按比例筛选数据
        if self.sample_rate != 1:
            valid_size = len(valid_index)
            sample_size = int(valid_size * self.sample_rate)
            indices = torch.randperm(valid_size)[:sample_size]  # 这个操作本身会打乱索引排序
            valid_index = valid_index[indices]
        valid_index = valid_index.numpy()

        # # 6. 分离变量
        # air_datas = {}
        # static_datas = {}
        # pollutant_datas = {}
        # len_air = valid_air_data.shape[1]
        # for pollutant in self.target:
        #     _air = valid_air_data[:, air_dict[pollutant], :]
        #     _pollutant = valid_air_data[:, target_dict[pollutant], :]
        #     static_index = [static_dict[pollutant][i]-len_air for i in range(len(static_dict[pollutant]))]
        #     _static = valid_static_data[static_index, :]
        #     air_datas[pollutant] = _air
        #     static_datas[pollutant] = _static
        #     pollutant_datas[pollutant] = _pollutant

        self.height, self.width = height, width
        indices = list(range(valid_air_data.shape[1]))
        for index in self.pollutant_indices:
            indices.remove(index)
        air_datas = valid_air_data[:, indices]
        pollutant_datas = valid_air_data[:, self.pollutant_indices]

        return air_datas, valid_static_data, pollutant_datas, valid_index

    def index_to_coordinate_2d(self, index):
        row = index // self.width  # 行索引
        col = index % self.width  # 列索引
        return row, col

    def __getitem__(self, index):
        if self.label:
            x = self.pollutant_datas[:self.seq_len, :, index]
            label = self.pollutant_datas[self.seq_len:self.seq_len+self.label_len, :, index]
            y = self.pollutant_datas[self.seq_len+self.label_len:, :, index]
            x_time_stamp = self.time_stamp[:self.seq_len]
            label_time_stamp = self.time_stamp[self.seq_len: self.seq_len + self.label_len]
            air = self.air_datas[:self.seq_len, :, index]
            air_label = self.air_datas[self.seq_len: self.seq_len + self.label_len, :, index]
            static = self.static_datas[:, index]
            return x, label, y, x_time_stamp, label_time_stamp, air, air_label, static

        else:
            x = self.pollutant_datas[self.label_len:self.label_len+self.seq_len, :, index]
            label = self.pollutant_datas[:self.label_len, :, index]
            y = self.pollutant_datas[self.label_len+self.seq_len:, :, index]
            x_time_stamp = self.time_stamp[self.label_len:self.label_len+self.seq_len]
            label_time_stamp = self.time_stamp[:self.label_len]
            air = self.air_datas[self.label_len:self.label_len+self.seq_len, :, index]
            air_label = self.air_datas[:self.label_len, :, index]
            static = self.static_datas[:, index]
            return x, label, y, x_time_stamp, label_time_stamp, air, air_label, static

    def __len__(self):
        if self.dataset_stride == 0:
            return len(self.valid_index)
        else:
            seq_len = self.air_datas[self.target[0]].shape[0]  # 获取序列长度
            one_step_window = self.seq_len + self.pred_len
            steps = ((seq_len - one_step_window) // int(self.dataset_stride)) + 1
            return len(self.valid_index) * steps


def get_meteo_dataloader():
    pass


if __name__ == "__main__":
    dataset = MeteoDataset(target=["NO2", "PM2.5", "O3"], seq_len=336, label_len=168, pred_len=168, dataset_norm=True, dataset_stride=0, start_time="202301010000",
                           end_time="202301010100", sample_rate=1., label=True)
    x, label, y, x_time_stamp, label_time_stamp, air, air_label, static = dataset[0]
    print(x.shape, label.shape, y.shape, x_time_stamp.shape, label_time_stamp.shape, air.shape, air_label.shape, static.shape)
