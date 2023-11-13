from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import time
import torch
import numpy as np
import pandas as pd
import json
from torch.utils.data import Dataset
from modules.config import config

import psutil
import os

""" REFERENCE 
Two different dataset types: https://pytorch.org/docs/stable/data.html#dataset-types
"""


class LOBDataset(Dataset):
    """ A packaged dataset to load a training set """
    # TODO 应该还是要用Dataset-DataLoader 的结构，并充分利用num_workers来加速数据读取

    def __init__(self,
                 is_train: bool,
                 config:dict,
                 pred_label: int = 0) -> None:
        super().__init__()
        data_path = './data/train_data.npy' if is_train is True else './data/valid_data.npy'
        label_path = './data/train_labels.npy' if is_train is True else './data/valid_labels.npy'
        self.data: torch.Tensor = torch.from_numpy(np.load(data_path))
        self.label: torch.Tensor = torch.from_numpy(np.load(label_path))
        # 选出对应的预测维度(stock_num, daily_sec)
        self.label = self.label[:, :, pred_label]
        # self.acc_sym_num: np.ndarray = np.load('./data/cum_sym_num_dict.npy')
        self.seq_len = config['seq_len']
        self.stride = config['stride']
        self.training_days = config['train_days'] if is_train is True else 64 - config['train_days']

    def __len__(self) -> int:
        return self.data.shape[0] * ((self.data.shape[1] - self.seq_len) // self.stride + 1)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        start_sec, stock_id = divmod(index, self.data.shape[0])
        start_sec *= self.stride  # 步长参数
        end_sec = start_sec + self.seq_len
        X = self.data[stock_id, start_sec:end_sec, :]
        y = self.label[stock_id, end_sec - 1]  # 这里的0表示以label_5为预测对象
        return X, y

if __name__ == '__main__':
    # train_iter = DataIterable('./data/train_data.npy', './data/train_labels.npy')
    train_iter = DataLoader(
        LOBDataset(is_train=True, seq_len=config['seq_len'], stride=config['stride'], train_days=config['train_days']), batch_size=8, shuffle=False)
    process = psutil.Process(os.getpid())
    time1 = time.time()
    for i, batch in enumerate(train_iter):
        mm_info = process.memory_full_info()
        print('\r', i, batch[0].shape, mm_info.uss /
              1024 / 1024, "MB", end='', flush=True)

    time2 = time.time() - time1
    print(f"\n{time2:.2f} sec")
