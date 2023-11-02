import torch
import numpy as np
import json
from torch.utils.data import Dataset
from modules.config import config


class LOBDataset(Dataset):

    def __init__(self, data_path:str, label_path:str, dict_path:str, is_train:bool) -> None:
        super().__init__()
        self.is_train = is_train
        self.daily_sym_num_dict: dict = np.load(dict_path)  # 这里存储了每一天有多少只股票参与计算
        self.data: torch.tensor = torch.tensor(np.load(data_path), dtype=torch.float32)
        self.labels: torch.tensor = torch.tensor(np.load(label_path), dtype=torch.int64)
        # data (total_sym, time, feature_size)
        # 例如(3998, 64*10, 2000)

    def __len__(self) -> int:
        # 返回的是训练集或验证集所有天的秒数和
        days = config['train_days'] if self.is_train is True else 64 - config['train_days']
        return self.data.shape[0] * days

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        cur_day, cur_sec = divmod(index, config['daily_secs']) # 获得现在index所指的天数和秒数
        cur_day_sym_num: int  = self.daily_sym_num_dict[cur_day] #获取今天有多少个股票

        present_sym_start_index: int = sum(list(self.daily_sym_num_dict.values())[:cur_day]) # 不包含currday

        X = self.data[cur_sec, present_sym_start_index:present_sym_start_index + cur_day_sym_num,:]
        y = self.labels[cur_sec, present_sym_start_index:present_sym_start_index + cur_day_sym_num, 0] # 这里的0表示以label_5为预测对象
        return X, y 


def gen_dataiter(data, labels, daily_sym_num_dict, days,) -> tuple[np.ndarray, np.ndarray]:
    '''An iterator to get data sequentially'''

    for day in range(days):

        cur_day_sym_num: int = daily_sym_num_dict[str(day)] #获取今天有多少个股票
        present_sym_start_index: int = sum(list(daily_sym_num_dict.values())[:day]) # 不包含currday
        for sec in range(config['daily_secs'] - config['seq_len']):

            X = data[present_sym_start_index:present_sym_start_index + cur_day_sym_num, 
                    sec:sec + config['seq_len'], 
                    :]
            y = labels[present_sym_start_index:present_sym_start_index + cur_day_sym_num, 
                    sec + config['seq_len'],  
                    0] 

            yield X, y

class DataIterable:
    def __init__(self,
                data_dir: str, 
                label_dir: str, 
                dict_path: str = './daily_sym_num_dict.json', 
                is_train: bool = True,):
        '''
        read data from file and return a Iterator
        reference: https://dogwealth.github.io/2021/07/08/Pytorch——DataLoader源码学习笔记/
        '''
        with open(dict_path, 'r') as f:
            self.daily_sym_num_dict: dict = json.load(f)  # 这里存储了每一天有多少只股票参与计算

        self.data: torch.tensor = torch.tensor(np.load(data_dir), dtype=torch.float32)
        self.labels: torch.tensor = torch.tensor(np.load(label_dir), dtype=torch.int64)
        # data (total_sym, time, feature_size)
        # 例如(48*10, 3998, 2000)
        self.days = config['train_days'] if is_train is True else 64 - config['train_days']
        
    def __iter__(self):
        return gen_dataiter(self.data, self.labels,self.daily_sym_num_dict, self.days)
    


## Hints from GPT4
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class LargeCSVDataset(Dataset):
    def __init__(self, csv_file, chunk_size=1000):
        self.csv_file = csv_file
        self.chunk_size = chunk_size

    def __len__(self):
        # This might not be the most efficient way to get the length
        # of a large file, but for the sake of the example:
        return sum(1 for _ in pd.read_csv(self.csv_file, chunksize=self.chunk_size))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Compute the start and end row of the chunk
        start_row = idx * self.chunk_size
        end_row = (idx + 1) * self.chunk_size

        chunk_data = pd.read_csv(self.csv_file, skiprows=range(1, start_row),
                                 nrows=self.chunk_size)
        
        # Process your data here (convert it to tensors, preprocess, etc.)
        # For this example, let's assume you are trying to predict a value
        # based on other features in the CSV.
        x = torch.tensor(chunk_data.drop('target', axis=1).values, dtype=torch.float32)
        y = torch.tensor(chunk_data['target'].values, dtype=torch.float32)
        
        return x, y

'''
# Instantiate your dataset
dataset = LargeCSVDataset('path_to_large_file.csv', chunk_size=1000)

# Use DataLoader to handle batching
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Training loop
for epoch in range(0):
    for batch_x, batch_y in dataloader:
        # Your training code here
        pass
'''
if __name__ == '__main__':
    train_iter = DataIterable(config['data_dir'],config['label_dir'])
    X, y = next(iter(train_iter))
    print(X, y)
    print(X.shape, y.shape)