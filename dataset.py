from copy import deepcopy
import time
import torch
import numpy as np
import json
from torch.utils.data import Dataset
from modules.config import config

import psutil
import os


class LOBDataset(Dataset):
    """ A packaged dataset to load a training set """
    # TODO 应该还是要用Dataset-DataLoader 的结构，并充分利用num_workers来加速数据读取
    def __init__(self, data_path:str, label_path:str, 
                 dict_path:str='./daily_sym_num_dict.json',
                 is_train:bool = True,
                 pred_label:int=0) -> None:
        super().__init__()
        self.is_train = is_train
        with open(dict_path, 'r') as f:
                self.daily_sym_num_dict: dict = json.load(f) # 这里存储了每一天有多少只股票参与计算
        self.data: torch.tensor = torch.tensor(np.load(data_path), dtype=torch.float32)
        self.labels: torch.tensor = torch.tensor(np.load(label_path), dtype=torch.int64)
        # data (total_sym, time, feature_size)
        # 例如(3998, 64*10, 2000)
        self.pred_label = pred_label  # label to predict

    """ 
    def __len__(self) -> int:
        # 返回的是训练集或验证集所有天的秒数和
        days = config['train_days'] if self.is_train is True else 64 - config['train_days']
        return self.data.shape[0] * days - config['seq_len'] # FIXME 要保证秒数指针是不能够跨过某一天的，并要重新探索临界条件。

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        cur_day, cur_sec = divmod(index, config['daily_secs']) # 获得现在index所指的天数和秒数
        cur_day_sym_num: int = self.daily_sym_num_dict[str(cur_day)] #获取今天有多少个股票

        present_sym_start_index: int = sum(list(self.daily_sym_num_dict.values())[:cur_day]) # 不包含currday
        # FIXME DataLoader要求每次穿进去的tensor的shape是一样的。但是这样取很明显会不一样
        # TODO Transformer真的关注时序信息吗？我们可不可以把所有的股票在时间上拼起来直接遍历？
        X = self.data[present_sym_start_index:present_sym_start_index + cur_day_sym_num,
                        cur_sec,
                        :]
        y = self.labels[present_sym_start_index:present_sym_start_index + cur_day_sym_num, # 只取结束的秒
                        cur_sec, 
                        self.pred_label] # 这里的0表示以label_5为预测对象
        return X, y  """
    
    def __len__(self) -> int:
        # 返回的是训练集或验证集所有天的秒数和
        print(self.data.shape[1] - config['seq_len'] )
        return self.data.shape[1] - config['seq_len'] 

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        X = self.data[:, index:index + config['seq_len'], :-1]
        y = self.labels[:, index + config['seq_len'], self.pred_label] # 这里的0表示以label_5为预测对象
        return X, y

def gen_dataiter(data, labels, daily_sym_num_dict, days,) -> tuple[np.ndarray, np.ndarray]:
    '''An iterator to get data sequentially'''
    # DEPRECATED 自定义的generator有内存泄露的风险
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
            # TODO: Try to find out whether deepcopy helps the memory overuse.
            X, y = deepcopy(X), deepcopy(y)  
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
        # DEPRECATED 自定义的generator有内存泄露的风险

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


# Ideas from DTQ
class RandomTimeDataset(Dataset):
    def init (self, data_val, data_idx, data_col, window_size=3,**kwargs) -> None:
        super().__init__(**kwargs)
        self.data_val = data_val
        self.data_idx = data_idx.reset_index(drop=True)
        self.data_col = data_col
        self.window_size = window_size
        self.key_code_map = dict(enumerate(data_idx['sym'].unique( )))
    def __len__(self):
        return len(self.key_code_map)
    
    def __getitem__(self,idx):
        code = self.key_code_map[idx]
        getIdx = self.data_idx.query("sym = @code")
        min_index,max_index = self.window_size,len(getIdx)
        if max_index < min_index:
            return self.__getitem__(np.random.randint(0,self.__len__()))#随机生成时间区域
        
        end_index = np.random.randint(min_index,max_index)
        start_indx = end_index-self.window_size
        data = self.data_val[getIdx.index[start_indx:end_index],:]
        X,y = data[:,:-1], data[:,-1:]
        return torch.tensor(X),torch.tensor(y)


class HisData:

    def getRandomData(self):
        """ 
        返回时间顺序随机的batch，每次的batch_size固定，适用于Transformer模型
        return 
            X; torch[batch_size, seq_len, feature_size]
            Y: torch[batch_size, seq_len, output_size]
        """
        for _ in range(self.total_seq // self.seq_len):
            rdl = DataLoader(self.rtd, batch_size=self.batch_size)
            for data in rdl:
                yield data[0], data[1]



if __name__ == '__main__':
    # train_iter = DataIterable('./data/train_data.npy', './data/train_labels.npy')
    train_iter = DataLoader(LOBDataset('./data/train_data.npy', './data/train_labels.npy'),shuffle=False)
    process = psutil.Process(os.getpid())
    time1 = time.time()
    for i, batch in enumerate(train_iter):
        mm_info = process.memory_full_info()
        print('\r',i, batch[0].shape, mm_info.uss / 1024 / 1024, "MB",end='',flush=True)
        # X, y = next(iter(train_iter))
        # print(X, y)
        # print(X.shape, y.shape)
    time2 = time.time() - time1
    print(f"\n{time2:.2f} sec")