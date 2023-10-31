import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from config import config

class LOBDataset(Dataset):

    def __init__(self, data:torch.tensor, labels:torch.tensor) -> None:
        super().__init__()
        # if is_train is True:
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        # TODO 要对取数据的操作做详细的处理。
        # 包括sym,date, time的细致处理
        return self.data[index], self.labels[index].long()

    # TODO: 划分训练集和测试集


def get_dataloader(batch_size:int, 
                   train_ratio:float, 
                   label_idx: int = 0, # 需要预测的标签序号，这里先预测label_5的
                  )->tuple[DataLoader, DataLoader]:
    '''read data from file and return a tuple of DataLoader'''
    data_dir = config['data_dir']
    label_dir = config['label_dir']

    data: torch.tensor = torch.from_numpy(np.load(data_dir))
    labels: torch.tensor = torch.from_numpy(np.load(label_dir))
    # TODO 暴力切一刀是有问题的
    # 将切分Valid 和Train 放到LOBDataset外面来实现，可以尽可能地减小LOBDataset的大小
    train_data = data[:round(train_ratio * data.shape[0])]
    train_labels = labels[:round(train_ratio * data.shape[0]),label_idx]

    valid_data = data[round(train_ratio * data.shape[0]):]
    valid_labels = labels[round(train_ratio * data.shape[0]):,label_idx]

    train_iter = DataLoader(LOBDataset(train_data, train_labels), batch_size=batch_size, shuffle=False)
    valid_iter = DataLoader(LOBDataset(valid_data, valid_labels), batch_size=batch_size, shuffle=False)
    return train_iter, valid_iter


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


# Instantiate your dataset
dataset = LargeCSVDataset('path_to_large_file.csv', chunk_size=1000)

# Use DataLoader to handle batching
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Training loop
for epoch in range(0):
    for batch_x, batch_y in dataloader:
        # Your training code here
        pass