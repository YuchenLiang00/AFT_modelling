from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from modules.config import config

""" REFERENCE 
Two different dataset types: https://pytorch.org/docs/stable/data.html#dataset-types
"""


class LOBDataset(Dataset):
    """ A packaged dataset to load a training set """
    # TODO 应该还是要用Dataset-DataLoader 的结构，并充分利用num_workers来加速数据读取

    def __init__(self,
                 is_train: bool,
                 config:dict,
                 pred_label: int = 0,
                 ) -> None:
        super().__init__()
        data_path = './data/train_data.npy' if is_train is True else './data/valid_data.npy'
        label_path = './data/train_labels.npy' if is_train is True else './data/valid_labels.npy'
        self.data: torch.Tensor = torch.from_numpy(np.load(data_path))
        self.label: torch.Tensor = torch.from_numpy(np.load(label_path))
        # 选出对应的预测维度(stock_num, secs)
        self.label: torch.Tensor = self.label[:, :, pred_label]  # 这里的pred_label表示预测对象
        self.seq_len: int = config['seq_len']

    def __len__(self) -> int:
        return self.data.shape[0] * (self.data.shape[1] - self.seq_len + 1)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        start_sec, stock_id = divmod(index, self.data.shape[0])
        end_sec = start_sec + self.seq_len
        X = self.data[stock_id, start_sec:end_sec, :]
        y = self.label[stock_id, end_sec - 1] 
        return X, y

if __name__ == '__main__':
    config['seq_len'] = 1
    dataset=LOBDataset(is_train=True, config=config)
    train_iter = DataLoader(dataset, batch_size=256, shuffle=False)
    # print(len(dataset))

    for i, batch in enumerate(tqdm(train_iter)):
        print(batch[0].shape)

