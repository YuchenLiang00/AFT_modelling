import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from modules.transformer import Transformer
from modules.config import config


class LOBDataset(Dataset):

    def __init__(self, data: torch.tensor, labels:torch.tensor) -> None:
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


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad is True)


def plot_loss(train_losses: list, valid_losses: list, output_path: str):
    fig, ax = plt.subplots(1)
    ax.plot(train_losses)
    ax.plot(valid_losses)
    ax.legend(['train_loss', 'test_loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(1, len(train_losses))
    plt.ylim(0, 1)
    plt.savefig(output_path)

    return True


def train(model: Transformer,
          train_loader: DataLoader, valid_loader:DataLoader,
          loss,
          optimizer,
          config: dict):
    
    print(f'On {config["device"]} : {count_parameters(model)} parameters to train...')

    train_losses = []
    valid_losses = []
    best_valid_loss = 1e10
    t1 = time.time()

    for epoch in tqdm(range(config['num_epochs'])):
        # train
        model.train() # Turn on the Training Mode
        epoch_train_loss = []
        for batch in train_loader:
            X, y = batch
            X, y = X.to(config['device']), y.to(config['device'])
            # Compute prediction error
            optimizer.zero_grad()
            pred = model(X)
            l = loss(pred, y) 
            # Backpropagation
            l.backward()
            optimizer.step()
            epoch_train_loss.append(l.item())

        train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        train_losses.append(train_loss)

        # validation
        model.eval()  # Trun on the Evaluation Mode
        epoch_valid_loss = []
        for batch in valid_loader:
            X, y = batch
            X, y = X.to(config['device']), y.to(config['device'])
            # Compute prediction error
            with torch.no_grad():
                pred = model(X)
            l = loss(pred, y)
            epoch_valid_loss.append(l.item())
        valid_loss = sum(epoch_valid_loss) / len(epoch_valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, config['model_path'])

        valid_losses.append(valid_loss)

    t2 = time.time()

    print(f'Training Finished with Best Valid Loss: {best_valid_loss:.3f}')
    print(f'Total Time Cost in Training: {(t2 - t1) / 60 :.2f} mins.')

    plot_loss(train_losses, valid_losses, output_path='./Figure_1.png')

    return True

if __name__ == '__main__':
    
    train_iter, valid_iter = get_dataloader(batch_size=config['batch_size'], 
                                            train_ratio=config['train_ratio'])
    # m = next(iter(train_iter))
    # print(m[0].shape, m[0].dtype, m[1].shape,m[1].dtype)

    model = Transformer(config).to(config['device'])
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])

    train(model, train_iter, valid_iter, loss, optimizer, config)