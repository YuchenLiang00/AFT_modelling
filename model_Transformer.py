import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import Any
from tqdm import tqdm

from moduals.transformer import Transformer
from moduals.config import config


class LOBDataset(Dataset):

    def __init__(self, data: torch.tensor, labels:torch.tensor) -> None:
        super().__init__()
        # if is_train is True:
        self.data = data
        self.labels = labels
        

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index) -> Any:
        return self.data[index], self.labels[index].long()

    # TODO: 划分训练集和测试集


def get_dataloader(batch_size:int, train_ratio:float, label_idx: int = 0,
                   )->tuple[DataLoader,DataLoader]:
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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model: Transformer,
          train_loader: DataLoader, valid_loader:DataLoader,
          loss,
          optimizer,
          config: dict):
    print(f'On {config["device"]} : {count_parameters(model):,} parameters to train...')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    train_losses = []
    valid_losses = []
    best_valid_loss = 1e10

    for epoch in tqdm(range(config['num_epochs'])):
        # train
        model.train()

        epoch_train_loss = []
        for batch in train_loader:
            X, y = batch
            X, y = X.to(config['device']), y.to(config['device'])
            # Compute prediction error
            optimizer.zero_grad()
            pred = model(X)
            l = loss(pred, y).sum()
            # Backpropagation
            l.backward()
            optimizer.step()
            epoch_train_loss.append(l.item())

        train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        train_losses.append(train_loss)
        # tqdm.write(
        #     f"[ Train | {epoch + 1:03d}/{config['num_epochs']:03d} ] loss = {train_loss:.5f}"
        # )

        # validation
        model.eval()
        epoch_valid_loss = []
        for batch in valid_loader:
            X, y = batch
            X, y = X.to(config['device']), y.to(config['device'])
            # Compute prediction error
            with torch.no_grad():
                pred = model(X)
            l = loss(pred, y).sum()
            epoch_valid_loss.append(l.item())
        valid_loss = sum(epoch_valid_loss) / len(epoch_valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, config['model_path'])
        valid_losses.append(valid_loss)
        # tqdm.write(
        #     f"[ Valid | {epoch + 1:03d}/{config['num_epochs']:03d} ] loss = {valid_loss:.5f}"
        # )

    print(f'finishing training for best valid loss: {best_valid_loss}')
    model = torch.load(config['model_path'])
    fig, ax = plt.subplots(1)
    ax.plot(train_losses)
    ax.plot(valid_losses)
    ax.legend(['train_loss', 'test_loss'])
    plt.show()
    return model

if __name__ == '__main__':
    
    train_iter, valid_iter = get_dataloader(batch_size=config['batch_size'], 
                                            train_ratio=config['train_ratio'])
    m = next(iter(train_iter))
    # print(m[0].shape, m[0].dtype, m[1].shape,m[1].dtype)

    model = Transformer(config).to(config['device'])
    # model = TransformerClassifier(config).to(config['device'])
    loss = nn.CrossEntropyLoss(reduction='none').to(config['device'])
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])

    train(model, train_iter, valid_iter, loss, optimizer, config)


