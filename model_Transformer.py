import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from torch import nn

from tqdm import tqdm

from modules.transformer import Transformer
from modules.config import config
from modules.dataset import *


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
    ax.set_title('Losses of Train and Valid')
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