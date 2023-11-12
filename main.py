import torch
import time
import json
import psutil
import os
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Iterator
# -- Personal Modules -- 
from modules.transformer import TransformerClassifier
from modules.config import config
from dataset import LOBDataset
from modules.functions import *


def train(model: nn.Module,
          train_loader: DataLoader, valid_loader: DataLoader,
          loss,
          optimizer,
          config: dict):

    print(f'On {config["device"]} : {count_parameters(model)} parameters to train...')
    model.apply(initialize_weight)
    train_losses = []
    valid_losses = []
    best_valid_loss = 1e10
    t1 = time.time()
    mem_usage = dict()

    process = psutil.Process(os.getpid())
    for epoch in tqdm(range(config['num_epochs']), desc='Epochs'):
        # training
        model.train()  # Turn on the Training Mode
        epoch_train_loss = []

        for X, y in tqdm(train_loader, desc='Processing Train'):

            X = X.to(config['device'])
            y = y.to(config['device'])  # 只要一个标签就可以
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
        for X, y in tqdm(valid_loader, 'Processing Valid'):

            # X, y = X.to(config['device']), y.to(config['device'])
            X, y = X.to(
                config['device']), y.to(config['device'])
            # Compute prediction error
            with torch.no_grad():
                pred = model(X)
            l = loss(pred, y)
            epoch_valid_loss.append(l.item())

        valid_loss = sum(epoch_valid_loss) / len(epoch_valid_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, config['model_path'])
        # Record memory usage
        mem_usage[epoch+1] = process.memory_full_info().uss / (1024 * 1024)
        print(f"\nepoch:{epoch+1}, Mem Usage: {mem_usage[epoch+1]:.2f}, MB.")
        print(f"GPU Sleeping...")
        # time.sleep(300)  # Protect the GPU from over heating

    t2 = time.time() - 300 * config['num_epochs']
    elapsed_time = (t2 - t1) / 60
    save_log(train_losses=train_losses, 
             valid_losses=valid_losses,
             mem_usage=mem_usage,
             config=config,
             time_cost=elapsed_time)

    print(f'Training Finished with Best Valid Loss: {best_valid_loss:.3f}')
    print(f'Total Time Cost in Training: {elapsed_time:.2f} mins.')

    plot_loss(train_losses, valid_losses,)

    return True


def train_Transformer() -> bool:
    train_iter = DataLoader(LOBDataset(is_train=True, config=config), 
                            shuffle=False, batch_size=config['batch_size'])
    valid_iter = DataLoader(LOBDataset(is_train=False, config=config), 
                            shuffle=False, batch_size=config['batch_size'])

    model = TransformerClassifier(config).to(config['device'])
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])

    train(model, train_iter, valid_iter, loss, optimizer, config)
    return True


if __name__ == '__main__':

    """ 
    train_iter = DataIterable('./data/train_data.npy','./data/train_labels.npy', is_train=True)
    valid_iter = DataIterable('./data/valid_data.npy','./data/valid_labels.npy', is_train=False)
    """
    train_Transformer()
    # model = TransformerClassifier(config)
    # print(model)

    # pass