from typing import Iterator
import torch
import time
from torch import nn
from tqdm import tqdm

from modules.transformer import TransformerClassifier
from modules.config import config
from dataset import DataIterable
from modules.functions import *


def train(model: nn.Module,
          train_loader: Iterator, valid_loader:Iterator,
          loss,
          optimizer,
          config: ):
    
    print(f'On {config["device"]} : {count_parameters(model)} parameters to train...')

    train_losses = []
    valid_losses = []
    best_valid_loss = 1e10
    t1 = time.time()

    for epoch in tqdm(range(config['num_epochs'])):
        # training
        model.train() # Turn on the Training Mode
        epoch_train_loss = []
        for X, y in train_loader:

            # 暂时的解决方法，因为咱们现在的features只有23维
            X, y = X[:,:,:-1].to(config['device']), y.to(config['device']) # 只要一个标签就可以
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
        for X, y in valid_loader:

            # X, y = X.to(config['device']), y.to(config['device'])
            X, y = X[:,:,:-1].to(config['device']), y.to(config['device'])
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
    
    train_iter = DataIterable('./data/train_data.npy','./data/train_labels.npy', is_train=True)
    valid_iter = DataIterable('./data/valid_data.npy','./data/valid_labels.npy', is_train=False)
    # m = next(iter(train_iter))
    # print(m[0].shape, m[0].dtype, m[1].shape,m[1].dtype)

    model = TransformerClassifier(config).to(config['device'])
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])

    train(model, train_iter, valid_iter, loss, optimizer, config)