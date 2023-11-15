import torch
import time
import psutil
import os
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
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
    num_params = count_parameters(model)
    print(f'On {config["device"]} : {num_params} parameters to train...')
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    best_valid_loss = 1e10
    mem_usage = dict()
    process = psutil.Process(os.getpid())

    t1 = time.time()

    for epoch in tqdm(range(config['num_epochs']), desc='Epochs'):
        # training
        model.train()  # Turn on the Training Mode
        epoch_train_loss = []
        epoch_train_acc = []
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

            y_hat = pred.argmax(dim=1)
            epoch_train_acc.append((sum(y_hat == y) / len(y)).item())
            epoch_train_loss.append(l.item())

        train_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        train_acc = sum(epoch_train_acc) / len(epoch_train_acc)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # validation
        model.eval()  # Trun on the Evaluation Mode
        epoch_valid_loss = []
        epoch_valid_acc = []
        for X, y in tqdm(valid_loader, 'Processing Valid'):

            # X, y = X.to(config['device']), y.to(config['device'])
            X = X.to(config['device'])
            y = y.to(config['device'])
            # Compute prediction error
            with torch.no_grad():
                pred = model(X)
                l = loss(pred, y)
                epoch_valid_loss.append(l.item())
                y_hat = pred.argmax(dim=1)
                epoch_valid_acc.append((sum(y_hat == y) / len(y)).item())

        valid_loss = sum(epoch_valid_loss) / len(epoch_valid_loss)
        valid_losses.append(valid_loss)
        valid_acc = sum(epoch_valid_acc) / len(epoch_valid_acc)
        valid_accs.append(valid_acc)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model, config['model_path'])
            torch.save(optimizer.state_dict(), config['optimizer_path'])
        # Record memory usage
        mem_usage[epoch+1] = process.memory_full_info().uss / (1024 * 1024)
        # print(f"\nepoch:{epoch+1}, Mem Usage: {mem_usage[epoch+1]:.2f}, MB.")

        print(f"\nGPU Sleeping...")
        time.sleep(300)  # Protect the GPU from over heating

    t2 = time.time() - 300 * config['num_epochs']
    elapsed_time = (t2 - t1) / 60
    save_log(train_losses=train_losses,
             num_params=num_params,
             valid_losses=valid_losses,
             mem_usage=mem_usage,
             config=config,
             time_cost_mins=elapsed_time)

    print(f'Training Finished with Best Valid Loss: {best_valid_loss:.3f}')
    print(f'Total Time Cost: {elapsed_time:.2f} mins.')

    plot_loss(train_losses, valid_losses, train_accs,
              valid_accs)  # TODO绘制测试精度！！！！！

    return True


def train_Transformer() -> bool:
    train_iter = DataLoader(LOBDataset(is_train=True, config=config),
                            shuffle=True, batch_size=config['batch_size'])
    valid_iter = DataLoader(LOBDataset(is_train=False, config=config),
                            shuffle=False, batch_size=config['batch_size'])

    # 如果是从头开始训练，则需要初始化，但是如果model是load进来的，则一定要去掉这句话。
    # model = TransformerClassifier(config).to(config['device'])
    # model.apply(initialize_weight)

    # load 模型重新训练
    model = torch.load('./transformer_models/model_round_3').to(config['device'])
    # optimizer = torch.optim.SGD().load_state_dict(config['optimizer_path'])
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)

    train(model, train_iter, valid_iter, loss, optimizer, config)
    return True


if __name__ == '__main__':

    train_Transformer()
    # model = TransformerClassifier(config)
    # print(model)

    # pass
