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

    valid_len = len(valid_loader)
    train_len = len(train_loader)
    train_losses = torch.zeros(config['num_epochs']) + 1e8
    valid_losses = torch.zeros(config['num_epochs']) + 1e8
    train_accs = torch.zeros(config['num_epochs'])
    valid_accs = torch.zeros(config['num_epochs'])
    best_valid_loss = 1e8
    mem_usage = dict()
    process = psutil.Process(os.getpid())

    t1 = time.perf_counter()

    for epoch in tqdm(range(config['num_epochs']), desc='Epochs'):
        # training
        model.train()  # Turn on the Training Mode
        epoch_train_loss = torch.zeros(train_len) + 1e8
        epoch_train_acc = torch.zeros(train_len)
        for i, (X, y) in enumerate(tqdm(train_loader, desc='Processing Train')):
            X = X.to(config['device'])
            y = y.to(config['device'])
            # Compute prediction error
            optimizer.zero_grad()
            pred = model(X)
            l = loss(pred, y)
            # Backpropagation
            l.backward()
            optimizer.step()

            # 保存误差和精度
            epoch_train_loss[i] = l.item()
            y_hat = pred.argmax(dim=1)
            # epoch_train_acc.append((sum(y_hat == y) / y.shape[0]).item()) # 非常慢，时间消耗是下面的600倍
            # epoch_train_acc[i] = ((y_hat == y).sum() / y.shape[0]).item() # 下面的更快，耗时是本条的78%
            epoch_train_acc[i] = (y_hat == y).float().mean().item()


        train_losses[epoch] = epoch_train_loss.mean()
        train_accs[epoch] = epoch_train_acc.mean()

        print(f"\nGPU Sleeping...")
        time.sleep(300)  # Protect the GPU from over heating

        # validation
        model.eval()  # Trun on the Evaluation Mode
        epoch_valid_loss = torch.zeros(valid_len) + 1e8
        epoch_valid_acc = torch.zeros(valid_len)
        for i, (X, y) in enumerate(tqdm(valid_loader, 'Processing Valid')):

            # X, y = X.to(config['device']), y.to(config['device'])
            X = X.to(config['device'])
            y = y.to(config['device'])
            # Compute prediction error
            with torch.no_grad():
                pred = model(X)
                l = loss(pred, y)
                epoch_valid_loss[i] = l.item()
                y_hat = pred.argmax(dim=1)
                epoch_valid_acc[i] = (y_hat == y).float().mean().item()

        valid_losses[epoch] = epoch_valid_loss.mean()
        valid_accs[epoch] = epoch_valid_acc.mean()

        if valid_losses[epoch] < best_valid_loss:
            best_valid_loss = valid_losses[epoch]
            torch.save(model, config['model_path'])
            torch.save(optimizer.state_dict(), config['optimizer_path'])
        # Record memory usage
        mem_usage[epoch+1] = process.memory_full_info().uss / (1024 * 1024)
        # print(f"\nepoch:{epoch+1}, Mem Usage: {mem_usage[epoch+1]:.2f}, MB.")

        print(f"\nGPU Sleeping...")
        time.sleep(60)  # Protect the GPU from over heating

    t2 = time.perf_counter() - 360 * config['num_epochs']
    elapsed_time = (t2 - t1) / 60

    print(f'Training Finished with Best Valid Loss: {best_valid_loss:.3f}')
    print(f'Total Time Cost: {elapsed_time:.2f} mins.')

    save_log(num_params=num_params,
             train_losses=train_losses.tolist(),
             valid_losses=valid_losses.tolist(),
             train_accs=train_accs.tolist(),
             valid_accs=valid_accs.tolist(),
             mem_usage=mem_usage,
             config=config,
             time_cost_mins=elapsed_time)

    plot_loss(train_losses, valid_losses, train_accs, valid_accs) 

    return True


def train_Transformer() -> bool:
    train_iter = DataLoader(LOBDataset(is_train=True, config=config),
                            shuffle=False, batch_size=config['batch_size'])
    valid_iter = DataLoader(LOBDataset(is_train=False, config=config),
                            shuffle=False, batch_size=config['batch_size'])

    # 如果是从头开始训练，则需要初始化，但是如果model是load进来的，则一定要去掉这句话。
    model = TransformerClassifier(config).to(config['device'])
    model.apply(initialize_weight)

    # load 模型继续训练
    # model = torch.load('./transformer_models/model_round_4').to(config['device'])
    # optimizer = torch.optim.SGD().load_state_dict(config['optimizer_path'])
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], 
                                 weight_decay=config['weight_decay'])
    # optimizer.load_state_dict(torch.load('./transformer_models/optimizer_round_4'))
   

    train(model, train_iter, valid_iter, loss, optimizer, config)
    return True


if __name__ == '__main__':

    train_Transformer()
    # pass
