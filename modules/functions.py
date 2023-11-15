from torch import nn
import matplotlib.pyplot as plt
import json
from datetime import datetime


def initialize_weight(x):
    """ 初始化模型的权重 """
    if isinstance(x, nn.LayerNorm) is True:
        return
    if hasattr(x, 'weight') is True:
        nn.init.xavier_normal_(x.weight)
    if (x,'bias') is True:
        nn.init.constant_(x.bias, 0)


def count_parameters(model):
    """ 计算模型的总参数量 """
    return sum(p.numel() for p in model.parameters() if p.requires_grad is True)


def plot_loss(train_losses: list, valid_losses: list,
              train_accs:list, valid_accs:list):
    """ 绘制损失函数、精度图线 """
    # TODO 绘制测试精度

    fig, ax = plt.subplots(1)
    ax.plot(train_losses,'-')
    ax.plot(valid_losses,'-')
    ax.plot(train_accs,'-.')
    ax.plot(valid_accs,'m-.')
    ax.legend(['train_loss', 'valid_loss','train_accs', 'valid_accs'])
    ax.set_title('Losses and Accuracies')
    plt.xlabel('epochs')
    plt.grid()
    # ax.set_title('Accuracies')
    # plt.xlabel('epochs')
 

    str_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'./figures/Figure_{str_now}.png')

    return True


def save_log(**kwargs):
    """ 记录想要保存的日志文件为json格式 """
    str_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    kwargs['time'] = str_now
    with open(f'./logs/log_{str_now}.json', 'w') as f:
        json.dump(kwargs, f, indent=4)
    return


if __name__ == '__main__':
    a = [1,2,3,4]
    b = [2,3,4,5]
    c = [10, 9 ,8 ,7]
    d = [10,9,9,8]
    plot_loss(a,b,c,d)
