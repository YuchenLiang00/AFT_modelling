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


def plot_loss(train_losses: list, valid_losses: list):
    """ 绘制损失函数、精度图线 """
    # TODO 绘制测试精度
    fig, ax = plt.subplots(1)
    ax.plot(train_losses)
    ax.plot(valid_losses)
    ax.legend(['train_loss', 'valid_loss'])
    ax.set_title('Losses of Train and Valid')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_losses))
    
    plt.ylim(0, 1)
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
    d = {1: 'abs', 2: 'cde'}
    e = {'Anna': 23, 'Logan': 36}
    save_log(d=d, e=e)
