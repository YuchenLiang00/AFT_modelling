from typing import Callable
from numpy import record
from torch import nn
import matplotlib.pyplot as plt


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
    ax.legend(['train_loss', 'valid_loss'])
    ax.set_title('Losses of Train and Valid')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(train_losses))
    plt.ylim(0, 1)
    plt.savefig(output_path)

    return True
