import torch
from torch import nn

class MovingAvg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DLinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = SeriesDecomp(kernel_size)
        self.individual = config.individual
        self.channels = config.enc_in

        if self.individual is True:
            self.linear_seasonal = nn.ModuleList()
            self.linear_trend = nn.ModuleList()
            self.linear_decoder = nn.ModuleList()
            for i in range(self.channels):
                self.linear_seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.linear_seasonal[i].weight = nn.Parameter(
                    (1/self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.linear_trend.append(nn.Linear(self.seq_len, self.pred_len))
                self.linear_trend[i].weight = nn.Parameter(
                    (1/self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.linear_decoder.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
            self.linear_decoder = nn.Linear(self.seq_len, self.pred_len)
            self.linear_seasonal.weight = nn.Parameter(
                (1/self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.linear_trend.weight = nn.Parameter(
                (1/self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual is True:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                         dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                        dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.linear_seasonal(seasonal_init)
            trend_output = self.linear_trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2,1)  # to [Batch, Output length, Channel]