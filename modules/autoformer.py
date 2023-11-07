import torch.nn as nn
import torch
import math
from torch import nn


class MyAutoCorrelation(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

    def _autocorrelation(self, query_states, key_states):
        """
        Computes autocorrelation(Q,K) using `torch.fft`.
        Think about it as a replacement for the QK^T in the self-attention.
        
        Assumption: states are resized to same shape of [batch_size, time_length, embedding_dim].
        """
        query_states_fft = torch.fft.rfft(query_states, dim=1)
        key_states_fft = torch.fft.rfft(key_states, dim=1)
        attn_weights = query_states_fft * torch.conj(key_states_fft)
        attn_weights = torch.fft.irfft(attn_weights, dim=1)
        
        return attn_weights


    def _time_delay_aggregation(self, attn_weights,
                                value_states, 
                                autocorrelation_factor=2):
        """
        Computes aggregation as value_states.roll(delay)* top_k_autocorrelations(delay).
        The final result is the autocorrelation-attention output.
        Think about it as a replacement of the dot-product between attn_weights and value states.
        
        The autocorrelation_factor is used to find top k autocorrelations delays.
        Assumption: value_states and attn_weights shape: [batch_size, time_length, embedding_dim]
        """
        bsz, num_heads, tgt_len, channel = ...
        time_length = value_states.size(1)
        autocorrelations = attn_weights.view(bsz, num_heads, tgt_len, channel)

        # find top k autocorrelations delays
        top_k = int(autocorrelation_factor * math.log(time_length))
        autocorrelations_mean = torch.mean(autocorrelations, dim=(1, -1)) # bsz x tgt_len
        top_k_autocorrelations, top_k_delays = torch.topk(autocorrelations_mean, top_k, dim=1)

        # apply softmax on the channel dim
        top_k_autocorrelations = torch.softmax(top_k_autocorrelations, dim=-1) # bsz x top_k

        # compute aggregation: value_states.roll(delay)* top_k_autocorrelations(delay)
        delays_agg = torch.zeros_like(value_states).float() # bsz x time_length x channel
        for i in range(top_k):
            value_states_roll_delay = value_states.roll(shifts=-int(top_k_delays[i]), dims=1)
            top_k_at_delay = top_k_autocorrelations[:, i]
            # aggregation
            top_k_resized = top_k_at_delay.view(-1, 1, 1).repeat(num_heads, tgt_len, channel)
            delays_agg += value_states_roll_delay * top_k_resized

        attn_output = delays_agg.contiguous()
        return attn_output
    
    def forward(self, Q, K, V):
        attention_weights = self._autocorrelation(Q, K)
        time_delay_agg_output = self._time_delay_aggregation(attention_weights, V)

        return time_delay_agg_output


class MyDecompositionLayer(nn.Module):
    """
    Returns the trend and the seasonal parts of the time series.

    将Series Decomp Block在模型中使用将不仅能提取输入的高阶时间信息
    更能提取模型中间隐变量的高阶时间信息。
    """

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0) # moving average 滑动窗口的均值

    def forward(self, x):
        """Input shape: (batch_size, seq_len, embed_dim)"""
        # padding on the both ends of time series
        num_of_pads = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, num_of_pads, 1)
        end = x[:, -1:, :].repeat(1, num_of_pads, 1)
        x_padded = torch.cat([front, x, end], dim=1)

        # calculate the trend and seasonal part of the series
        x_trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        x_seasonal = x - x_trend # 保留季节性的平滑序列

        return x_seasonal, x_trend
    

class MyEncoderLayer(nn.Module):
    def __init__(self,d_model, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.auto_corr = AutoCorrelation()
        self.series_decomp1 = DecompositionLayer(
            #TODO kernel_size
        )
        self.series_decomp2 = DecompositionLayer(
            #TODO kernel_size
        )
        self.feedforward = nn.Sequential(nn.Linear(d_model, 4 * d_model),
                                         nn.ReLU(),
                                         nn.Linear(4 * d_model, d_model))
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, input):

        auto_corr_output = self.auto_corr(input, input,input)
        auto_corr_output = self.norm1(input + auto_corr_output)

        series_decomp_output = self.series_decomp1(auto_corr_output)

        ff_output = self.feedforward(series_decomp_output)
        ff_output = self.norm2(series_decomp_output + ff_output)

        output = self.series_decomp2(ff_output)
        return output


class MyAutoformerClassifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        pass

    def forward(self, input):

        return


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]]
                              for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                    1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + \
                delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                    1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(
                    1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(
            0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(
                0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(
                0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
