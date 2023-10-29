import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建⼀个⾜够⻓的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000,
                               torch.arange(0, num_hiddens, 2,
                                             dtype=torch.float32) /
                              num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = int(d_model / num_heads)

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        attention_weights = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.depth, dtype=torch.float32))
        attention_weights = torch.softmax(attention_weights, dim=-1)

        output = torch.matmul(attention_weights, V)
        output = self._combine_heads(output)

        output = self.W_O(output)
        return output

    def _split_heads(self, tensor):
        tensor = tensor.view(tensor.size(0), -1, self.num_heads, self.depth)
        return tensor.transpose(1, 2)

    def _combine_heads(self, tensor):
        tensor = tensor.transpose(1, 2).contiguous()
        tensor = tensor.view(tensor.size(0), -1, self.num_heads * self.depth)
        return tensor

'''
class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = nn.Sequential(nn.Linear(d_model, 4 * d_model),
                                         nn.ReLU(),
                                         nn.Linear(4 * d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_output = self.attention(x, x, x)
        attention_output = self.norm1(x + attention_output)

        feedforward_output = self.feedforward(attention_output)
        output = self.norm2(attention_output + feedforward_output)
        return output


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads)
        self.feedforward = nn.Sequential(nn.Linear(d_model, 4 * d_model),
                                         nn.ReLU(),
                                         nn.Linear(4 * d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        self_attention_output = self.self_attention(x, x, x)
        self_attention_output = self.norm1(x + self_attention_output)

        encoder_attention_output = self.encoder_attention(
            self_attention_output, encoder_output, encoder_output)
        encoder_attention_output = self.norm2(self_attention_output +
                                              encoder_attention_output)

        feedforward_output = self.feedforward(encoder_attention_output)
        output = self.norm3(encoder_attention_output + feedforward_output)
        return output
'''

class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads):
        super(EncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads) # TODO mask
        self.feedforward = nn.Sequential(nn.Linear(d_model, 4 * d_model),
                                         nn.ReLU(),
                                         nn.Linear(4 * d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attention_output = self.attention(x, x, x)[0]
        attention_output = self.norm1(x + attention_output)

        feedforward_output = self.feedforward(attention_output)
        output = self.norm2(attention_output + feedforward_output)
        return output


class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, num_heads)
        self.encoder_attention = nn.MultiheadAttention(d_model, num_heads)
        self.feedforward = nn.Sequential(nn.Linear(d_model, 4 * d_model),
                                         nn.ReLU(),
                                         nn.Linear(4 * d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        self_attention_output = self.self_attention(x, x, x)[0]
        self_attention_output = self.norm1(x + self_attention_output)

        encoder_attention_output = self.encoder_attention(
            self_attention_output, encoder_output, encoder_output)[0]
        encoder_attention_output = self.norm2(self_attention_output +
                                              encoder_attention_output)

        feedforward_output = self.feedforward(encoder_attention_output)
        output = self.norm3(encoder_attention_output + feedforward_output)
        return output


class Transformer(nn.Module):

    def __init__(self, config: dict):
        super(Transformer, self).__init__()
        self.config = config
        self.pe = PositionalEncoding(config['hidden_dim'], config['dropout'])
        self.input_layer = nn.Linear(config['input_dim'], config['hidden_dim'])
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config['hidden_dim'], config['num_heads'])
            for _ in range(config['num_layers'])
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config['hidden_dim'], config['num_heads'])
            for _ in range(config['num_layers'])
        ])
        self.output_layer = nn.Linear(config['hidden_dim'], config['output_dim'])

    def forward(self, x):

        # Input layer
        x = self.input_layer(x)
        x = self.pe(x) if self.config['pe'] is True else x

        # Encoder layers
        # encoder_output = x.transpose(0, 1)
        encoder_output = x
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)

        # Decoder layers
        # decoder_output = encoder_output[-1, :, :].unsqueeze(0)
        decoder_output = encoder_output
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output)

        # Output layer
        output = self.output_layer(
            decoder_output)
        return output


class TransformerClassifier(nn.Module):
    def __init__(self, config:dict,):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Embedding(config['input_dim'], config['hidden_dim'])
        self.transformer = nn.Transformer(
            d_model=config['hidden_dim'],
            num_encoder_layers=config['num_layers'],
            num_decoder_layers=config['num_layers'],
            nhead=config['num_heads'],
            dim_feedforward=config['hidden_dim'],
            dropout=config['dropout'])
        self.fc = nn.Linear(config['hidden_dim'], config['output_dim'])

    def forward(self, X):
        embedded = self.embedding(X)
        transformed = self.transformer(embedded)
        logits = self.fc(transformed)
        return logits


if __name__ == '__main__':
    from config import config
    model = Transformer(config)
    print('OK')