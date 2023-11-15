import torch
import math
from torch import dropout, nn

'''
Stock Embeddings: If using a single model, 
consider adding an embedding layer for stock identifiers.
This way, the model can learn stock-specific embeddings, 
allowing it to capture some stock-specific nuances.
'''

class FeatureNorm(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        return
    
    def forward(self, X):
        """ Return the Feature-wise Normalization of A tensor """
        # X (batch_size, seq_len, feature_num)
        # Ensure no gradients are computed for the mean and std
        with torch.no_grad():
            mean = X.nanmean(dim=[0, 1], keepdim=True)
            X = X - mean
            X.nan_to_num_(nan=0)
            std = X.std(dim=[0, 1], keepdim=True)

        # Normalize the tensor
        # Adding a small constant to avoid division by zero
        normalized_x = X / (std + 1e-6)
        
        return normalized_x
    
    def _internalize_outliers(self, X):
        # TODO deal with out liers
        q1: torch.Tensor = torch.quantile(X, .01, dim=2)  # 小
        q99: torch.Tensor = torch.quantile(X, .99, dim=2)  # 大


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建⼀个⾜够⻓的P
        self.pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        X = position * div_term
        self.pe[:, :, 0::2] = torch.sin(X)
        self.pe[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.pe[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)  # TODO dropout 层，想加上，但是现在先算了
        self.attention = nn.MultiheadAttention(d_model, num_heads)
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


class TransformerClassifier(nn.Module):
    """
    TransformerClassifier 的输入维度应该是(batch_size, seq_len, embed_size)
    在我们的例子中 embed_size 就是 feature_size
    """

    def __init__(self, config: dict):
        super(TransformerClassifier, self).__init__()
        self.config = config
        self.feature_norm = FeatureNorm()
        
        self.pe = PositionalEncoding(config['hidden_dim'], config['dropout'])
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config['hidden_dim'], config['num_heads'], config['dropout'])
            for _ in range(config['num_layers'])
        ])

        self.fc = nn.Linear(config['hidden_dim'] *
                            config['seq_len'], config['output_dim'])

    def forward(self, X):
        """ X: (batch_size, seq_len, feature_size) """
        # Feature-wise Normalization
        X = self.feature_norm(X)

        # Positional Encoding
        X = self.pe(X) if self.config['pos_enco'] is True else X

        # Encoder layers
        encoder_output = X  # 不会产生copy
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)
        # encoder_output: (batch_size, seq_len, hidden_size=embed_size=feature_size)

        # Output layer
        output = self.fc(encoder_output.flatten(
            start_dim=1))  # 将seq_len, feature_size展平
        # output: (batch_size, class_num), 在本例中，class_num = 3
        return output


if __name__ == '__main__':
    from config import config
    model = TransformerClassifier(config)
    print('OK')
