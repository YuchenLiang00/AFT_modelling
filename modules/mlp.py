import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, config:dict,) -> None:
        super().__init__()
        self.input_layer = nn.Linear(config['input_dim'], config['hidden_dim'])
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(config['dropout'])
        self.hidden_layers = nn.Sequential(nn.Linear(config['hidden_dim'], config['hidden_dim']),
                                           nn.ReLU(),
                                           nn.Linear(config['hidden_dim'], config['hidden_dim']),
                                           nn.ReLU())
        self.output_layer = nn.Linear(config['hidden_dim'], config['output_dim'])
        # self.dropout2 = nn.Dropout(dropout)

    def forward(self, X):
        X = self._feature_norm(X).squeeze()
        X = self.dropout1(self.relu(self.input_layer(X)))
        X = self.hidden_layers(X)
        output = self.output_layer(X)
        return output
    
    def _feature_norm(self, X):
        X -= X.nanmean(dim=[0, 1], keepdim=True)
        X.nan_to_num_(nan=0) # replace nan to 0
        # Normalize the tensor
        # Adding a small constant to avoid division by zero
        X /= X.std(dim=[0, 1], keepdim=True) + 1e-6
        return X

if __name__ == '__main__':
    X = torch.randn(512, 1, 426)
    model = MLP(426, 64, 3, .1)
    print(model)
    print(model(X).shape)