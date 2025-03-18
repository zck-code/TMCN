import torch.nn as nn


class DDC(nn.Module):
    def __init__(self, input_dim, n_hidden, n_clusters):
        """
        DDC clustering module

        :param input_dim: Shape of inputs.
        :param cfg: DDC config. See `config.defaults.DDC`
        """
        super().__init__()

        hidden_layers = [nn.Linear(input_dim, n_hidden), nn.ReLU(), nn.BatchNorm1d(num_features=n_hidden)]
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(nn.Linear(n_hidden, n_clusters), nn.Softmax(dim=1))

    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return output, hidden
