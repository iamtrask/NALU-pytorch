import math
import torch
import torch.nn as nn

from .nac import NAC
from .nalu import NALU


class MultiLayerNet(nn.Module):
    def __init__(self, activation, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        if activation is 'none':
            self.activation = None
        elif activation is 'hardtanh':
            self.activation = nn.Hardtanh()
        elif activation is 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation is 'relu6':
            self.activation = nn.ReLU6()
        elif activation is 'tanh':
            self.activation = nn.Tanh()
        elif activation is 'tanhshrink':
            self.activation = nn.Tanhshrink()
        elif activation is 'hardshrink':
            self.activation = nn.Hardshrink()
        elif activation is 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activation is 'softshrink':
            self.activation = nn.Softshrink()
        elif activation is 'softsign':
            self.activation = nn.Softsign()
        elif activation is 'relu':
            self.activation = nn.ReLU()
        elif activation is 'prelu':
            self.activation = nn.PReLU()
        elif activation is 'softplus':
            self.activation = nn.Softplus()
        elif activation is 'elu':
            self.activation = nn.ELU()
        elif activation is 'selu':
            self.activation = nn.SELU()
        else:
            raise ValueError("[!] Invalid activation function.")


        layers = []
        if self.activation is not None:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                self.activation,
            ])
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
        for i in range(num_layers - 2):
            if self.activation is not None:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    self.activation,
                ])
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.model = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        out = self.model(x)
        return out


class MultiLayerNAC(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        layers = []
        layers.append(NAC(in_dim, hidden_dim))
        for i in range(num_layers - 2):
            layers.append(NAC(hidden_dim, hidden_dim))
        layers.append(NAC(hidden_dim, out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out


class MultiLayerNALU(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        layers = []
        layers.append(NALU(in_dim, hidden_dim))
        for i in range(num_layers - 2):
            layers.append(NALU(hidden_dim, hidden_dim))
        layers.append(NALU(hidden_dim, out_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out
