import math
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, activation, input_dim=1, encoding_dim=8):
        super().__init__()

        if activation is 'hardtanh':
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

        self.i2h = nn.Linear(input_dim, encoding_dim)
        self.h2h1 = nn.Linear(encoding_dim, encoding_dim)
        self.h2h2 = nn.Linear(encoding_dim, encoding_dim)
        self.h2h3 = nn.Linear(encoding_dim, encoding_dim)
        self.h2o = nn.Linear(encoding_dim, input_dim)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        out = self.activation(self.i2h(x))
        out = self.activation(self.h2h1(out))
        out = self.activation(self.h2h2(out))
        out = self.activation(self.h2h3(out))
        out = self.h2o(out)
        return out
