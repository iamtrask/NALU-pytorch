import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn.parameter import Parameter


class NAC(nn.Module):
    """A Neural Accumulator [1].

    NAC supports the ability to accumulate quantities
    additively which is a desirable inductive bias for
    linear extrapolation.

    Attributes:
        in_features: size of the input sample.
        out_features: size of the output sample.

    Sources:
        [1]: https://arxiv.org/abs/1808.00508
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W_hat = Parameter(torch.Tensor(out_features, in_features))
        self.M_hat = Parameter(torch.Tensor(out_features, in_features))
        self.W = F.tanh(self.W_hat) * F.sigmoid(self.M_hat)

        init.kaiming_uniform_(self.W_hat, a=math.sqrt(5))
        init.kaiming_uniform_(self.M_hat, a=math.sqrt(5))

    def forward(self, input):
        return F.linear(input, self.W, None)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(self.in_features, self.out_features)
