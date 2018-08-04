# Neural Arithmetic Logic Units

[WIP]

This is a PyTorch implementation of [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508) by *Andrew Trask, Felix Hill, Scott Reed, Jack Rae, Chris Dyer and Phil Blunsom*.

<p align="center">
 <img src="./imgs/arch.png" alt="Drawing", width=60%>
</p>

## API

```python
from models import *

# single layer modules
NAC(in_features, out_features)
NALU(in_features, out_features)

# stacked layers
MultiLayerNAC(num_layers, in_dim, hidden_dim, out_dim)
MultiLayerNALU(num_layers, in_dim, hidden_dim, out_dim)
```

## Experiments

To reproduce "Numerical Extrapolation Failures in Neural Networks" (Section 1.1), run:

```python
python failures.py
```

This should generate the following plot:

<p align="center">
 <img src="./imgs/extrapolation.png" alt="Drawing", width=60%>
</p>

To reproduce "Simple Function Learning Tasks" (Section 4.1), run:

```python
python function_learning.py
```
This should generate a text file called `interpolation.txt` with the following results. (Currently only supports interpolation, I'm working on the rest)

|       | Relu6    | None     | NAC      | NALU   |
|-------|----------|----------|----------|--------|
| a + b | 0.002    | 0.004    | 0.001    | 0.017  |
| a - b | 0.046    | 0.005    | 0.000    | 0.003  |
| a * b | 83.012   | 0.444    | 5.218    | 5.218  |
| a / b | 106.441  | 0.338    | 2.096    | 2.096  |
| a ^ 2 | 94.103   | 0.630    | 3.871    | 0.196  |
