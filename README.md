# Neural Arithmetic Logic Units

[WIP]

This is a PyTorch implementation of [Neural Arithmetic Logic Units](https://arxiv.org/abs/1808.00508) by *Andrew Trask, Felix Hill, Scott Reed, Jack Rae, Chris Dyer and Phil Blunsom*.

<p align="center">
 <img src="./imgs/arch.png" alt="Drawing", width=60%>
</p>

## API

```python

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
This should generate a text file called `interpolation.txt` with the following results. (Currently only supports interpolation, I'm working on the rest. Also getting `nans` which I'm investigating.) 

|       | Relu6    | None     | NAC      | NALU   |
|-------|----------|----------|----------|--------|
| a + b | 0.002    | 0.000    | 0.000    | 1.399  |
| a - b | 0.046    | 0.000    | 0.000    | 0.224  |
| a * b | 83.012   | 99.590   | 98.822   | 12.237 |
| a / b | 2245.560 | 2888.195 | 2765.908 | nan    |
| a ^ 2 | 76.126   | 99.106   | 99.559   | nan    |
