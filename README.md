# AiLearning — Neural Network from Scratch

> Full neural network engine built **without any ML framework** — no TensorFlow, no PyTorch, no sklearn.  
> Every forward pass, every gradient, every weight update — written by hand in Python & NumPy.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-only-013243?style=flat&logo=numpy&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-from%20scratch-FF6B6B?style=flat)
![Status](https://img.shields.io/badge/status-active-brightgreen?style=flat)

Based on [Neural Networks from Scratch](https://nnfs.io/) by Harrison Kinsley & Daniel Kukieła.

## Why this project

Using PyTorch or TensorFlow is easy. Understanding *why* they work is harder.  
This project forces a deep understanding of:
- What a neuron actually computes (dot product + bias)
- How gradients flow backward through a network (chain rule)
- Why optimizers like Adam converge faster than SGD
- What "loss" really measures and how to minimize it

## What's implemented

### Layers
```python
class Layer_Dense:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, d_out):
        self.dweights = np.dot(self.inputs.T, d_out)   # gradient w.r.t weights
        self.dbiases  = np.sum(d_out, axis=0, keepdims=True)
        self.dinputs  = np.dot(d_out, self.weights.T)   # gradient w.r.t inputs
```

### Activation functions
| Function | Forward | Backward |
|---|---|---|
| **ReLU** | `max(0, x)` | `1 if x > 0 else 0` |
| **Softmax** | `exp(x) / sum(exp(x))` | Jacobian matrix |
| **Softmax + CCE** | fused for numerical stability | simplified gradient |

### Loss functions
- **Categorical Cross-Entropy** — standard multi-class classification loss
- **Loss base class** — mean of sample losses + regularization hooks

### Optimizers
| Optimizer | Key idea |
|---|---|
| **SGD** | Fixed learning rate, optional momentum + LR decay |
| **Adagrad** | Per-parameter adaptive learning rate (accumulates squared gradients) |
| **RMSProp** | Exponential moving average of squared gradients (fixes Adagrad decay) |
| **Adam** | Momentum + RMSProp — most commonly used in practice |

## How backpropagation works (in this codebase)

```
Forward pass:
  Input → Dense → ReLU → Dense → Softmax → Loss

Backward pass (chain rule):
  dLoss/dSoftmax → dSoftmax/dDense2 → dDense2/dReLU → dReLU/dDense1 → dDense1/dInput
                        ↓                   ↓
                   dWeights2           dWeights1
                        ↓                   ↓
                   Optimizer.update()  Optimizer.update()
```

Each layer stores its inputs during `forward()` so it can compute gradients during `backward()`.

## Run it

```bash
git clone https://github.com/GeoffreyFRANZ/AiLearning
cd AiLearning
pip install numpy nnfs
python main.py
```

Trains on the **spiral dataset** (3 classes, 100 samples each) — a non-linearly separable problem that requires hidden layers to solve.

Expected output:
```
epoch 0   loss: 1.099  acc: 0.340
epoch 100 loss: 0.721  acc: 0.673
...
epoch 10000 loss: 0.089  acc: 0.967
```

## Goal

Understand neural network internals well enough to implement them in any language.  
This is the foundation before working with PyTorch, JAX, or custom CUDA kernels.

## Related

- [photoshoplike](https://github.com/GeoffreyFRANZ/photoshoplike) — Low-level GPU image processing in Go + C + OpenCL
