# AI Learning — Neural Network from Scratch

Learning project implementing a neural network **without any ML framework** (no TensorFlow, no PyTorch).  
Built following the [Neural Networks from Scratch](https://nnfs.io/) book to understand what happens under the hood.

## What's implemented

### Layers
- **Dense layer** — forward pass (dot product + bias) and backward pass (chain rule for gradients)

### Activation functions
- **ReLU** — `max(0, x)` with backpropagation
- **Softmax** — normalized exponentials for multi-class output
- **Softmax + Categorical Cross-Entropy** — fused for numerical stability

### Loss functions
- **Loss base class** — mean of sample losses
- **Categorical Cross-Entropy** — standard multi-class classification loss

### Optimizers
- **SGD** — Stochastic Gradient Descent with optional momentum and learning rate decay
- **Adagrad** — adapts learning rate per parameter
- **RMSProp** — exponential moving average of squared gradients
- **Adam** — momentum + adaptive learning rate (most commonly used in practice)

## How it works

```
Input → Dense → ReLU → Dense → Softmax → Cross-Entropy Loss
                                              ↓
                    Backpropagation ← gradient flows back
                                              ↓
                              Optimizer updates weights
```

## Run it

```bash
pip install numpy nnfs
python main.py
```

Trains on spiral dataset (3 classes, 100 samples each) and prints accuracy + loss every 100 epochs.

## Goal

Understand the internals of neural networks before using high-level frameworks.  
Every forward pass, every gradient, every weight update — written by hand.
