# NN: A Simple Neural Network Library

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSFNDr9Kz6GrjIzDrDp0uRhzZyDM75OVspxh8EwxoEaY21yx31Nt_dsZkRyH4aFyu2szRA&usqp=CAU" alt="NN Logo" width="1200"/>

## Overview

NN is a lightweight neural network library implemented in Python. It provides the essential building blocks for constructing, training, and evaluating deep neural networks from scratch. The project is modular, making it easy to extend or modify components such as layers, activation functions, loss functions, and optimizers.

## Project Structure

```
NN/
│
├── main.py                # Entry point for running experiments or training models
├── requirements.txt       # Python dependencies
├── setup.py               # Setup script for packaging
├── LICENSE
├── README.md
│
├── activations/           # Activation functions
│   ├── __init__.py
│   ├── activations.py     # Implementations of activation functions (ReLU, Sigmoid, etc.)
│   └── base.py            # Base class/interface for activations
│
├── layer/                 # Neural network layers
│   ├── __init__.py
│   ├── base.py            # Base class/interface for layers
│   └── layers.py          # Implementations of layers (Dense, etc.)
│
├── loss/                  # Loss functions
│   ├── __init__.py
│   ├── base.py            # Base class/interface for loss functions
│   └── loss.py            # Implementations of loss functions (MSE, CrossEntropy, etc.)
│
├── model/                 # Model definitions
│   ├── __init__.py
│   └── dnn.py             # Deep Neural Network class (model architecture, forward/backward pass)
│
└── optimizer/             # Optimizers
    ├── __init__.py
    ├── base.py            # Base class/interface for optimizers
    └── optimizer.py       # Implementations of optimizers (SGD, Adam, etc.)
```

## Key Components

- **Activations**: Contains various activation functions and a base class for extensibility.
- **Layer**: Implements different types of neural network layers, with a base class for custom layers.
- **Loss**: Provides loss functions for training, with a base class for custom loss definitions.
- **Model**: Defines the main neural network model, including forward and backward propagation logic.
- **Optimizer**: Implements optimization algorithms for training neural networks.

## Getting Started

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run an example or start training**:

   ```bash
   python main.py
   ```

3. **Customize**:
   - Add new layers, activations, losses, or optimizers by extending the respective base classes.

## Extending the Library

- To add a new activation function, implement it in `activations/activations.py` and register it in the module.
- To create a custom layer, inherit from the base class in `layer/base.py` and implement the required methods.
- For new loss functions or optimizers, follow the same pattern in their respective directories.

## License

This project is licensed under the terms of the LICENSE file.
