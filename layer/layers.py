import numpy as np
from .base import Layer
from typing import Optional
from activations.activations import Activation


class Dense(Layer):
    """
    A fully connected neural network layer.

    This layer performs a linear transformation on the input data, followed by an
    element-wise activation function. It learns a set of weights and biases
    to map inputs to outputs.
    """

    def __init__(self, input_size: int, output_size: int, activation: Activation):
        """
        Initializes the dense layer.

        Args:
            input_size: The number of neurons in the previous layer, or the
                        dimension of the input features.
            output_size: The number of neurons in this layer.
            activation: An instance of an activation function class to apply
                        to the layer's output.
        """
        self.weights: np.ndarray = np.random.randn(input_size, output_size) * 0.01
        self.bias: np.ndarray = np.zeros((1, output_size))
        self.activation: Activation = activation

        self.last_input: Optional[np.ndarray] = None
        self.last_z: Optional[np.ndarray] = None
        self.weight_grad: Optional[np.ndarray] = None
        self.bias_grad: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass for the dense layer.

        The forward pass computes the dot product of the input `X` with the
        layer's `weights`, adds the `bias`, and then applies the activation function.

        Args:
            X: The input data, a NumPy array of shape (batch_size, input_size).

        Returns:
            The output of the layer after applying the activation function,
            a NumPy array of shape (batch_size, output_size).
        """
        self.last_input = X

        self.last_z = np.dot(self.last_input, self.weights) + self.bias

        return self.activation.forward(self.last_z)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass for the dense layer.

        This method calculates the gradients for the weights and biases of this layer
        and propagates the gradient backwards to the previous layer.

        Args:
            grad: The gradient of the loss with respect to the output of this layer,
                    a NumPy array of shape (batch_size, output_size).

        Returns:
            The gradient of the loss with respect to the input of this layer,
            a NumPy array of shape (batch_size, input_size), which is passed to
            the previous layer in the network.
        """
        activation_grad = self.activation.backward(self.last_z)
        grad_z = grad * activation_grad

        self.weight_grad = np.dot(self.last_input.T, grad_z)
        self.bias_grad = np.sum(grad_z, axis=0, keepdims=True)

        return np.dot(grad_z, self.weights.T)

    def get_params(self) -> int:
        """
        Returns the number of trainable parameters in the dense layer.

        The number of parameters is the sum of the number of weights and biases.
        """
        return self.weights.size + self.bias.size
