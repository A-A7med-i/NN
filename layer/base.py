import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    """
    Abstract base class for neural network layers.

    This class defines the interface that all custom neural network layers
    must implement. It ensures that every layer has a `forward` method
    for computing output and a `backward` method for computing gradients.
    """

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass through the layer.

        Args:
            X: The input data for the layer, typically a NumPy array.
                The shape of X depends on the specific layer (e.g.,
                (batch_size, input_dim) for a dense layer).

        Returns:
            The output of the layer after processing the input, as a NumPy array.
            The shape of the output depends on the layer's operation.
        """
        pass

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """
        Performs the backward pass through the layer.

        This method computes the gradients of the loss with respect to
        the layer's inputs and/or its own parameters, propagating the
        gradient backwards through the network.

        Args:
            grad: The gradient of the loss with respect to the output of this layer.
                    This `grad` typically comes from the next layer in the network
                    during backpropagation. It's a NumPy array with a shape
                    matching the output of the `forward` pass.

        Returns:
            The gradient of the loss with respect to the input of this layer,
            as a NumPy array. This gradient is then passed to the previous layer.
        """
        pass

    @abstractmethod
    def get_params(self) -> int:
        """Returns the number of trainable parameters in the layer."""
        pass
