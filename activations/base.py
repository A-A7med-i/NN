import numpy as np


class Activation:
    """
    Base class for neural network activation functions.

    This class defines the interface for activation functions,
    requiring concrete implementations for both forward and backward passes.
    Subclasses should inherit from this and provide their specific logic.
    """

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the activation function.

        Args:
            Z (np.ndarray): The input array (usually the weighted sum
                            from the previous layer).

        Returns:
            np.ndarray: The activated output array.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    def backward(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass (gradient) of the activation function.

        This method calculates the derivative of the activation function
        with respect to its input Z. This is crucial for backpropagation.

        Args:
            Z (np.ndarray): The input array to the activation function
                            during the forward pass.

        Returns:
            np.ndarray: The gradient of the activation function with respect to Z.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
