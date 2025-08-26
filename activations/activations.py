import numpy as np
from .base import Activation


class Sigmoid(Activation):
    """
    Implements the Sigmoid activation function and its derivative.

    The Sigmoid function squashes input values to a range between 0 and 1,
    making it suitable for binary classification outputs or probabilities.
    """

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the Sigmoid activation.

        The input `Z` is clipped to prevent numerical overflow during the
        exponentiation of very large or small numbers, which could lead to
        NaNs or Infs.

        Args:
            Z (np.ndarray): The input array (e.g., weighted sum from a layer).

        Returns:
            np.ndarray: The activated output array, where each element is
                        between 0 and 1.
        """
        Z = np.clip(Z, -500, 500)
        return 1 / (1 + np.exp(-Z))

    def backward(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass (gradient) of the Sigmoid function.

        The derivative of Sigmoid is s * (1 - s), where 's' is the output
        of the forward pass of Sigmoid.

        Args:
            Z (np.ndarray): The input array to the Sigmoid function during
                            the forward pass.

        Returns:
            np.ndarray: The gradient of the Sigmoid function with respect to Z.
        """
        s = self.forward(Z)
        return s * (1 - s)


class Relu(Activation):
    """
    Implements the Rectified Linear Unit (ReLU) activation function and its derivative.

    ReLU outputs the input directly if it's positive, otherwise, it outputs zero.
    It's widely used for its computational efficiency and ability to mitigate
    the vanishing gradient problem.
    """

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the ReLU activation.

        Args:
            Z (np.ndarray): The input array.

        Returns:
            np.ndarray: The activated output array, where negative values are
                        replaced with zero.
        """
        return np.maximum(0, Z)

    def backward(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass (gradient) of the ReLU function.

        The derivative of ReLU is 1 for positive inputs and 0 for negative inputs.
        For Z=0, the derivative is typically considered 0 (or undefined, but 0 is
        common in practice).

        Args:
            Z (np.ndarray): The input array to the ReLU function during the
                            forward pass.

        Returns:
            np.ndarray: The gradient of the ReLU function with respect to Z,
                        containing 1s where Z > 0 and 0s otherwise.
        """
        return (Z > 0).astype(float)


class Tanh(Activation):
    """
    Implements the Hyperbolic Tangent (Tanh) activation function and its derivative.

    Tanh squashes input values to a range between -1 and 1. It is similar to Sigmoid
    but is zero-centered, which can sometimes aid in training.
    """

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the Tanh activation.

        Args:
            Z (np.ndarray): The input array.

        Returns:
            np.ndarray: The activated output array, where each element is
                        between -1 and 1.
        """
        return np.tanh(Z)

    def backward(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass (gradient) of the Tanh function.

        The derivative of Tanh is 1 - (tanh(Z))^2, or 1 - (forward(Z))^2.

        Args:
            Z (np.ndarray): The input array to the Tanh function during the
                            forward pass.

        Returns:
            np.ndarray: The gradient of the Tanh function with respect to Z.
        """
        return 1 - np.square(self.forward(Z))


class Softmax(Activation):
    """
    Implements the Softmax activation function.

    Softmax is typically used in the output layer of a neural network for
    multi-class classification problems. It converts a vector of arbitrary
    real values into a probability distribution, where the sum of probabilities
    for each sample is 1.

    Note: The backward pass for Softmax is often computed in conjunction with
    the loss function (e.g., Cross-Entropy Loss) for numerical stability.
    The provided `backward` method here is a placeholder for a generic
    derivative if treated in isolation, which is less common in practice.
    """

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass of the Softmax activation.

        It applies numerical stability trick by subtracting the maximum value
        from Z before exponentiation to prevent overflow with large inputs.

        Args:
            Z (np.ndarray): The input array (logits). Assumes Z is 2D,
                            where rows are samples and columns are classes.

        Returns:
            np.ndarray: The array of probabilities, where each row sums to 1.
        """
        Z_stable = Z - np.max(Z, axis=1, keepdims=True)
        Z_EXP = np.exp(Z_stable)
        return Z_EXP / np.sum(Z_EXP, axis=1, keepdims=True)

    def backward(self, Z: np.ndarray) -> np.ndarray:
        """
        Computes a simplified backward pass (gradient) of the Softmax function.

        WARNING: In typical neural network training, the backward pass for Softmax
        is *not* computed in isolation like this. It's usually combined with
        the cross-entropy loss function to get a simpler and numerically more
        stable gradient (output - target).

        This implementation returns 1 for demonstration, but it's not the
        mathematically correct Jacobian for Softmax in a general context.
        A proper backward for Softmax requires considering the loss function.

        Args:
            Z (np.ndarray): The input array to the Softmax function during the
                            forward pass.

        Returns:
            np.ndarray: A placeholder value.
        """
        return 1
