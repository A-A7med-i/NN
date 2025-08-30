import numpy as np


class Loss:
    """
    Base class for loss functions.

    This class provides a standard interface for all loss functions
    to ensure they can be used interchangeably within a neural network
    training framework.
    """

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates the loss value between predicted and true labels.

        Args:
            y_pred: The predicted values from the model, as a NumPy array.
            y_true: The ground truth labels, as a NumPy array.

        Returns:
            The scalar loss value, as a float.
        """
        raise NotImplementedError

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the loss with respect to the predicted values.

        This derivative is essential for backpropagation to compute the gradients
        needed to update the model's weights.

        Args:
            y_pred: The predicted values from the model, as a NumPy array.
            y_true: The ground truth labels, as a NumPy array.

        Returns:
            The gradient of the loss with respect to `y_pred`, as a NumPy array.
            The shape of this array should match the shape of `y_pred`.
        """
        raise NotImplementedError
