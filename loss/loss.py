import numpy as np
from .base import Loss


class MeanSquaredError(Loss):
    """
    Mean Squared Error (MSE) loss function.

    This loss function calculates the average of the squares of the errors,
    which is the average squared difference between the predicted values and
    the true values. It is commonly used for regression tasks.
    """

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates the Mean Squared Error loss.

        Args:
            y_pred: The predicted values, a NumPy array.
            y_true: The true values, a NumPy array.

        Returns:
            The scalar MSE loss value.
        """
        return np.mean((y_pred - y_true) ** 2)

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the MSE loss with respect to y_pred.

        Args:
            y_pred: The predicted values, a NumPy array.
            y_true: The true values, a NumPy array.

        Returns:
            The gradient of the MSE loss, a NumPy array with the same shape
            as y_pred.
        """
        return 2 * (y_pred - y_true) / y_true.shape[0]


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy loss function.

    This loss function is used for multi-class classification problems. It
    measures the performance of a classification model whose output is a
    probability value between 0 and 1.
    """

    def calculate(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates the Categorical Cross-Entropy loss.

        A small value (1e-15) is clipped to prevent a log of zero, which
        would result in an undefined value.

        Args:
            y_pred: The predicted probabilities, a NumPy array of shape
                    (batch_size, num_classes).
            y_true: The one-hot encoded true labels, a NumPy array of shape
                    (batch_size, num_classes).

        Returns:
            The scalar cross-entropy loss value.
        """
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred_clipped))

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculates the derivative of the Categorical Cross-Entropy loss.

        Args:
            y_pred: The predicted probabilities, a NumPy array of shape
                    (batch_size, num_classes).
            y_true: The one-hot encoded true labels, a NumPy array of shape
                    (batch_size, num_classes).

        Returns:
            The gradient of the loss with respect to y_pred, a NumPy array with
            the same shape as y_pred.
        """
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred_clipped - y_true) / y_true.shape[0]
