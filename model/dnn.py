import numpy as np
from loss.loss import Loss
from layer.layers import Layer
import matplotlib.pyplot as plt
from typing import List, Optional
from optimizer.base import Optimizer


class DNN:
    """
    Deep Neural Network model.

    This class provides a framework for building, compiling, training,
    and making predictions with a deep neural network. It supports adding
    custom layers, specifying a loss function, and choosing an optimizer.
    """

    def __init__(self):
        """
        Initializes the Deep Neural Network.

        Sets up an empty list for layers, and initializes loss, optimizer,
        and loss history as None or empty.
        """
        self.layers: List[Layer] = []
        self.loss: Optional[Loss] = None
        self.optimizer: Optional[Optimizer] = None
        self.loss_history: List[float] = []

    def add(self, layer: Layer):
        """
        Adds a layer to the network.

        Layers are added in the order they will be processed during the
        forward pass.

        Args:
            layer: An instance of a `Layer` (or a subclass) to be added
                    to the neural network.
        """
        self.layers.append(layer)

    def compile(self, loss: Loss, optimizer: Optimizer):
        """
        Configures the model for training.

        This method sets the loss function and optimizer to be used
        during the training process.

        Args:
            loss: An instance of a `Loss` (or a subclass) to calculate
                    the training error.
            optimizer: An instance of an `Optimizer` (or a subclass) to
                    update the model's weights during training.
        """
        self.loss = loss
        self.optimizer = optimizer

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass through the network to make predictions.

        Args:
            X: The input data for which to make predictions, a NumPy array.
                Expected shape is (batch_size, input_dim).

        Returns:
            The output of the final layer in the network, representing the
            model's predictions, as a NumPy array.
        """
        output: np.ndarray = X

        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad: np.ndarray):
        """
        Performs a backward pass through the network to compute gradients.

        The gradients are propagated from the output layer back through
        each preceding layer, updating the `weight_grad` and `bias_grad`
        attributes within each layer.

        Args:
            grad: The initial gradient from the loss function, a NumPy array.
                    Its shape should match the output of the final layer.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_weights(self):
        """
        Updates weights for all layers using the optimizer.

        This method iterates through all layers in the network and calls
        the optimizer's `update_param` method for each layer, applying
        the calculated gradients to adjust weights and biases.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer not compiled. Call .compile() first.")
        for layer in self.layers:
            self.optimizer.update_params(layer)

    def accuracy(self, y_pred, y_true):
        return (np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)).mean() * 100

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        verbose: bool = True,
    ):
        """
        Trains the neural network using the provided data.

        The training process involves iterating through epochs, shuffling data,
        processing in batches, performing forward and backward passes,
        calculating loss, and updating weights.

        Args:
            x: The input training data, a NumPy array of shape (num_samples, features).
            y: The true labels for the training data, a NumPy array of shape
                (num_samples, output_dim) for one-hot encoded or (num_samples, 1) for regression.
            epochs: The number of full passes through the entire training dataset.
            batch_size: The number of samples per gradient update.
            verbose: If True, prints the loss at regular intervals during training.
                    Defaults to True.

        Raises:
            RuntimeError: If the model has not been compiled with a loss function
                        and an optimizer.
        """
        if self.loss is None or self.optimizer is None:
            raise RuntimeError(
                "Model not compiled. Call .compile() first with a loss and an optimizer."
            )

        num_samples: int = x.shape[0]

        for epoch in range(epochs):
            epoch_loss: float = 0
            num_batches: int = 0

            permutation: np.ndarray = np.random.permutation(num_samples)
            x_shuffled: np.ndarray = x[permutation]
            y_shuffled: np.ndarray = y[permutation]

            for i in range(0, num_samples, batch_size):
                num_batches += 1

                x_batch: np.ndarray = x_shuffled[i : i + batch_size]
                y_batch: np.ndarray = y_shuffled[i : i + batch_size]

                # Forward pass
                y_pred: np.ndarray = self.predict(x_batch)

                # Calculate loss and gradient
                loss_val: float = self.loss.calculate(y_pred, y_batch)
                grad: np.ndarray = self.loss.derivative(y_pred, y_batch)

                epoch_loss += loss_val

                # Backward pass
                self.backward(grad)

                # Update weights
                self.update_weights()

            avg_loss: float = epoch_loss / num_batches
            self.loss_history.append(avg_loss)

            train_accuracy = self.accuracy(y_pred, y_batch)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"\nEpoch: {epoch + 1}")
                print(f"Train loss: {avg_loss:.6f} | Train accuracy: {train_accuracy}%")

        return avg_loss

    def summary(self) -> None:
        """Prints a simple summary of the model architecture."""
        print("\nModel Summary")
        print("=" * 60)

        print(f"{'Layer':<15}{'Type':<20}{'Parameters':>15}")
        print("-" * 60)

        total_params = 0

        for i, layer in enumerate(self.layers):
            params = layer.get_params()
            total_params += params
            layer_type = layer.__class__.__name__

            print(f"{i+1:<15}{layer_type:<20}{params:>15,}")

        print("=" * 60)
        print(f"{'Total Trainable Parameters:':<35}{total_params:>15,}")
        print("=" * 60)

    def plot_history(self, history: list) -> None:
        """
        Plots the training loss history.

        Args:
            history (list): A list of loss values for each epoch.
        """
        plt.plot(history, label="Training Loss")
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
