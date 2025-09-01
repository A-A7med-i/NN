from layer.layers import Dense
from .base import Optimizer, Layer


class GD(Optimizer):
    """
    Optimizer that implements the Gradient Descent algorithm.

    This optimizer updates the weights and biases of a layer by taking
    a step in the opposite direction of the gradient, scaled by the
    learning rate.
    """

    def __init__(self, learning_rate: int = 0.01):
        """
        Initializes the optimizer with a learning rate.

        Args:
            learning_rate: A float that determines the step size at each
                            iteration while moving toward a minimum of the loss
                            function. Defaults to 0.01.
        """
        self.learning_rate = learning_rate

    def update_params(self, layer: Layer):
        """
        Updates the weights and biases of a dense layer using Gradient Descent.

        This method checks if the layer is an instance of `Dense` and, if so,
        applies the update rule: `param = param - learning_rate * grad`.

        Args:
            layer: An instance of a `Layer` whose parameters (weights and biases)
                    are to be updated. It is expected to be a `Dense` layer
                    for this optimizer to function.
        """
        if isinstance(layer, Dense):
            layer.weights -= self.learning_rate * layer.weight_grad
            layer.bias -= self.learning_rate * layer.bias_grad
