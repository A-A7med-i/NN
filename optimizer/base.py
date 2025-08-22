from layer.layers import Layer


class Optimizer:
    """Base class for optimizers.

    This abstract class defines the interface for all optimization algorithms
    used to update the weights and biases of a neural network's layers.
    """

    def __init__(self, learning_rate: float = 0.01):
        """Initializes the optimizer with a learning rate.

        Args:
            learning_rate: A float that determines the step size at each
                            iteration while moving toward a minimum of the loss
                            function. Defaults to 0.01.
        """
        self.learning_rate = learning_rate

    def update_params(self, layer: Layer):
        """Updates the parameters (weights and biases) of a given layer.

        This method must be implemented by all subclasses to define how
        the gradients are used to modify a layer's parameters.

        Args:
            layer: An instance of a `Layer` (or a subclass) whose parameters
                   need to be updated.
        """
        raise NotImplementedError
