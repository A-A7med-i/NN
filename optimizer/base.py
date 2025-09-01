from layer.layers import Layer


class Optimizer:
    """
    Base class for optimizers.

    This abstract class defines the interface for all optimization algorithms
    used to update the weights and biases of a neural network's layers.
    """

    def update_params(self, layer: Layer):
        """Updates the parameters (weights and biases) of a given layer.

        This method must be implemented by all subclasses to define how
        the gradients are used to modify a layer's parameters.

        Args:
            layer: An instance of a `Layer` (or a subclass) whose parameters
                    need to be updated.
        """
        raise NotImplementedError
