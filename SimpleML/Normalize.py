import numpy as np

"""
Gamma (gamma): It's a scale parameter that allows the model to
learn the optimal scaling of the normalized inputs. This means
it controls the amplification or reduction of the normalized values.
In other words, it enables the model to decide whether to scale up or
scale down the normalized values.

Bias (bias): It's an offset parameter that allows the model to
learn the optimal shift of the normalized inputs. This means it
enables the model to adjust the mean of the normalized values.
"""


class BatchNorm1D:
    """
    Batch Normalization Layer for 1-dimensional input data.

    Attributes:
        gamma (numpy.ndarray): Scale parameter controlling the amplification or reduction of normalized inputs.
        bias (numpy.ndarray): Offset parameter allowing the adjustment of the mean of normalized inputs.
        running_mean_x (numpy.ndarray): Running average of mean during training.
        running_var_x (numpy.ndarray): Running average of variance during training.
        var_x (numpy.ndarray): Variance of the input data.
        stddev_x (numpy.ndarray): Standard deviation of the input data.
        x_minus_mean (numpy.ndarray): Difference between input data and mean.
        standard_x (numpy.ndarray): Standardized input data.
        num_examples (int): Number of examples in the input data.
        mean_x (numpy.ndarray): Mean of the input data.
        running_avg_gamma (float): Running average factor for mean and variance.
        epsilon (float): Small constant to avoid division by zero.
        gamma_grad (numpy.ndarray): Gradient of the loss with respect to gamma.
        bias_grad (numpy.ndarray): Gradient of the loss with respect to bias.
        output (numpy.ndarray): Output of the layer after forward pass.
        dinputs (numpy.ndarray): Gradient of the loss with respect to the input data.
    """

    def __init__(self, dims: int) -> None:
        """Initializes BatchNorm1D with scale and offset parameters."""
        self.gamma = np.ones((1, dims), dtype="float32")
        self.bias = np.zeros((1, dims), dtype="float32")

        self.running_mean_x = np.zeros(0)
        self.running_var_x = np.zeros(0)

        # forward params
        self.var_x = np.zeros(0)
        self.stddev_x = np.zeros(0)
        self.x_minus_mean = np.zeros(0)
        self.standard_x = np.zeros(0)
        self.num_examples = 0
        self.mean_x = np.zeros(0)
        self.running_avg_gamma = 0.9
        self.epsilon = 10**-5

        # backward params
        self.gamma_grad = np.zeros(0)
        self.bias_grad = np.zeros(0)

    def update_running_variables(self) -> None:
        """Updates running averages of mean and variance."""
        is_mean_empty = np.array_equal(np.zeros(0), self.running_mean_x)
        is_var_empty = np.array_equal(np.zeros(0), self.running_var_x)
        if is_mean_empty != is_var_empty:
            raise ValueError(
                "Mean and Var running averages should be "
                "initilizaded at the same time"
            )
        if is_mean_empty:
            self.running_mean_x = self.mean_x
            self.running_var_x = self.var_x
        else:
            momentum = self.running_avg_gamma
            self.running_mean_x = (
                momentum * self.running_mean_x + (1.0 - momentum) * self.mean_x
            )
            self.running_var_x = (
                momentum * self.running_var_x + (1.0 - momentum) * self.var_x
            )

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward function of the BatchNormalization layer. It computes the output of
        the layer, the formula is :
            output = scale * input_norm + bias
        Where input_norm is:
            input_norm = (input - mean) / sqrt(var + epsil)
        where mean and var are the mean and the variance of the input batch of
        images computed over the first axis (batch)
        Parameters:
            inputs  : numpy array, batch of input images in the format (batch, w, h, c)
            self.epsilon : float, used to avoid division by zero when computing 1. / var
        """
        self.num_examples = inputs.shape[0]
        if training:
            self.mean_x = np.mean(inputs, axis=0, keepdims=True)
            self.var_x = np.mean((inputs - self.mean_x) ** 2, axis=0, keepdims=True)
            self.update_running_variables()
        else:
            self.mean_x = self.running_mean_x.copy()
            self.var_x = self.running_var_x.copy()

        self.var_x += self.epsilon
        self.stddev_x = np.sqrt(self.var_x)
        self.x_minus_mean = inputs - self.mean_x
        self.standard_x = self.x_minus_mean / self.stddev_x
        self.output = self.gamma * self.standard_x + self.bias

    def backward(self, dvalues: np.ndarray) -> np.ndarray:
        """
        Performs backward pass of BatchNorm1D.

        Args:
            dvalues (numpy.ndarray): Gradient of the loss with respect to the output.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input data.
        """
        standard_grad = dvalues * self.gamma

        var_grad = np.sum(
            standard_grad * self.x_minus_mean * -0.5 * self.var_x ** (-3 / 2),
            axis=0,
            keepdims=True,
        )
        stddev_inv = 1 / self.stddev_x
        aux_x_minus_mean = 2 * self.x_minus_mean / self.num_examples

        mean_grad = np.sum(
            standard_grad * -stddev_inv, axis=0, keepdims=True
        ) + var_grad * np.sum(-aux_x_minus_mean, axis=0, keepdims=True)

        self.gamma_grad = np.sum(dvalues * self.standard_x, axis=0, keepdims=True)
        self.bias_grad = np.sum(dvalues, axis=0, keepdims=True)

        self.apply_gradients()

        self.dinputs = (
            standard_grad * stddev_inv
            + var_grad * aux_x_minus_mean
            + mean_grad / self.num_examples
        )

    def apply_gradients(self, learning_rate: float) -> None:
        """
        Applies gradients to update scale and offset parameters.

        Args:
            learning_rate (float): Learning rate for gradient descent.
        """
        self.gamma -= learning_rate * self.gamma_grad
        self.bias -= learning_rate * self.bias_grad
