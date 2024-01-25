import random
from micrograd.engine import Value

class Module:
    """
    Base class for all neural network modules.
    """

    def zero_grad(self):
        """
        Sets the gradients of all parameters to 0.
        """
        for param in self.parameters():
            param.grad = 0

    def parameters(self):
        """
        Returns a list of all parameters (weights and biases) in the module.
        """
        return []


class Neuron(Module):
    """
    A single neuron with configurable activation function.
    """

    def __init__(self, nin, nonlin='linear'):
        """
        Initializes a new Neuron instance.

        :param nin: Number of input neurons.
        :param nonlin: Activation function ('linear', 'relu', or 'tanh').
        """
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.nonlin = nonlin

    def __call__(self, x):
        """
        Computes the output of the neuron given an input vector.

        :param x: Input vector.
        :return: Output value.
        """
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.nonlin == 'linear':
            out = act
        elif self.nonlin == 'relu':
            out = act.relu()
        elif self.nonlin == 'tanh':
            out = act.tanh()
        else:
            raise ValueError(f"Invalid nonlin value '{self.nonlin}'")
        return out

    def parameters(self):
        """
        Returns a list of all parameters (weights and biases) in the neuron.
        """
        return self.w + [self.b]

    def __repr__(self):
        """
        Returns a string representation of the neuron.
        """
        if self.nonlin == 'linear':
            return f"LinearNeuron({len(self.w)})"
        else:
            return f"{self.nonlin.capitalize()}Neuron({len(self.w)})"


class Layer(Module):
    """
    A layer of neurons.
    """

    def __init__(self, nin, nout, **kwargs):
        """
        Initializes a new Layer instance.

        :param nin: Number of input neurons.
        :param nout: Number of output neurons.
        :param kwargs: Keyword arguments for Neuron constructor.
        """
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        """
        Computes the output of the layer given an input vector.

        :param x: Input vector.
        :return: Output vector.
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        """
        Returns a list of all parameters (weights and biases) in the layer.
        """
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        """
        Returns a string representation of the layer.
        """
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    """
    A multi-layer perceptron (MLP) with configurable activation function.
    """

    def __init__(self, nin, nouts, nonlin):
        """
        Initializes a new MLP instance.

        :param nin: Number of input neurons.
        :param nouts: List of output neurons for each layer.
        :param nonlin: Activation function ('linear', 'relu', or 'tanh').
        """
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=nonlin) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Computes the output of the MLP given an input vector.

        :param x: Input vector.
        :return: Output vector.
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        """
        Returns a list of all parameters (weights and biases) in the MLP.
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        """
        Returns a string representation of the MLP.
        """
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"