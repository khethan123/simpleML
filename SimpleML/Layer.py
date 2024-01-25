import numpy as np

class LayerDense:
    '''
    Creates a dense layer and randomly initializes
    weights and biases.

    args:
        n_inputs: batch of inputs, size
        n_neurons: numbers of neurons in hidden layers, size

    returns:
        outputs the value after performing calculations
    '''

    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        ''' 
        Backward pass which is used to calculate the gradients
        (partial derivatives) & derivatives from the last layer
        towards the first layer.
        '''

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights & biases
        if self.weight_regularizer_l1 > 0: 
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 on weights & biases
        if self.weight_regularizer_l2 > 0: 
            self.dweights += 2*self.weight_regularizer_l2*self.weights

        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2*self.bias_regularizer_l2*self.biases

        # Gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    # Retrieve layer parameters
    def get_parameters(self):
        return self.weights, self.biases

    # Set weights and biases in a layer instance/object
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases


# Dropout
class LayerDropout:

    def __init__(self, rate):
        # Store rate, we invert it. for ex: dropout of 0.1
        # we need success rate of 0.9
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return

        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs*self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask


# Input "layer"
class LayerInput:

    def forward(self, inputs, training):
        self.output = inputs