import numpy as np

class ActivationReLU:
    '''
    Outputs max of output values from inputs to this layer
    '''

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        '''
        Since we need to modify the original variable,
        let's make a copy of the values first and put
        gradient as `zero` where input values are negative
        '''
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs
    

class ActivationSoftmax:
    '''
    Outputs normalized values (probabilities) between
    the range 0 and 1
    '''

    def forward(self, inputs, training):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    # perform the backward pass
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix,
            single_dvalues)

    # Calc predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
    

class ActivationLinear:
    '''
    Outputs linear values for given inputs, y = x.
    '''
    
    def forward(self, inputs, training):
        # Just remember values
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        # derivative is 1, 1 * dvalues = dvalues - the chain rule
        self.dinputs = dvalues.copy()

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs
    

class ActivationSoftmaxLossCategoricalCrossentropy():
    '''
    Softmax classifier - combined Softmax activation
    and cross-entropy loss for faster backward step
    '''

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples


# Sigmoid activation
class ActivationSigmoid:
    '''
    Outputs values between zero and one
    '''

    def forward(self, inputs, training):
        # save the i/p & calc o/p of the sigmoid func
        self.inputs = inputs
        self.output = 1/ (1 + np.exp(-inputs))

    def backward(self, dvalues):
        # Derivative * o/p of previous layer - chain rule
        self.dinputs = dvalues * (1 - self.output) * self.output

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1
    

# Hyperbolic Tangent Activation
class ActivationTanh:
    '''
    Outputs values in the range of [-1, 1]
    '''

    def forward(self, inputs, training):
        # save the i/p and calc o/p of tanh func
        self.inputs = inputs
        self.output = np.tanh(inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output**2)

    def predictions(self, outputs):
        # For tanh activation, predictions can be the raw outputs
        return outputs