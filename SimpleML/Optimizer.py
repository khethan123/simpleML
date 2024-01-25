import numpy as np


# This is a normal GD and not a pure SGD
class OptimizerSGD:

    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        '''
        Initialize SGD Optimizer.

        Parameters:
            learning_rate (float): The learning rate.
            decay (float): Learning rate decay.
            momentum (float): Momentum factor.        
        '''
        self.current_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.iterations = 0  # the step at which we want to update LR

    def preUpdateParams(self):
        '''
        Perform pre-update operations.
        '''
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))


    def updateParams(self, layer):
        '''
        Update parameters based on SGD.

        Parameters:
            layer: The neural network layer
        '''
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            # Vanilla SGD updates ( = - lr * parameters)
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def postUpdateParams(self):
        self.iterations += 1


# Adagrad optimizer
class OptimizerAdagrad:

    def __init__(self, learning_rate=1.0, decay=0.0, epsilon=1e-7):
        '''
        Initialize Adagrad optimizer.

        Parameters:
            learning_rate (float): The learning rate.
            decay (float): Learning rate decay.
            epsilon (float): Small constant to avoid division by zero.
        '''
        self.current_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.decay = decay
        self.iterations = 0

    def preUpdateParams(self):
        '''
        Perform pre-update operations.
        '''
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay * self.iterations))

    def updateParams(self, layer):
        '''
        Update parameters based on Adagrad.

        Parameters:
            layer: The neural network layer.       
        '''
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def postUpdateParams(self):
        '''
        Perform post-update operations.
        '''
        self.iterations += 1


# RMSprop optimizer
class OptimizerRMSprop:

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
    rho=0.9):
        '''
        Initialize RMSprop optimizer.

        Parameters:
            learning_rate (float): The learning rate.
            decay (float): Learning rate decay.
            epsilon (float): Small constant to avoid division by zero.
            rho (float): Decay factor for squared gradients.
        '''
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def preUpdateParams(self):
        '''
        Perform pre-update operations.
        '''
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def updateParams(self, layer):
        '''
        Update parameters based on RMSprop.

        Parameters:
            layer: The neural network layer.
        '''
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def postUpdateParams(self):
        '''
        Perform post-update operations.
        '''
        self.iterations += 1


# Adam Optimizer
class OptimizerAdam:

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7,
                beta_1=0.9, beta_2=0.999):
        '''
        Initialize Adam optimizer.

        Parameters:
            learning_rate (float): The learning rate.
            decay (float): Learning rate decay.
            epsilon (float): Small constant to avoid division by zero.
            beta_1 (float): Exponential decay rate for the first moment estimates.
            beta_2 (float): Exponential decay rate for the second moment estimates.
        '''
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def preUpdateParams(self):
        '''
        Perform pre-update operations.
        '''
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1.0 / (1.0 + self.decay*self.iterations))

    def updateParams(self, layer):
        '''
        Update parameters based on Adam.

        Parameters:
            layer: The neural network layer.
        '''
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) +self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) +self.epsilon)

    def postUpdateParams(self):
        '''
        Perfrom post-update operations.
        '''
        self.iterations += 1