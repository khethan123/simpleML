import numpy as np

class Loss:
    '''
    A common loss class which calculates the data and
    regularization losses for given model output and
    ground truth values and returns the average loss
    '''
    # Reset variables for accumulated loss
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate(self, output, y, *, include_regularization=False):
        '''
        Calculate the loss given model output and ground truth values.

        Parameters:
            output: Model output.
            y: Ground truth values.
            include_regularization (bool): Whether to include regularization loss.

        Returns:
            loss: The calculated loss.
        '''       
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularizationLoss()

    def calculate_accumulated(self, *, include_regularization=False):
        '''
        Calculate accumulated loss.

        Parameters:
            include_regularization (bool): Whether to include regularization loss.

        Returns:
            loss: The accumulated loss.
        '''
        data_loss = self.accumulated_sum / self.accumulated_count
        if not include_regularization:
            return data_loss

        return data_loss, self.regularizationLoss()

    def remember_trainable_layers(self, trainable_layers):
        '''
        Set/remember trainable layers.

        Parameters:
            trainable_layers: List of trainable layers.        
        '''
        self.trainable_layers = trainable_layers

    def regularizationLoss(self):
        '''
        Regularization loss calculation.

        Returns:
            regularization_loss: The calculated regularization loss.      
        '''
        regularization_loss = 0  # by default

        for Layer in self.trainable_layers:
            # L1 regularization
            if Layer.weight_regularizer_l1 > 0:
                regularization_loss += Layer.weight_regularizer_l1 * np.sum(np.abs(Layer.weights))
            if Layer.bias_regularizer_l1 > 0:
                regularization_loss += Layer.bias_regularizer_l1 * np.sum(np.abs(Layer.biases))

            # L2 regularization
            if Layer.weight_regularizer_l2 > 0:
                regularization_loss += Layer.weight_regularizer_l2 * np.sum(Layer.weights * Layer.weights)
            if Layer.bias_regularizer_l2 > 0:
                regularization_loss += Layer.bias_regularizer_l2 * np.sum(Layer.biases * Layer.biases)

        return regularization_loss
    

class LossCategoricalCrossentropy(Loss):
    '''
    Calculates the Categorical cross-entropy loss
    '''

    def forward(self, y_pred, y_true):
        '''
        Forward pass for Categorical Cross-Entropy loss.

        Parameters:
            y_pred: Predicted values.
            y_true: Ground truth values.

        Returns:
            negative_log_likelihoods: Calculated negative log likelihoods.       
        '''
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        """
        Calculates the gradient of the loss with respect to the input.

        Args:
            dvalues (np.ndarray): The derivative of loss w.r.t the output.
            y_true (np.ndarray): The true labels.

        Returns:
            None (updates self.dinputs)
        """
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class LossBinaryCrossentropy(Loss):
    '''
    Calculate the Binary CE loss i.e., 0 or 1
    '''
    def forward(self, y_pred, y_true):
        '''
        Forward pass for Binary Cross-Entropy loss.

        Parameters:
            y_pred: Predicted values.
            y_true: Ground truth values.

        Returns:
            sample_losses: Calculated sample losses.        
        '''
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true*np.log(y_pred_clipped) + (1 - y_true)*np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        '''
        Backward pass for Binary Cross-Entropy loss.

        Parameters:
            dvalues: Derivative of the loss with respect to the output.
            y_true: True labels.

        Returns:
            None (updates self.dinputs).
        '''
        samples = len(dvalues)
        outputs = len(dvalues[0])
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs = self.dinputs / samples


class LossMeanSquaredError(Loss):  # L2 loss
    '''
    Calculates the MSE loss.
    '''
    def forward(self, y_pred, y_true):
        '''
         Forward pass for Mean Squared Error loss.

        Parameters:
            y_pred: Predicted values.
            y_true: Ground truth values.

        Returns:
            sample_losses: Calculated sample losses.       
        '''
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        '''
        Backward pass for Mean Squared Error loss.

        Parameters:
            dvalues: Derivative of the loss with respect to the output.
            y_true: True labels.

        Returns:
            None (updates self.dinputs).        
        '''
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = -2 * (y_true - dvalues)/outputs
        self.dinputs = self.dinputs / samples


class LossMeanAbsoluteError(Loss):  # L1 loss
    '''
    Calculates the MAE loss.
    '''
    def forward(self, y_pred, y_true):
        '''
        Forward pass for Mean Absolute Error loss.

        Parameters:
            y_pred: Predicted values.
            y_true: Ground truth values.

        Returns:
            sample_losses: Calculated sample losses.       
        '''
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        '''
        Backward pass for Mean Absolute Error loss.

        Parameters:
            dvalues: Derivative of the loss with respect to the output.
            y_true: True labels.

        Returns:
            None (updates self.dinputs).        
        '''
        samples = len(dvalues)
        outputs = len(dvalues[0])
        self.dinputs = np.sign(y_true - dvalues)/outputs
        self.dinputs = self.dinputs / samples