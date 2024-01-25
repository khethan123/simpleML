import numpy as np

class Accuracy:
    '''
    Calculates accuracy based on predictions and ground truth values.
    '''
    def calculate(self, predictions, y):
        '''
        Calculate accuracy given predictions and ground truth values.

        Parameters:
            predictions: The predicted values.
            y: The ground truth values.

        Returns:
            accuracy: The calculated accuracy.
        '''
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return accuracy

    def calculate_accumulated(self):
        '''
        Calculate accumulated accuracy for each step in an epoch.

        Returns:
            accuracy: The accumulated accuracy.
        '''    
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy

    def new_pass(self):
        '''
        Reset variables for accumulated accuracy.
        '''
        self.accumulated_sum = 0
        self.accumulated_count = 0


# Accuracy calculation for regression model
class AccuracyRegression(Accuracy):

    def __init__(self):
        self.precision = None

    def init(self, y, reinit=False):
        '''
        Initialize precision value based on passed ground truth values.

        Parameters:
            y: The ground truth values.
            reinit (bool): Whether to reinitialize precision.

        Notes:
            Precision is set to std(y)/250.
        '''
        if self.precision is None or reinit:
            self.precision = np.std(y)/250

    def compare(self, predictions, y):
        '''
        Compare predictions to ground truth values for regression.

        Parameters:
            predictions: The predicted values.
            y: The ground truth values.

        Returns:
            Boolean array indicating whether predictions are close to ground truth.
        '''
        return np.absolute(predictions-y) < self.precision


# Accuracy calc for classification model
class AccuracyCategorical(Accuracy):

    def init(self, y):
        pass  # Used in Model.train code

    def compare(self, predictions, y):
        '''
        Compare predictions to ground truth values for classification.

        Parameters:
            predictions: The predicted values.
            y: The ground truth values.

        Returns:
            Boolean array indicating whether predictions are equal to ground truth.
        '''
        if len(y.shape) == 2:
            y = np.argmax(y,axis=1)
        return predictions == y