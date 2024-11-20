import numpy as np
class activation_function:
    '''
    Several different activation functions for different purposes
    '''
    @staticmethod
    def sigmoid(x):
        '''
        Sets large positive values to 1, large negative values to 0 and smoothly interpolates between
        '''
        return 1/(1+np.exp(-x))
    @staticmethod
    def sigmoidDerivative(x):
        return activation_function.sigmoid(x)*(1-activation_function.sigmoid(x))
    def relu(x):
        '''
        Acts as the zero function on negative values and identity on positive values.
        '''
        return np.maximum(0,x)
    def reluDerivative(x):
        if (x<0):
            return 0
        else:
            return 1
class data_function:
    @staticmethod
    def cost(prediction, actual):
        if (isinstance(prediction, (float, int)) and isinstance(actual, (float, int))):
            return (prediction - actual) ** 2
        else:
            prediction = np.array(prediction)
            actual = np.array(actual)
            return np.mean((prediction-actual)**2)
            
            