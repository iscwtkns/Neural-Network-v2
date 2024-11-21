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
    def constant(x):
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
    @staticmethod
    def costDerivative(prediction, actual):
        if (isinstance(prediction, (float, int)) and isinstance(actual, (float, int))):
            return 2*(prediction-actual)
        else:
            prediction = np.array(prediction)
            actual = np.array(actual)
            return np.mean(2*(prediction-actual))
class array_function:
    @staticmethod
    def strip(x):
        '''
        Takes in the annoying n x 1 x 1 matrix to return a list of length n
        '''
        new_x = []
        return np.array(i[0] for i in x)