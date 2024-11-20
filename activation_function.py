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
        return 1/(1+np.exp(-x));
    @staticmethod
    def relu(x):
        '''
        Acts as the zero function on negative values and identity on positive values.
        '''
        return np.maximum(0,x);