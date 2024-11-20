import numpy as np
class Neuron:
    '''
    Stores basic operations fundamental to all neuron types
    '''
    def __init__(self, n_inputs, activation_function=None):
        '''
        Generates a neuron with the specified number of inputs as well as random weights and bias values from 0 to 1.
        The activation function can be specified through the activation_function class
        '''
        self.weights = [np.random.random() for input in range(n_inputs)]
        self.bias = np.random.random()
        self.activation_function = activation_function or (lambda x : x)

    def forward(self, inputs):
        '''
        Defines the output value of the neuron by the dot product of the weights and inputs, with the addition of the
        bias term. The activation function is defined inside the activation_function class and is initialised with
        creation.
        '''
        self.output = self.activation_function(np.dot(self.weights, inputs)+self.bias)
