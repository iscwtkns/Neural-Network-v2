import numpy as np, math_utils as mu
class Neuron:
    '''
    Stores basic operations fundamental to all neuron types
    '''
    def __init__(self, n_inputs, activation_function=None):
        '''
        Generates a neuron with the specified number of inputs as well as random weights and bias values from 0 to 1.
        The activation function can be specified through the activation_function class
        '''
        self.weights = [2*np.random.random()-1 for input in range(n_inputs)]
        self.bias = 2*np.random.random()-1
        self.activation_function = activation_function or (lambda x : x)
        self.define_activation_derivative()

    def forward(self, inputs):
        '''
        Defines the output value of the neuron by the dot product of the weights and inputs, with the addition of the
        bias term. The activation function is defined inside the activation_function class and is initialised with
        creation.
        '''
        if isinstance(inputs, (float, int)):
            self.inputs = inputs
        else:
            self.inputs = [float(input) for input in inputs]
        self.weightedOutput = np.dot(self.weights, inputs)+self.bias
        self.activation = self.activation_function(self.weightedOutput)

    def adjustWeights(self, learn_step):
        '''
        Adjusts weights according to the error term. Need to run calculate_error_terms from network.py first
        '''
        for i in range(len(self.weights)):
            self.weights[i] -= learn_step*self.error_term*self.inputs[i]
    def adjustBias(self, learn_step):
        '''
        Adjusts bias according to the error term, need to run calculate_error_terms from network.py first
        '''
        self.bias -= learn_step*self.error_term
    
    def define_activation_derivative(self):
        acfun = self.activation_function
        if acfun == mu.activation_function.sigmoid:
            self.activation_derivative = mu.activation_function.sigmoidDerivative
        elif acfun == mu.activation_function.relu:
            self.activation_derivative = mu.activation_function.reluDerivative
        else:
            self.activation_derivative = mu.activation_function.constant


