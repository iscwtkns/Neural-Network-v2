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
        self.bias = 0
        if activation_function == mu.activation_function.relu:
            self.bias = 0.01*np.random.random()
        self.weights = [float(0.0)]*n_inputs
        self.activation_function = activation_function or (lambda x : x)
        acfun = self.activation_function
        print(acfun)
        if acfun == mu.activation_function.sigmoid:
            self.activation_derivative = mu.activation_function.sigmoidDerivative
        elif acfun == mu.activation_function.relu:
            self.activation_derivative = mu.activation_function.reluDerivative
        else:
            self.activation_derivative = mu.activation_function.constant



        
    def forward(self, inputs):
        '''
        Defines the output value of the neuron by the dot product of the weights and inputs, with the addition of the
        bias term. The activation function is defined inside the activation_function class and is initialised with
        creation.
        '''
        if isinstance(inputs, (float, int)):
            self.inputs = inputs
        else:
            if isinstance(inputs[0], (float,int)):
                self.inputs = [float(input) for input in inputs]
            else:
                self.inputs = [float(input[0]) for input in inputs]

        self.weighted_output = np.dot(self.weights, inputs)+self.bias
        self.activation = float(self.activation_function(self.weighted_output))
    def adjustWeights(self, learn_step, noise=0):
        '''
        Adjusts weights according to the error term. Need to run calculate_error_terms from network.py first
        '''
        self.weight_derivatives = [] * len(self.weights)
        for i in range(len(self.weights)):
            noise_value = (np.random.random()-1/2)*noise
            if (isinstance(self.inputs,(float,int))):
                if isinstance(self.error_term, (float,int)):
                    self.weight_derivatives[i] = self.error_term*self.inputs
                    self.weights[i] -= learn_step*self.weight_derivatives[i] + noise_value
                else:
                    self.weight_derivatives[i] = self.error_term[0]*self.inputs
                    self.weights[i] -= learn_step*self.weight_derivatives[i] + noise_value
            else:
                if isinstance(self.error_term, (float,int)):
                    self.weight_derivatives[i] = self.error_term*self.inputs[i]
                    self.weights[i] -= learn_step*self.weight_derivatives[i] + noise_value
                else:
                    self.weight_derivatives[i] = self.error_term[0]*self.inputs[i]
                    self.weights[i] -= learn_step*self.weight_derivatives[i] + noise_value

    def adjustBias(self, learn_step, noise = (10**(-8))):
        '''
        Adjusts bias according to the error term, need to run calculate_error_terms from network.py first
        '''
        if isinstance(self.error_term, (float,int)):
            self.bias -= learn_step*self.error_term + (np.random.random()-1/2)*noise
        else:
            self.bias -= learn_step*self.error_term[0] + (np.random.random()-1/2)*noise

   

