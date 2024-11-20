import numpy as np, neuron, activation_function as af

class Layer:
    '''
    Stores collections of neurons that make up single layers of neural networks.
    '''
    def __init__(self, n_neurons, n_inputs, activation_function=None):
        '''
        Creates a new layer of a specified number of neurons with particular number of inputs and specified activation 
        function
        '''
        self.neurons = [neuron.Neuron(n_inputs, activation_function=activation_function) for i in range(n_neurons)]
    def forward(self,inputs):
        '''
        Uses forward function in neuron class on all neurons inside the layer, stores the outputs as a separate variable
        '''
        for i in range(len(self.neurons)):
            self.neurons[i].forward(inputs)
        self.outputs = [self.neurons[i].output for i in range(len(self.neurons))]
    


