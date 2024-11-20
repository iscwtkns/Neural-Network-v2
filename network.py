import layer

class Network:
    def __init__(self, n_inputs):
        '''
        Initialises an empty network with a specified number of inputs
        '''
        self.n_inputs = n_inputs
        self.layers = []
    def addLayer(self, n_neurons, activation_function=None):
        '''
        Adds a layer to the network with a specified number of neurons and activation function
        '''
        if (len(self.layers) == 0):
            newLayer = layer.Layer(n_neurons, self.n_inputs, activation_function)
        else:
            newLayer = layer.Layer(n_neurons, len(self.layers[-1].neurons), activation_function)
        self.layers += [newLayer]
    def forward(self, inputs):
        '''
        Propagates an input through the neural network and stores the final outputs
        '''
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.outputs
            self.outputs = inputs
