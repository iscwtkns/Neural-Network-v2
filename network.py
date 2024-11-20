import layer, numpy as np

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
    def massForward(self, inputList):
        '''
        Takes in large array of inputs and collects outputs one by one to form large array of corresponding predictions
        inputArray should be a one-dimensional list containing input values written in series
        '''
        inputs = []
        numInputs = len(inputList)//self.n_inputs
        for i in range(numInputs):
            inputs.append(inputList[i*self.n_inputs:(i+1)*self.n_inputs])
        output = [0] * len(inputList)
        for i in range(int(len(inputs)/self.n_inputs+0.0001)):
            self.forward(inputs[i])
            for j in range(len(self.outputs)):
                output[i * len(self.outputs) + j] = self.outputs[j]
        return output
