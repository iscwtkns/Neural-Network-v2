import neuron, math_utils as mu, numpy as np

class Layer:
    '''
    Stores collections of neurons that make up single layers of neural networks.
    '''
    def __init__(self, n_neurons, n_inputs, activation_function=None):
        '''
        Creates a new layer of a specified number of neurons with particular number of inputs and specified activation 
        function
        '''
        self.n_inputs = n_inputs
        self.neurons = [neuron.Neuron(int(n_inputs), activation_function=activation_function) for i in range(n_neurons)]
        self.activation_function = activation_function
        self.assign_weights()
    def forward(self,inputs):
        '''
        Uses forward function in neuron class on all neurons inside the layer, stores the outputs as a separate variable
        '''
        for i in range(len(self.neurons)):
            self.neurons[i].forward(inputs)
        self.activation = [self.neurons[i].activation for i in range(len(self.neurons))]
        self.inputs = [self.neurons[i].inputs for i in range(len(self.neurons))]
    def assign_error_terms(self, error_vector):
        '''
        Takes in a vector of error terms (list) sets the error term of each neuron in the layer to coincide with index
        in the vector
        '''
        self.error_vector = error_vector
        for i in range(len(self.neurons)):
            self.neurons[i].error_term = error_vector[i]
    def adjust_weights_and_biases(self, learn_step):
        for neuron in self.neurons:
            neuron.adjustBias(learn_step)
            neuron.adjustWeights(learn_step)
    def assign_weights(self):
        if self.activation_function == mu.activation_function.sigmoid:
            # Xavier Weight Initialisation
            limit = np.sqrt(6/(len(self.neurons)+self.n_inputs))
            weight_matrix = np.random.uniform(-limit,limit,(len(self.neurons),self.n_inputs))
        if self.activation_function == mu.activation_function.relu:
            # He Weight Initialisation
            std_dev = np.sqrt(2/self.n_inputs)
            weight_matrix = np.random.normal(0,std_dev,(len(self.neurons),self.n_inputs))
        else:
            weight_matrix = np.random.random((len(self.neurons),self.n_inputs))
        for i in range(len(weight_matrix)):
            self.neurons[i].weights = weight_matrix[i]

