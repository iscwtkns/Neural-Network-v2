import layer, math_utils as mu, numpy as np

class Network:
    def __init__(self, n_inputs):
        '''
        Initialises an empty network with a specified number of inputs
        '''
        self.n_inputs = n_inputs
        self.layers = []
        self.normalisation_factor = 1
    
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
        self.inputs = inputs
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.activation
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
    
    def learn(self, inputs, desired_output):
        '''
        Takes in inputs that match length of input neurons and output matching length of output neurons.
        Computes the error term for each neuron and then stores derivatives in array for each neuron
        '''
        self.forward(inputs)
        self.calculate_error_terms(desired_output)
        for layer in self.layers:
            for neuron in layer.neurons:
                for i in range(len(neuron.weights)):
                    if isinstance(neuron.inputs, (float,int)):
                        neuron.weight_derivatives[i].append(neuron.error_term*neuron.inputs)
                    else:
                        neuron.weight_derivatives[i].append(neuron.error_term*neuron.inputs[i])
                neuron.bias_derivatives.append(neuron.error_term)
    
   
    def train(self, inputs, desired_outputs, learn_step, epochs, announcer=False, learn_decay =0.99, batch_size = 10):
        '''
        Takes in a list of inputs corresponding to a large sample of data. Uses the learn function iteratively to 
        learn over the entire data set for a number of epochs. Data does need to be normalised.
        '''
    
        inputs = [inputs[i*self.n_inputs:(i+1)*self.n_inputs][0] for i in range(len(inputs)//self.n_inputs)]
        
        outputs = [desired_outputs[i*self.n_inputs:(i+1)*self.n_inputs][0] for i in range(len(inputs)//self.n_inputs)]
        for i in range(epochs):
            for layer in self.layers:
                for neuron in layer.neurons:
                    neuron.weight_derivatives = [[] for i in range(len(neuron.weights))]
                    neuron.bias_derivatives = []   
            for j in range(len(inputs)):
                #Sets up large gradient arrays that can be averaged to find cost gradients
                self.learn(inputs[j],outputs[j])
            for layer in self.layers:
                for neuron in layer.neurons:
                    neuron.weight_derivatives =[(np.mean(neuron.weight_derivatives[i])) for i in range(len(neuron.weights))]
                    neuron.bias_derivatives = np.mean(neuron.bias_derivatives)
                    neuron.error_term = neuron.bias_derivatives
                layer.adjust_weights_and_biases(learn_step)
            predictions = self.massForward(inputs)
            cost = mu.data_function.cost(predictions, inputs)
            if announcer:
                print("Epoch Completed, cost function at:",cost)
                
                        
            learn_step *= learn_decay


    def calculate_error_terms(self, desired_output):
        '''
        Uses the currently saved input, activation and weighted_output variable from each neuron to evaluate error terms
        and set them
        '''
        # Set error terms of the output layer
        output_error_vector = self.calculate_output_error_vector(desired_output)
        self.layers[-1].assign_error_terms(output_error_vector)

        #Calculate output error vectors and assign to rest of layers sequentially
        for i in range(len(self.layers)-1):
            error_vector = []
            for j in range(len(self.layers[-2-i].neurons)):
                currentNeuron = self.layers[-2-i].neurons[j]
                error_term = 0
                for k in range(len(self.layers[-1-i].neurons)):
                    neuron = self.layers[-1-i].neurons[k]
                    error_term += neuron.error_term*neuron.weights[j]
                error_term *= currentNeuron.activation_derivative(currentNeuron.weighted_output)
                error_vector.append(error_term)
            
            self.layers[-2-i].assign_error_terms(error_vector)
        
    def calculate_output_error_vector(self, desired_output):

        output_error_vector = []
        for i in range(len(self.layers[-1].neurons)):
            neuron = self.layers[-1].neurons[i]
            if isinstance(desired_output, (float, int)):
                #print("Cost Derivative:",mu.data_function.costDerivative(neuron.activation,desired_output))
                #print("Activation Derivative:",neuron.activation_derivative)
                output_error_vector.append(mu.data_function.costDerivative(neuron.activation,desired_output)*neuron.activation_derivative(neuron.weighted_output))
            else:
                output_error_vector.append(mu.data_function.costDerivative(neuron.activation,desired_output[i])*neuron.activation_derivative(neuron.weighted_output))
        
        return output_error_vector
    
    
    def predict(self, input):
        '''
        Scales inputs to the correct magnitude and then propagates through the network, rescales before outputting. Network
        should be trained before use.
        '''
        if (isinstance(input, (float,int)) or len(input) == self.n_inputs):
            self.forward(input)
            predictions = self.outputs
            return predictions
        else:
            predictions = []
            for value in input:
                self.forward(value)
                predictions += self.outputs
            return predictions
    
    def calculateNumericalDerivatives(self,input, actual):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            for j in range(len(layer.neurons)):
                neuron = layer.neurons[j]
                neuron.numerical_weight_derivatives = [0]*len(neuron.weights)
                for k in range(len(neuron.weights)):
                    neuron.weights[k] += 1e-3
                    prediction = self.predict(input)
                    loss1 = mu.data_function.cost(prediction, actual)
                    neuron.weights[k] -= 2e-3
                    prediction = self.predict(input)
                    loss2 = mu.data_function.cost(prediction,actual)
                    neuron.weights[k] += 1e-3
                    neuron.numerical_weight_derivatives[k] = (loss1 - loss2) / 2e-3
                neuron.bias += 1e-3
                predictions = self.predict(input)
                loss1 = mu.data_function.cost(predictions, actual)
                neuron.bias -= 2e-3
                predictions = self.predict(input)
                loss2 = mu.data_function.cost(predictions, actual)
                neuron.bias += 1e-3
                neuron.numerical_bias_derivative = (loss1 - loss2) / 2e-3
