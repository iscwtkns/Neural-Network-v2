import network, math_utils as mu, matplotlib.pyplot as plt, numpy as np


# Create a simple neural network
neural_net = network.Network(1)
neural_net.addLayer(5, activation_function=mu.activation_function.sigmoid)
neural_net.addLayer(1, activation_function=mu.activation_function.relu)
inputX = [x/10 for x in range(-50,51)]
inputY = [np.sin(x) for x in inputX]
neural_net.train(inputX, inputY, 0.000000001, 1000, announcer=True)
for layer in neural_net.layers:
    print("Layer")
    for neuron in layer.neurons:
        print("Activation:",neuron.activation)
        print("Activation Derivative:", neuron.activation_derivative(neuron.activation))
        print("Error Term:",neuron.error_term)
        print("Weighted Output:",neuron.weighted_output)
        print("Weights:",neuron.weights)
neural_net.calculateNumericalDerivatives(inputY)
predictions = neural_net.predict(inputX)
actual_values = inputY
input_values = inputX
plt.scatter(input_values,actual_values)
plt.plot(input_values,predictions)
plt.show()
#Visualise net
visual_net = neural_net
