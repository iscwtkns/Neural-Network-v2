import network, math_utils as mu

# Create a simple neural network
neural_net = network.Network(1)
neural_net.addLayer(3, activation_function=mu.activation_function.relu)
neural_net.addLayer(4, activation_function=mu.activation_function.relu)
neural_net.addLayer(1)

inputX = [i/5 for i in range(50)]
inputY = [i**2/25 for i in range(50)]
neural_net.learn(inputX[3],inputY[3],1)
for layer in neural_net.layers:
    print(layer.error_vector)
    for neuron in layer.neurons:
        print(neuron.error_term)