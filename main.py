import network, activation_function as af

# Create a simple neural network
neural_net = network.Network(2)
neural_net.addLayer(3, activation_function=af.activation_function.sigmoid)
neural_net.addLayer(4)
neural_net.addLayer(2)