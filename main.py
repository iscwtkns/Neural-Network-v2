import network, math_utils as mu, matplotlib.pyplot as plt, numpy as np


# Create a simple neural network
neural_net = network.Network(1)
neural_net.addLayer(2, activation_function=mu.activation_function.leakyRelu)
neural_net.addLayer(5, activation_function=mu.activation_function.leakyRelu)
neural_net.addLayer(2, activation_function=mu.activation_function.leakyRelu)
neural_net.addLayer(1)


inputX = [x for x in range(1000)]
inputY = [x**2 for x in inputX]
scaleFactorX = np.max(inputX)
scaleFactorY = np.max(inputY)
inputX = inputX/scaleFactorX
inputY = inputY/scaleFactorY
random = int(np.floor(np.random.random()*1000))
randomInput = inputX[random]
neural_net.learn(randomInput,inputY[random])
print("Random Input was:", randomInput)
for layer in neural_net.layers:
    print("New layer")
    for neuron in layer.neurons:
        print("New neuron")
        print("Input:", neuron.inputs)
        print("Error Term:", neuron.error_term)
        print("Weight values:", neuron.weights)
        print("Bias value:", neuron.bias)
        print("Weighted Output:", neuron.weighted_output)
        print("activation:", neuron.activation)
for value in neural_net.outputs:
    error = (value-inputY[random])**2
print("Error:", error, "Cost:", mu.data_function.cost(neural_net.outputs[0],inputY[random]))


neural_net.train(inputX, inputY, 0.01,1000, announcer=True,learn_decay=1)
for layer in neural_net.layers:
    print("New layer")
    for neuron in layer.neurons:
        print("New neuron")
        print("Input:", neuron.inputs)
        print("Error Term:", neuron.error_term)
        print("Weight values:", neuron.weights)
        print("Bias value:", neuron.bias)
        print("Weighted Output:", neuron.weighted_output)
        print("activation:", neuron.activation)
        print("Bias Derivative:", neuron.bias_derivatives)
        print("Weight Derivatives:",neuron.weight_derivatives)
neural_net.calculateNumericalDerivatives(inputX, inputY)
predictions = neural_net.predict(inputX)
predictions = [predictions[i]*scaleFactorY for i in range(len(predictions))]
actual_values = inputY*scaleFactorY
input_values = inputX*scaleFactorX
plt.scatter(input_values,actual_values)
plt.plot(input_values,predictions)
plt.show()
cost = mu.data_function.cost(predictions, actual_values)

#Visualise net
visual_net = neural_net
