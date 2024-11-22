import network, math_utils as mu, matplotlib.pyplot as plt, numpy as np


# Create a simple neural network
neural_net = network.Network(1)
neural_net.addLayer(5, activation_function=mu.activation_function.leakyRelu)
neural_net.addLayer(5, activation_function=mu.activation_function.leakyRelu)
neural_net.addLayer(5, activation_function=mu.activation_function.leakyRelu)
neural_net.addLayer(1, activation_function=mu.activation_function.leakyRelu)

inputX = [x for x in range(0,1000)]
inputY = [x**2 for x in inputX]
scaleFactor = np.maximum(np.max(inputX),np.max(inputY))
inputX = inputX/scaleFactor
inputY = inputY/scaleFactor
for i in range(100):
    randomIndices = [int(np.floor(np.random.random()*1000)) for j in range(16)]
    randomX = [0]*16
    randomY = [0]*16
    for j in range(16):
        randomX[j] = inputX[randomIndices[j]]
        randomY[j] = inputY[randomIndices[j]]
    neural_net.train(randomX, randomY, 0.001,1, announcer=True)

neural_net.calculateNumericalDerivatives(inputX, inputY)
predictions = neural_net.predict(inputX)
actual_values = inputY
input_values = inputX
plt.scatter(input_values,actual_values)
plt.plot(input_values,predictions)
plt.show()
cost = mu.data_function.cost(predictions, actual_values)

#Visualise net
visual_net = neural_net
