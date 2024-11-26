import network, math_utils as mu, matplotlib.pyplot as plt, numpy as np


# Create a simple neural network
neural_net = network.Network(1)
neural_net.addLayer(1)


inputX = [x for x in range(100)]
inputY = [2*x+(20*np.random.random()-10) for x in inputX]
scaleFactorX = np.max(inputX)
scaleFactorY = np.max(inputY)
inputX = inputX/scaleFactorX
inputY = inputY/scaleFactorY



neural_net.train(inputX, inputY, 0.1,1000, announcer=True,learn_decay=1)

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
