# This is a check of updating weights using gradient descent and backpropagation.
# Result weights are to be equal to the ones calculated in the example you can find at
# https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/.


import NeuralNetworksLibrary as nnl
import numpy as np


model = nnl.Model(2, 2)
layer1 = nnl.Layer(2, nnl.logistic)
layer1.setWeights(np.array([[0.15, 0.20], [0.25, 0.30]]))
layer1.setBias(np.array([0.35, 0.35]))
model.addLayer(layer1)
layer2 = nnl.Layer(2, nnl.logistic)
layer2.setWeights(np.array([[0.40, 0.45], [0.50, 0.55]]))
layer2.setBias(np.array([0.60, 0.60]))
model.addLayer(layer2)

network = nnl.Network(model, nnl.squaredError)
network1 = nnl.Network(model, nnl.squaredError)

print(network.fit(np.array([0.05, 0.10]), np.array([0.01, 0.99]), 0.5))
print()
for layer in network.layers:
    print(layer.weights)
