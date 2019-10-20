import NeuralNetworksLibrary as nnl
import numpy as np
import math


def logisticFunc(x):
    return 1 / (1 + math.exp(-x))


def logisticDerivative(x):
    return logisticFunc(x) * (1 - logisticFunc(x))


def logisticDerivativeFromAnswer(ans):
    return ans * (1 - ans)


def squaredErrorFunction(x, y):
    # add a check if x.size == y.size
    result = 0.
    for i in range(x.size):
        result += (y[i] - x[i])**2
    result /= 2
    return result


def squaredErrorDerivative(x, y):
    # works for numpy.array
    # each element of a result contains derivative on this component
    return x - y


logistic = nnl.ActivationFunction(logisticFunc, logisticDerivative, logisticDerivativeFromAnswer)
model = nnl.Model(2, 2)
layer1 = nnl.Layer(2, logistic)
layer1.setWeights(np.array([[0.15, 0.20], [0.25, 0.30]]))
layer1.setBias([0.35, 0.35])
model.addLayer(layer1)
layer2 = nnl.Layer(2, logistic)
layer2.setWeights(np.array([[0.40, 0.45], [0.50, 0.55]]))
layer2.setBias(np.array([0.60, 0.60]))
model.addLayer(layer2)

squaredError = nnl.ErrorFunction(squaredErrorFunction, squaredErrorDerivative)
network = nnl.Network(model, squaredError)

print(network.fit(np.array([0.05, 0.10]), np.array([0.01, 0.99])))
print()
for layer in network.layers:
    print(layer.weights)
