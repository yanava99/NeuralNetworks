import numpy as np
import copy


# Class ActivationFunction has 3 methods: call function itself, call it's derivative
# and call derivative receiving result.
# It's going to be used when specifying activation functions for the layers.
class ActivationFunction:
    def __init__(self, function, derivative, derivativeFromAnswer):
        self.func = function
        self.derivative = np.vectorize(derivative)
        self.derivativeFromAnswer = np.vectorize(derivativeFromAnswer)

    def __call__(self, x):
        return self.func(x)

    def callDerivative(self, x):
        return self.derivative(x)

    def callDerivativeFromAnswer(self, x):
        return self.derivativeFromAnswer(x)


# Class ErrorFunction is similar to ActivationFunction class.
# The difference between them if that ActivationFunction methods need 1 variable,
# while ErrorFunction methods need 2 variables.
class ErrorFunction:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative

    def __call__(self, x, y):
        return self.function(x,y)

    def callDerivative(self, x, y):
        return self.derivative(x, y)


# Class Layer has information about number of neurons, activation function, dropout probability,
# current weights and bias.
class Layer:
    def __init__(self, n, function, dropout=0):
        self.n = n
        self.function = function
        self.dropout = dropout
        self.weights = np.array([[]])
        self.bias = np.array([])

    def setWeights(self, array):
        # specify dimensions
        self.weights = array.copy()

    def setBias(self, array):
        self.bias = array.copy()

    def setDefaultWeights(self, previousN):
        self.weights = np.full((self.n, previousN), 1 / (self.n * previousN))

    def setDefaultBias(self):
        self.bias = np.zeros(self.n)

    def goThrough(self, x):
        x = self.weights @ x + self.bias
        x = self.function(x)
        return x

    def updateWeights(self, grad, step):
        self.weights -= step * grad

    def updateBias(self, grad, step):
        self.bias -= step * grad


# Class Model sets structure of the future Model.
# It is changeable: you can add new layers.
class Model:
    def __init__(self, dimInput, dimOutput):
        self.dimX = dimInput
        self.dimY = dimOutput
        self.layers = []

    def addLayer(self, layer):
        self.layers.append(layer)


# Class Network is not changeable, its entity is created on the base of a model.
class Network:
    def __init__(self, model, errorFunc):  # add a default errorFunc
        # add a check if number of neurons on the top layer equals dimOutput
        # OR if they are not equal add a layer (with(out) activation function? weights?)
        self.dimX = model.dimX
        self.dimY = model.dimY
        self.layers = copy.deepcopy(model.layers)
        for i in range(len(self.layers)):
            if self.layers[i].weights.size == 0:
                if i == 0:
                    self.layers[i].setDefaultWeights(self.dimX)
                else:
                    self.layers[i].setDefaultWeights(self.layers[i-1].n)
            if self.layers[i].bias.size == 0:
                self.layers[i].setDefaultBias()
        self.errorFunc = errorFunc

    def fit(self, x, y, step=0.5):
        tempX = [x]  # saving outputs of each neuron
        for layer in self.layers:
            x = layer.goThrough(x)
            tempX.append(x)
        buffer = self.errorFunc.callDerivative(x, y)  # accumulation of derivatives
        for i in reversed(range(len(self.layers))):
            buffer *= self.layers[i].function.derivativeFromAnswer(tempX[i + 1])
            self.layers[i].updateBias(buffer, step)
            gradWeights = buffer.reshape(buffer.shape[0], 1) @ tempX[i].reshape(1, tempX[i].shape[0])
            buffer = buffer @ self.layers[i].weights
            self.layers[i].updateWeights(gradWeights, step)
        return self.errorFunc(x, y)

    def fitForCrossEntropy(self, x, y, step=0.5):
        tempX = [x]  # saving outputs of each neuron
        for layer in self.layers:
            x = layer.goThrough(x)
            tempX.append(x)
        buffer = - y * (1 - tempX[-1])
        self.layers[-1].updateBias(buffer, step)
        gradWeights = buffer.reshape(buffer.shape[0], 1) @ tempX[-2].reshape(1, tempX[-2].shape[0])
        buffer = buffer @ self.layers[-1].weights
        self.layers[-1].updateWeights(gradWeights, step)
        for i in reversed(range(len(self.layers) - 1)):
            buffer *= self.layers[i].function.derivativeFromAnswer(tempX[i + 1])
            self.layers[i].updateBias(buffer, step)
            gradWeights = buffer.reshape(buffer.shape[0], 1) @ tempX[i].reshape(1, tempX[i].shape[0])
            buffer = buffer @ self.layers[i].weights
            self.layers[i].updateWeights(gradWeights, step)
        # return self.errorFunc(x, y)

    def setDefaultParameters(self):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].setDefaultWeights(self.dimX)
            else:
                self.layers[i].setDefaultWeights(self.layers[i - 1].n)
            self.layers[i].setDefaultBias()

    def evaluate(self, x):
        for layer in self.layers:
            x = layer.goThrough(x)
        return x


# Implementations of commonly used activation functions.

def logisticFunc(x):
    return 1 / (1 + np.exp(-x))


def logisticDerivative(x):
    return logisticFunc(x) * (1 - logisticFunc(x))


def logisticDerivativeFromAnswer(ans):
    return ans * (1 - ans)


def reLUFunc(x):
    if x < 0:
        return 0.
    else:
        return x


def reLUDerivative(x):
    if x < 0:
        return 0.
    else:
        return 1.


def reLUDerivativeFromAnswer(ans):
    if ans > 0:
        return 1.
    else:
        return 0.


def softmaxFunc(x):
    div = 0
    maximum = np.amax(x)
    for xi in x:
        div += np.exp(xi - maximum)
    vectorExp = np.exp
    return vectorExp(x - maximum) / div


def softmaxDerivative(x):
    # derivative is computed only with respect to the argument coordinate with the same index
    return softmaxFunc(x) * (1 - softmaxFunc(x))


def softmaxDerivativeFromAnswer(ans):
    # derivative is computed only with respect to the argument coordinate with the same index
    return ans * (1 - ans)


# Implementations of commonly used error functions.

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


def crossEntropyErrorFunction(x, y):
    # add a check if x.size == y.size
    result = 0.
    for i in range(x.size):
        result += y[i] * np.log(x[i])
    result *= -1
    return result


def crossEntropyErrorDerivative(x, y):
    return -y / x


# Most commonly used activation and error functions.

logistic = ActivationFunction(logisticFunc, logisticDerivative, logisticDerivativeFromAnswer)
reLU = ActivationFunction(reLUFunc, reLUDerivative, reLUDerivativeFromAnswer)
softmax = ActivationFunction(softmaxFunc, softmaxDerivative, softmaxDerivativeFromAnswer)

squaredError = ErrorFunction(squaredErrorFunction, squaredErrorDerivative)
crossEntropyError = ErrorFunction(crossEntropyErrorFunction, crossEntropyErrorDerivative)
