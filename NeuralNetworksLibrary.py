import numpy as np
import copy


# Class ActivationFunction has 3 methods: call function itself, call it's derivative
# and call derivative receiving result.
# It's going to be used when specifying activation functions for the layers.
class ActivationFunction:
    def __init__(self, function, derivative, derivativeFromAnswer, multivariate=False):
        self.multivariate = multivariate
        if multivariate:
            self.func = function
            self.derivativeFromAnswer = derivativeFromAnswer
            self.derivative = derivative
        else:
            self.func = np.vectorize(function)
            self.derivativeFromAnswer = np.vectorize(derivativeFromAnswer)
            self.derivative = np.vectorize(derivative)

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


# Class Layer has information about number of neurons, activation function, current weights and bias.
class Layer:
    def __init__(self, n, function):
        self.n = n
        self.function = function
        self.weights = np.array([[]])
        self.bias = np.array([])

    def setWeights(self, array):
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


# Class Model sets structure of the future neural network.
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
        self.dimX = model.dimX
        self.dimY = model.dimY
        self.layers = copy.deepcopy(model.layers)
        if len(self.layers) == 0 or self.layers[-1].n != self.dimY:
            newLayer = Layer(self.dimY, identity)
            self.layers.append(newLayer)
        for i in range(len(self.layers)):
            if self.layers[i].weights.size == 0:
                if i == 0:
                    self.layers[i].setDefaultWeights(self.dimX)
                else:
                    self.layers[i].setDefaultWeights(self.layers[i-1].n)
            if self.layers[i].bias.size == 0:
                self.layers[i].setDefaultBias()
        self.errorFunc = errorFunc

    # Method for training the network.
    def fit(self, x, y, step=0.1):
        tempX = [x]  # saving outputs of each neuron
        for layer in self.layers:
            x = layer.goThrough(x)
            tempX.append(x)
        buffer = self.errorFunc.callDerivative(x, y)  # accumulation of derivatives
        for i in reversed(range(len(self.layers))):
            if self.layers[i].function.multivariate:
                buffer = buffer @ self.layers[i].function.derivativeFromAnswer(tempX[i + 1])
            else:
                buffer *= self.layers[i].function.derivativeFromAnswer(tempX[i + 1])
            self.layers[i].updateBias(buffer, step)
            gradWeights = buffer.reshape(buffer.shape[0], 1) @ tempX[i].reshape(1, tempX[i].shape[0])
            buffer = buffer @ self.layers[i].weights
            self.layers[i].updateWeights(gradWeights, step)
        return self.errorFunc(x, y)

    # Setting weights and bias to default values.
    def setDefaultParameters(self):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].setDefaultWeights(self.dimX)
            else:
                self.layers[i].setDefaultWeights(self.layers[i - 1].n)
            self.layers[i].setDefaultBias()

    # Method for predicting.
    def decide(self, x):
        for layer in self.layers:
            x = layer.goThrough(x)
        if self.errorFunc == softmaxAndCrossEntropyBundleError:
            x = softmax(x)
        return x


# Implementations of commonly used activation functions and their derivatives.

def logisticFunction(x):
    return 1. / (1. + np.exp(-x))


def logisticDerivative(x):
    return logisticFunction(x) * (1. - logisticFunction(x))


def logisticDerivativeFromAnswer(ans):
    return ans * (1. - ans)


def reLUFunction(x):
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


def softmaxFunction(x):
    div = 0.
    maximum = np.amax(x)
    for xi in x:
        div += np.exp(xi - maximum)
    vectorExp = np.exp
    return vectorExp(x - maximum) / div


def softmaxDerivative(x):
    return softmaxDerivativeFromAnswer(softmaxFunction(x))


def softmaxDerivativeFromAnswer(ans):
    der = np.asarray([[- ans[i] * ans[j] if i != j else ans[i] * (1. - ans[i]) for j in range(ans.shape[0])]
                      for i in range(ans.shape[0])])
    return der


def identityFunction(x):
    return x


def identityDerivative(x):
    return 1.


def identityDerivativeFromAnswer(ans):
    return 1.


# Implementations of commonly used error functions and their derivatives.

def squaredErrorFunction(x, y):
    result = 0.
    for i in range(x.size):
        result += (y[i] - x[i])**2
    result /= 2.
    return result


def squaredErrorDerivative(x, y):
    return x - y


def crossEntropyErrorFunction(x, y):
    result = 0.
    for i in range(x.size):
        result += y[i] * np.log(x[i])
    result *= -1.
    return result


def crossEntropyErrorDerivative(x, y):
    return -y / x


def softmaxAndCrossEntropyBundleErrorFunction(x, y):
    return crossEntropyErrorFunction(softmaxFunction(x), y)


def softmaxAndCrossEntropyBundleErrorDerivative(x, y):
    return softmax(x) - y


# Most commonly used activation and error functions.

logistic = ActivationFunction(logisticFunction, logisticDerivative, logisticDerivativeFromAnswer)
reLU = ActivationFunction(reLUFunction, reLUDerivative, reLUDerivativeFromAnswer)
softmax = ActivationFunction(softmaxFunction, softmaxDerivative, softmaxDerivativeFromAnswer, True)
identity = ActivationFunction(identityFunction, identityDerivative, identityDerivativeFromAnswer)

squaredError = ErrorFunction(squaredErrorFunction, squaredErrorDerivative)
crossEntropyError = ErrorFunction(crossEntropyErrorFunction, crossEntropyErrorDerivative)
softmaxAndCrossEntropyBundleError = ErrorFunction(softmaxAndCrossEntropyBundleErrorFunction,
                                                  softmaxAndCrossEntropyBundleErrorDerivative)
