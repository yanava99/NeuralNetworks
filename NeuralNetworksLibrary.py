import numpy as np


# Class ActivationFunction has 2 methods: call function itself and call it's derivative.
# It's going to be used when specifying activation functions for the layers.
class ActivationFunction:
    def __init__(self, function, derivative, derivativeFromAnswer):
        self.func = np.vectorize(function)
        self.derivative = np.vectorize(derivative)
        self.derivativeFromAnswer = np.vectorize(derivativeFromAnswer)

    def __call__(self, x):
        return self.func(x)

    def callDerivative(self, x):
        return self.derivative(x)

    def callDerivativeFromAnswer(self, x):
        return self.derivativeFromAnswer(x)


# Class ErrorFunction is similar to activationFunction class.
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
        # add default initial weight
        # during the layer initialization? but user can call setWeights() method later
        # during the model initialization? network initialization?
        self.weights = np.array([[]])
        self.bias = np.array([])

    def setWeights(self, array):
        # specify dimensions
        self.weights = array.copy()
    
    def setDefaultWeights(self, previousN):
        self.weights = np.full((self.n, previousN), 1 / (self.n * previousN))

    def setDefaultBias(self):
        self.bias = np.zeros(self.n)

    def setBias(self, array):
        self.bias = array.copy()

    def goThrough(self, x):
        x = self.weights @ x + self.bias
        x = self.function(x)
        return x


# Class Model sets structure of the future Model.
# It is changeable: you can add new layers.
class Model:
    def __init__(self, dimInput, dimOutput):
        self.dimX = dimInput
        self.dimY = dimOutput
        self.layers = np.array([])

    def addLayer(self, layer):
        # redo to a non-numpy list?
        self.layers = np.append(self.layers, np.array([layer]), axis=0)


# Class Network is not changeable, its entity is created on the base of a model.
class Network:
    def __init__(self, model, errorFunc):  # add a default errorFunc
        # add a check if number of neurons on the top layer equals dimOutput
        # OR if they are not equal add a layer (with(out) activation function? weights?)
        self.dimX = model.dimX
        self.dimY = model.dimY
        self.layers = model.layers.copy()
        self.errorFunc = errorFunc

    def fit(self, x, y, step=0.5):
        # should it return error?
        tempX = [x]
        for layer in self.layers:
            x = layer.goThrough(x)
            tempX.append(x)
        buffer = self.errorFunc.callDerivative(x, y)
        for i in range(self.layers.shape[0], 0, -1):
            buffer *= self.layers[i - 1].function.derivativeFromAnswer(tempX[i])
            self.layers[i - 1].bias -= step * buffer
            # is the next part efficient enough?
            temp = self.layers[i - 1].weights.copy()
            toMult = np.vstack([tempX[i - 1]] * buffer.size)
            toMult *= buffer.reshape(buffer.size, 1)
            self.layers[i - 1].weights -= step * toMult
            buffer = buffer @ temp
        return self.errorFunc(x, y)
