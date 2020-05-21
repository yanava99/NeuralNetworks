import numpy as np
import copy
from math import ceil


# Class ActivationFunction has 3 methods: call function itself, call it's derivative
# and call derivative receiving result.
# It's going to be used when specifying activation functions for the layers.
class ActivationFunction:

    def __init__(self, function, derivative, derivativeFromAnswer, multivariate=False):
        self.multivariate = multivariate
        if multivariate:
            self.func = function
            self.derivative = derivative
            self.derivativeFromAnswer = derivativeFromAnswer
        else:
            self.func = np.vectorize(function)
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
        return self.function(x, y)

    def callDerivative(self, x, y):
        return self.derivative(x, y)


# Class DenseLayer has information about number of neurons, activation function, current weights and bias.
class DenseLayer:
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

    def goThroughBatch(self, x):
        x = x @ np.transpose(self.weights) + self.bias
        x = self.function(x)
        return x

    def updateWeights(self, grad, step):
        self.weights -= step * grad

    def updateBias(self, grad, step):
        self.bias -= step * grad

    def fit(self, gradient, step, tempXBefore, tempXAfter):
        if self.function.multivariate:
            gradient = gradient @ self.function.derivativeFromAnswer(tempXAfter)
        else:
            gradient = gradient * self.function.derivativeFromAnswer(tempXAfter)
        self.updateBias(gradient, step)
        gradWeights = gradient.reshape(gradient.shape[0], 1) @ tempXBefore.reshape(1, tempXBefore.shape[0])
        gradient = gradient @ self.weights
        self.updateWeights(gradWeights, step)
        return gradient

    def fitBatch(self, gradient, step, tempXBefore, tempXAfter):
        if self.function.multivariate:
            gradient = gradient @ self.function.derivativeFromAnswer(tempXAfter)  # Check this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        else:
            gradient = gradient * self.function.derivativeFromAnswer(tempXAfter)
        self.updateBias(np.average(gradient, axis=0), step)

        gradWeights = gradient.reshape(gradient.shape[0], gradient.shape[1], 1) @ \
                      tempXBefore.reshape(tempXBefore.shape[0], 1, tempXBefore.shape[1])
        gradient = gradient @ self.weights
        self.updateWeights(np.average(gradWeights, axis=0), step)
        return gradient

    def networkInit(self, prevN):
        if self.weights.size == 0:
            self.setDefaultWeights(prevN)
        if self.bias.size == 0:
            self.setDefaultBias()
        return self.n


# One-dimensional convolutional layer
class Conv1DLayer:
    def __init__(self, filters, kernelSize, function):
        self.filters = filters
        self.kernelSize = kernelSize
        self.function = function
        self.kernelWeights = np.array([[[]]])
        self.bias = np.array([])

    def setKernelWeights(self, array):
        self.kernelWeights = array.copy()

    def setBias(self, array):
        self.bias = array.copy()

    def setDefaultKernelWeights(self, channels):
        self.kernelWeights = np.full((self.filters, self.kernelSize, channels), 1 / (channels * self.kernelSize))

    def setDefaultBias(self):
        self.bias = np.zeros(self.filters)

    # # padding is valid
    # def goThrough(self, x):  # x is a 2-dimensional array (dim * channels)
    #     output = np.zeros((x.shape[0] - self.kernelSize + 1, self.filters))
    #     for i in range(self.filters):
    #         for j in range(x.shape[0] - self.kernelSize + 1):
    #             output[j, i] = np.sum(self.kernelWeights[i, :, :] * x[j: j + self.kernelSize, :]) + self.bias[i]
    #     x = self.function(output)
    #     return x

    # padding is valid
    def goThrough(self, x):  # x is a 2-dimensional array (dim * channels)
        output = np.zeros((x.shape[0] - self.kernelSize + 1, self.filters))
        x_broaden = np.broadcast_to(x, (self.filters, x.shape[0], x.shape[1]))
        for j in range(x.shape[0] - self.kernelSize + 1):
            output[j, :] = np.sum(x_broaden[:, j: j + self.kernelSize, :] * self.kernelWeights, axis=(1, 2)) + self.bias
        x = self.function(output)
        return x

    # padding is valid
    def goThroughBatch(self, x):  # x is a 3-dimensional array (batchSize * dim * channels)
        output = np.zeros((x.shape[0], x.shape[1] - self.kernelSize + 1, self.filters))
        # x_broaden = np.broadcast_to(x, (x.shape[0], self.filters, x.shape[1], x.shape[2]))
        x_broaden = np.repeat(x[:, np.newaxis, :, :], self.filters, 1)
        w_broaden = np.broadcast_to(self.kernelWeights, (x.shape[0],
                                                         self.kernelWeights.shape[0],
                                                         self.kernelWeights.shape[1],
                                                         self.kernelWeights.shape[2],
                                                         ))
        for j in range(x.shape[1] - self.kernelSize + 1):
            output[:, j, :] = np.sum(x_broaden[:, :, j: j + self.kernelSize, :] * w_broaden, axis=(2, 3)) + \
                              np.broadcast_to(self.bias, (x.shape[0], self.filters))
        x = self.function(output)
        return x

    # # padding is valid
    # def goThroughBatch(self, x):  # x is a 3-dimensional array (batchSize * dim * channels)
    #     output = np.zeros((x.shape[0], x.shape[1] - self.kernelSize + 1, self.filters))
    #     for i in range(self.filters):
    #         for j in range(x.shape[1] - self.kernelSize + 1):
    #             output[:, j, i] = np.add(np.sum(np.multiply(x[:, j: j + self.kernelSize, :],
    #                                                         self.kernelWeights[i, :, :]),
    #                                             axis=(1,2)),
    #                                      self.bias[i]).reshape(x.shape[0])
    #     x = self.function(output)
    #     return x

    def updateKernelWeights(self, grad, step):
        self.kernelWeights -= step * grad

    def updateBias(self, grad, step):
        self.bias -= step * grad

    def fit(self, gradient, step, tempXBefore, tempXAfter):
        channels = tempXBefore.shape[1]
        if self.function.multivariate:
            gradient = gradient @ self.function.derivativeFromAnswer(tempXAfter)  # Check this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        else:
            gradient = gradient * self.function.derivativeFromAnswer(tempXAfter)
        self.updateBias(np.sum(gradient, axis=0), step)
        gradKernelWeights = np.zeros((self.filters, self.kernelSize, channels))
        # gradient_broaden = np.broadcast_to(np.transpose(gradient), (self.filters, self.kernelSize..., channels))
        gradient_broaden = np.repeat(np.transpose(gradient)[:, :, np.newaxis], channels, 2)
        tempXBefore_broaden = np.broadcast_to(tempXBefore, (self.filters, tempXBefore.shape[0], channels))
        for j in range(self.kernelSize):
            gradKernelWeights[:, j, :] = np.sum(gradient_broaden *
                                                tempXBefore_broaden[:,
                                                j: j + tempXBefore.shape[0] - self.kernelSize + 1, :],
                                                axis=1)
        # for i in range(self.filters):
        #     for a in range(self.kernelSize):
        #         for ca in range(channels):
        #             gradKernelWeights[i, a, ca] = np.sum(gradient[:, i] * tempXBefore[a: a + tempXAfter.shape[0], ca])
        newGradient = np.zeros(tempXBefore.shape)
        gradient_broaden = np.pad(gradient_broaden,
                                  ((0, 0), (self.kernelSize - 1, self.kernelSize - 1), (0, 0)),
                                  'constant')
        for j in range(tempXBefore.shape[0]):
            newGradient[j, :] = np.sum(gradient_broaden[:, j: j + self.kernelSize, :] * self.kernelWeights, axis=(0, 1))
    #     for ca in range(channels):
    #         for j in range(tempXAfter.shape[0]):
    #             newGradient[j, ca] = np.sum(np.pad(gradient,
    #                                                ((self.kernelSize - 1, 0), (0, 0)),
    #                                                'constant',
    #                                                constant_values=(0, 0))[j: j + self.kernelSize, :] *
    #                                         np.transpose(self.kernelWeights[:, :, ca]))
        gradient = newGradient
        self.updateKernelWeights(gradKernelWeights, step)
        return newGradient

    def fitBatch(self, gradient, step, tempXBefore, tempXAfter):
        channels = tempXBefore.shape[2]
        if self.function.multivariate:
            gradient = gradient @ self.function.derivativeFromAnswer(tempXAfter)  # Check this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        else:
            gradient = gradient * self.function.derivativeFromAnswer(tempXAfter)
        self.updateBias(np.average(np.sum(gradient, axis=1), axis=0), step)
        gradKernelWeights = np.zeros((self.filters, self.kernelSize, channels))
        # gradient_broaden = np.broadcast_to(np.transpose(gradient),
        #                                    (tempXAfter.dim[0], self.filters, self.kernelSize..., channels))
        gradient_broaden = np.repeat(np.transpose(gradient, (0, 2, 1))[:, :, :, np.newaxis], channels, 3)
        # tempXBefore_broaden = np.broadcast_to(tempXBefore, (self.filters, tempXBefore.shape[0], channels))
        tempXBefore_broaden = np.repeat(tempXBefore[:, np.newaxis, :, :], self.filters, 1)
        for j in range(self.kernelSize):
            gradKernelWeights[:, j, :] = np.average(np.sum(gradient_broaden *
                                                           tempXBefore_broaden[:, :,
                                                           j: j + tempXBefore.shape[1] - self.kernelSize + 1, :],
                                                           axis=2), axis=0)
        newGradient = np.zeros(tempXBefore.shape)
        gradient_broaden = np.pad(gradient_broaden,
                                  ((0, 0), (0, 0), (self.kernelSize - 1, self.kernelSize - 1), (0, 0)),
                                  'constant')
        w_broaden = np.broadcast_to(self.kernelWeights, (tempXAfter.shape[0], self.filters, self.kernelSize, channels))
        for j in range(tempXBefore.shape[1]):
            newGradient[:, j, :] = np.sum(gradient_broaden[:, :, j: j + self.kernelSize, :] * w_broaden, axis=(1, 2))
        gradient = newGradient
        self.updateKernelWeights(gradKernelWeights, step)
        return newGradient

    # def fitBatch(self, gradient, step, tempXBefore, tempXAfter):
    #     channels = tempXBefore.shape[2]
    #     if self.function.multivariate:
    #         gradient = gradient @ self.function.derivativeFromAnswer(tempXAfter)  # Check this !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     else:
    #         gradient = gradient * self.function.derivativeFromAnswer(tempXAfter)
    #     self.updateBias(np.average(gradient.sum(axis=1), axis=0), step)
    #     gradKernelWeights = np.zeros((self.filters, self.kernelSize, channels))
    #     for i in range(self.filters):
    #         for a in range(self.kernelSize):
    #             for ca in range(channels):
    #                 # gradKernelWeights[i, a, ca] = np.sum(gradient[:, i] * tempXBefore[a: a + tempXAfter.shape[0], ca])
    #                 gradKernelWeights[i, a, ca] = np.average(np.sum(gradient[:, :, i] *
    #                                                                 tempXBefore[:, a: a + tempXAfter.shape[1], ca],
    #                                                                 axis=1))
    #     newGradient = np.zeros(tempXBefore.shape)
    #     for ca in range(channels):
    #         for j in range(tempXAfter.shape[1]):
    #             newGradient[:, j, ca] = np.sum(np.matmul(
    #                 np.pad(gradient,
    #                        ((0, 0), (self.kernelSize - 1, 0), (0, 0)),
    #                        'constant')
    #                 [:, j: j + self.kernelSize, :],
    #                 self.kernelWeights[:, :, ca]))
    #     gradient = newGradient
    #     self.updateKernelWeights(gradKernelWeights, step)
    #     return newGradient

    def networkInit(self, prevShape):
        if self.kernelWeights.size == 0:
            self.setDefaultKernelWeights(prevShape[1])
        if self.bias.size == 0:
            self.setDefaultBias()
        return prevShape[0] - self.kernelSize + 1, self.filters


class MaxPooling1DLayer:
    def __init__(self, poolSize, strides=None):
        self.poolSize = poolSize
        if strides:
            self.strides = strides
        else:
            self.strides = poolSize
        self.inputShape = (0, 0)
        self.maxIndexes = np.asarray([])
        self.n = -1

    def goThrough(self, x):
        self.inputShape = x.shape
        newSize = self.n // x.shape[1]
        newX = np.zeros((newSize, x.shape[1]))
        self.maxIndexes = np.zeros((newSize, x.shape[1]), dtype=int)
        for i in range(newSize):
            self.maxIndexes[i] = np.argmax(x[i * self.poolSize: (i + 1) * self.poolSize, :], axis=0)
            for ch in range(x.shape[1]):
                newX[i, ch] = x[i * self.poolSize + self.maxIndexes[i, ch], ch]
        return newX

    def goThroughBatch(self, x):
        self.inputShape = x.shape
        newSize = self.n // x.shape[2]
        newX = np.zeros((x.shape[0], newSize, x.shape[2]))
        self.maxIndexes = np.zeros((x.shape[0], newSize, x.shape[2]), dtype=int)
        for i in range(newSize):
            self.maxIndexes[:, i] = np.argmax(x[:, i * self.poolSize: (i + 1) * self.poolSize, :], axis=1)
            for k in range(x.shape[0]):
                for ch in range(x.shape[2]):
                    newX[k, i, ch] = x[k, i * self.poolSize + self.maxIndexes[k, i, ch], ch]
        return newX

    def fit(self, gradient, step, tempXBefore, tempXAfter):
        newGradient = np.zeros(self.inputShape)
        for i in range(self.maxIndexes.shape[0]):
            for ch in range(self.inputShape[1]):
                newGradient[i * self.strides + self.maxIndexes[i][ch], ch] = gradient[i, ch]
        gradient = newGradient
        return gradient

    def fitBatch(self, gradient, step, tempXBefore, tempXAfter):
        newGradient = np.zeros(self.inputShape)
        for i in range(self.maxIndexes.shape[1]):
            for k in range(self.inputShape[0]):
                for ch in range(self.inputShape[2]):
                    newGradient[k, i * self.strides + self.maxIndexes[k, i, ch], ch] = gradient[k, i, ch]
        gradient = newGradient
        return gradient

    def networkInit(self, prevShape):
        self.n = ceil(prevShape[0] / self.strides) * prevShape[1]
        return self.n // prevShape[1], prevShape[1]


class FlattenLayer:
    def __init__(self):
        self.xShape = -1

    def goThrough(self, x):
        self.xShape = x.shape
        x = x.flatten()
        return x

    def goThroughBatch(self, x):
        self.xShape = x.shape
        x = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))
        return x

    def fit(self, gradient, step, tempXBefore, tempXAfter):
        gradient.resize(self.xShape)
        return gradient

    def fitBatch(self, gradient, step, tempXBefore, tempXAfter):
        gradient.resize(self.xShape)
        return gradient

    def updateParameters(self, step, batchSize):
        return

    def networkInit(self, prevShape):
        return prevShape[0] * prevShape[1]


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
        # for i in range(len(self.layers)):
        #     if isinstance(self.layers[i], Conv1DLayer):
        #         if i == 0:
        #             self.layers[i].n = (self.dimX // channels - self.layers[i].kernelSize + 1) * \
        #                                self.layers[i].filters  # padding == valid # ???????????????????????????????????????????????????????
        #         else:
        #             self.layers[i].n = (self.layers[i - 1].n // channels - self.layers[i].kernelSize + 1) * \
        #                                self.layers[i].filters  # padding == valid # ???????????????????????????????????????????????????????
        #     elif isinstance(self.layers[i], FlattenLayer):
        #         if i == 0:
        #             self.layers[i].n = self.dimX  # ???????????????????????????????????????????????????????????????????????????
        #         else:
        #             self.layers[i].n = self.layers[i - 1].n  # ????????????????????????????????????????????????????????????????????
        #     elif isinstance(self.layers[i], MaxPooling1DLayer):
        #         if i == 0:
        #             self.layers[i].n = ceil(self.dimX // channels / self.layers[i].strides) * channels
        #         else:
        #             self.layers[i].n = ceil(self.layers[i - 1].n // channels / self.layers[i].strides) * channels
        if len(self.layers) == 0:
            newLayer = DenseLayer(self.dimY, identity)
            newLayer.networkInit(self.dimX)
            self.layers.append(newLayer)
        else:
            temp = self.layers[0].networkInit(self.dimX)
            for i in range(1, len(self.layers)):
                temp = self.layers[i].networkInit(temp)
        # for i in range(len(self.layers)):
        #     if isinstance(self.layers[i], Conv1DLayer):
        #         if self.layers[i].kernelWeights.size == 0:
        #             if i == 0 or not isinstance(self.layers[i - 1], Conv1DLayer):  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #                 self.layers[i].setDefaultKernelWeights(channels)
        #             else:
        #                 self.layers[i].setDefaultKernelWeights(self.layers[i - 1].filters)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #         if self.layers[i].bias.size == 0:
        #             self.layers[i].setDefaultBias()
        #     elif isinstance(self.layers[i], DenseLayer):
        #         if self.layers[i].weights.size == 0:
        #             if i == 0:
        #                 self.layers[i].setDefaultWeights(self.dimX)
        #             else:
        #                 self.layers[i].setDefaultWeights(self.layers[i - 1].n)
        #         if self.layers[i].bias.size == 0:
        #             self.layers[i].setDefaultBias()
        self.errorFunc = errorFunc

    # Method for training the network.
    def fit(self, x, y, step=0.1):
        tempX = [x]  # saving outputs of each neuron
        for layer in self.layers:
            x = layer.goThrough(x)
            tempX.append(x)
        gradient = self.errorFunc.callDerivative(x, y)  # accumulation of derivatives
        for i in reversed(range(len(self.layers))):
            gradient = self.layers[i].fit(gradient, step, tempX[i], tempX[i + 1])
        return self.errorFunc(x, y)

    def fitBatch(self, xSet, ySet, step=0.1, batchSize=128):
        batchNumber = xSet.shape[0] // batchSize
        tempX = []
        for batch in range(batchNumber):
            x = xSet[batch * batchSize: (batch + 1) * batchSize]
            y = ySet[batch * batchSize: (batch + 1) * batchSize]
            tempX.clear()
            tempX.append(x)  # saving outputs of each neuron
            for layer in self.layers:
                x = layer.goThroughBatch(x)  # Check if it works with multivariate activation function !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                tempX.append(x)
            gradient = self.errorFunc.callDerivative(x, y)  # accumulation of derivatives
            for i in reversed(range(len(self.layers))):
                gradient = self.layers[i].fitBatch(gradient, step, tempX[i], tempX[i + 1])
        # residueBatchSize = xSet.shape[0] - (batchNumber * batchSize)
        if xSet.shape[0] - (batchNumber * batchSize) > 0:
            x = xSet[batchNumber * batchSize:]
            y = ySet[batchNumber * batchSize:]
            tempX.clear()
            tempX.append(x)  # saving outputs of each neuron
            for layer in self.layers:
                x = layer.goThroughBatch(x)  # Check if it works with multivariate activation function !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                tempX.append(x)
            gradient = self.errorFunc.callDerivative(x, y)  # accumulation of derivatives
            for i in reversed(range(len(self.layers))):
                gradient = self.layers[i].fitBatch(gradient, step, tempX[i], tempX[i + 1])

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
        result += (y[i] - x[i]) ** 2
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
