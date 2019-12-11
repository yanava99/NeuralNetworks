import numpy as np
from tensorflow import keras
import NeuralNetworksLibrary as nnl
from sklearn.utils import shuffle


mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)
y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train.resize(x_train.shape[0], 784)
x_test.resize(x_train.shape[0], 784)

model = nnl.Model(784, 10)
# layer1 = nnl.Layer(128, nnl.logistic)
# model.addLayer(layer1)
# layer2 = nnl.Layer(10, nnl.logistic)
# model.addLayer(layer2)
# layer3 = nnl.Layer(10, nnl.logistic)
# model.addLayer(layer3)

# network = nnl.Network(model, nnl.squaredError)
# network = nnl.Network(model, nnl.crossEntropyError)
network = nnl.Network(model, nnl.softmaxAndCrossEntropyBundleError)

for epoch in range(1):
    print('epoch', epoch)
    for i in range(y_train.shape[0]):
        network.fit(x_train[i], y_train[i], 0.005)
    correct = 0
    for i in range(y_test.shape[0]):
        if np.argmax(network.decide(x_test[i])) == np.argmax(y_test[i]):
            correct += 1
    print('correct', correct / y_test.shape[0])
