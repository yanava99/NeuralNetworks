import numpy as np
from tensorflow import keras
import NeuralNetworksLibrary as nnl
from sklearn.utils import shuffle
from time import clock
from math import ceil


def prepare_dataset():
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
    y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train.resize(x_train.shape[0], 784)
    x_test.resize(x_test.shape[0], 784)

    return x_train, y_train, x_test, y_test


def evaluate_my_network(x_train, y_train, x_test, y_test, step, batch_size, epochs=1):
    model = nnl.Model(x_train.shape[1], y_train.shape[1])

    model.addLayer(nnl.DenseLayer(10, nnl.logistic))
    network = nnl.Network(model, nnl.squaredError)

    print("NNL:")
    for e in range(epochs):
        for i in range(y_train.shape[0]):
            network.fit(x_train[i], y_train[i], step)
        # network.fitBatch(x_train, y_train, step=step, batchSize=batch_size)

        correct = 0
        for i in range(y_test.shape[0]):
            if np.argmax(network.decide(x_test[i])) == np.argmax(y_test[i]):
                correct += 1
        print('correct', correct / y_test.shape[0])


def evaluate_my_network_Batch(x_train, y_train, x_test, y_test, step, batch_size, epochs=1):
    model = nnl.Model(x_train.shape[1], y_train.shape[1])

    model.addLayer(nnl.DenseLayer(10, nnl.logistic))
    network = nnl.Network(model, nnl.squaredError)

    print("NNL, batch_size:", batch_size)
    for e in range(epochs):
        # for i in range(y_train.shape[0]):
        #     network.fit(x_train[i], y_train[i], step)
        network.fitBatch(x_train, y_train, step=step, batchSize=batch_size)

        correct = 0
        for i in range(y_test.shape[0]):
            if np.argmax(network.decide(x_test[i])) == np.argmax(y_test[i]):
                correct += 1
        print('correct', correct / y_test.shape[0])


def run_experiment(epochs=3):
    x_train, y_train, x_test, y_test = prepare_dataset()
    step = 0.001
    batch_size = 8
    beginNNL = clock()
    evaluate_my_network(np.copy(x_train), y_train, np.copy(x_test), y_test, step, batch_size, epochs)
    endNNL = clock()
    timeNNL = endNNL - beginNNL
    beginNNLBatch = clock()
    evaluate_my_network_Batch(np.copy(x_train), y_train, np.copy(x_test), y_test, step, batch_size, epochs)
    endNNLBatch = clock()
    timeNNLBatch = endNNLBatch - beginNNLBatch
    beginNNLBatch16 = clock()
    evaluate_my_network_Batch(np.copy(x_train), y_train, np.copy(x_test), y_test, step, 16, epochs)
    endNNLBatch16 = clock()
    timeNNLBatch16 = endNNLBatch16 - beginNNLBatch16
    beginNNLBatch32 = clock()
    evaluate_my_network_Batch(np.copy(x_train), y_train, np.copy(x_test), y_test, step, 32, epochs)
    endNNLBatch32 = clock()
    timeNNLBatch32 = endNNLBatch32 - beginNNLBatch32
    beginNNLBatch64 = clock()
    evaluate_my_network_Batch(np.copy(x_train), y_train, np.copy(x_test), y_test, step, 64, epochs)
    endNNLBatch64 = clock()
    timeNNLBatch64 = endNNLBatch64 - beginNNLBatch64
    beginNNLBatch128 = clock()
    evaluate_my_network_Batch(np.copy(x_train), y_train, np.copy(x_test), y_test, step, 128, epochs)
    endNNLBatch128 = clock()
    timeNNLBatch128 = endNNLBatch128 - beginNNLBatch128
    beginNNLBatch256 = clock()
    evaluate_my_network_Batch(np.copy(x_train), y_train, np.copy(x_test), y_test, step, 256, epochs)
    endNNLBatch256 = clock()
    timeNNLBatch256 = endNNLBatch256 - beginNNLBatch256

    print("Average time:")
    print("NNL:", timeNNL )
    print("NNLBatch, batch_size:", batch_size, "time:", timeNNLBatch )
    print("NNLBatch, batch_size:", 16, "time:", timeNNLBatch16 )
    print("NNLBatch, batch_size:", 32, "time:", timeNNLBatch32 )
    print("NNLBatch, batch_size:", 64, "time:", timeNNLBatch64 )
    print("NNLBatch, batch_size:", 128, "time:", timeNNLBatch128 )
    print("NNLBatch, batch_size:", 256, "time:", timeNNLBatch256 )


run_experiment()
