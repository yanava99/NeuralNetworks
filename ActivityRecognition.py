# based on code from here:
# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

# cnn model
from numpy import mean
from numpy import std
from numpy import dstack
from numpy import argmax
from pandas import read_csv
from numpy import copy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.utils import to_categorical
from time import clock
from math import ceil
from sklearn.utils import shuffle

import NeuralNetworksLibrary as nnl


# load a single file as a numpy array
def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values


# load a list of files and return as a 3d numpy array
def load_group(filenames, prefix=''):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = dstack(loaded)
    return loaded


# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    return X, y


# load the dataset, returns train and test X and y elements
def load_dataset(prefix=''):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'HAR Dataset/')
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'HAR Dataset/')
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy


# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy, batch_size, step):
    verbose = 0
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=1, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    # model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    # model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='sigmoid'))
    sgd = SGD(learning_rate=step, momentum=0.0, nesterov=False)
    model.compile(optimizer=sgd,
                  # loss='mean_squared_error',
                  loss='categorical_crossentropy',
                  # loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=1, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy


def evaluate_my_nnl(trainX, trainy, testX, testy, batch_size, step):
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = nnl.Model(n_timesteps * n_features, n_outputs)
    model.addLayer(nnl.Conv1DLayer(1, 3, nnl.reLU, (n_timesteps - 2) * 1))
    # model.addLayer(nnl.Conv1DLayer(64, 3, nnl.reLU))
    poolSize = 2
    model.addLayer(nnl.MaxPooling1DLayer(poolSize, ceil((n_timesteps - 2) / poolSize) * 1))
    model.addLayer(nnl.FlattenLayer(ceil((n_timesteps - 2) * 1 / poolSize) * 1))
    # model.addLayer(nnl.DenseLayer(100, nnl.reLU))
    model.addLayer(nnl.DenseLayer(n_outputs, nnl.logistic))
    network = nnl.Network(model, nnl.crossEntropyError, channels=n_features)
    network.fitBatch(trainX, trainy, step, batch_size)
    # for i in range(trainy.shape[0]):
    #     network.fit(trainX[i, :, :], trainy[i, :], step)

    correct = 0
    for i in range(testy.shape[0]):
        if argmax(network.decide(testX[i])) == argmax(testy[i]):
            correct += 1
    return correct / testy.shape[0]


# summarize scores
def summarize_results(scores, time, repeats):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
    print('Average time:', time / repeats)


# run an experiment
def run_experiment(repeats=1):
    # load data
    trainX, trainy, testX, testy = load_dataset()
    trainX, trainy = shuffle(trainX, trainy)
    testX, testy = shuffle(testX, testy)
    step = 0.1
    batch_size = 8
    # repeat experiment
    scoresKeras = list()
    scoresNNL = list()
    timeKeras = 0
    timeNNL = 0
    for r in range(repeats):
        beginNNL = clock()
        scoreNNL = evaluate_my_nnl(copy(trainX), trainy, copy(testX), testy, batch_size, step)
        endNNL = clock()
        timeNNL += endNNL - beginNNL
        scoreNNL = scoreNNL * 100.0
        print('NNL: >#%d: %.3f' % (r+1, scoreNNL))
        scoresNNL.append(scoreNNL)
        beginKeras = clock()
        scoreKeras = evaluate_model(trainX, trainy, testX, testy, batch_size, step)
        endKeras = clock()
        timeKeras += endKeras - beginKeras
        scoreKeras = scoreKeras * 100.0
        print('Keras: >#%d: %.3f' % (r+1, scoreKeras))
        scoresKeras.append(scoreKeras)
    # summarize results
    print('Keras:')
    summarize_results(scoresKeras, timeKeras, repeats)
    print('My library:')
    summarize_results(scoresNNL, timeNNL, repeats)


# run the experiment
run_experiment()
