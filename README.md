# NeuralNetworks
Creating a python library for implementing neural networks of a given architecture.

## How to set up

See [test.py](https://github.com/yanava99/NeuralNetworks/blob/master/test.py) and [mnist.py](https://github.com/yanava99/NeuralNetworks/blob/master/mnist.py) for example.

1. Import *NeuralNetworksLibrary.py* to your python application.
2. Prepare training and testing datasets. They need to be presented as numpy arrays, each input - a 1-dimensional array. Normalizing, shuffling and encoding data is on you.
3. Create a new model using Model class, specify dimensions of input and output data.
4. Create layers: specify the number of neurons, activation function (you can either choose one of the implemented functions or create your own).
If you want to specify the initial weights and bias of a layer - this is the time to do it. Use *setWeights()* and *setBias()* methods. Weights need to be a 2-dimensional numpy array, where the first dimension equals number of neurons on the layer itself and the second dimension equals number of neurons oh the previous layer (or length of input data in case of a first layer); bias needs to be a 1-dimensional numpy array with a length equal to number of neurons on the current layer.
5. Add layers to the model. If the number of neurons on the last layer doesn't equal to the length of output one more layer with corresponding dimensions and *Identity* activation function is added when creating a network.
6. Create a network based on a model, specify error function (choose either one of the implemented functions or create your own). Note, that *softmaxAndCrossEntropyBundleError* adds one more layer with a *softmax* activation function.
7. Time to train the network. Method *fit()* is changing network parameters corresponding to given input and output. Learning rate may be specified (the default value is 0.1). Note, that *fit()* needs to be called for each element of training set individually.
8. When a network is trained you can use *decide()* method to predict the output by given input.
