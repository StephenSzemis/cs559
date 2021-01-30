# Author: Stephen Szemis
# Pledge: I pledge my honor that I have abided by the Stevens honor system. - Stephen Szemis
# Date: December, 9, 2020

# Note: This code is based of off the code we saw and discussed in class.
# See link: https://dev.to/shamdasani/build-a-flexible-neural-network-with-backpropagation-in-python

import numpy as np
import matplotlib.pyplot as plt
import random as rand

# Used for our final question
from keras import models
from keras import layers
from keras.utils import to_categorical

def split_data(X, Y, test_percent=0.3):
    testX = []
    trainX = []
    testY = []
    trainY = []
    for i, y in enumerate(Y[0]):
        num = rand.random()
        if (num < test_percent):
            testX.append(X[i])
            testY.append(y)
        else:
            trainX.append(X[i])
            trainY.append(y)
    return np.array(testX), np.atleast_2d(np.array(testY)), np.array(trainX), np.atleast_2d(np.array(trainY))

# Grab our iris data
def get_data():
    X = []
    Y = []
    f = open('iris.data')
    for line in f:
        z = line.strip().split(',')
        temp = [float(x) for x in z[:-1]]
        if (temp != []):
            X.append(temp)
            if (z[-1] == 'Iris-virginica'):
                Y.append(1)
            elif (z[-1] == 'Iris-versicolor'):
                Y.append(0.5)
            else:
                Y.append(0)
    return np.array(X), np.atleast_2d(np.array(Y))

class Neural_Network(object):
    def __init__(self):
        # parameters
        self.inputSize = 4
        self.outputSize = 1
        self.hiddenSize = 6

        # weights
        # (4x6) weight matrix from input to hidden layer
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        # (6x1) weight matrix from hidden to output layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        # forward propagation through our network
        # dot product of X (input) and first set of 3x2 weights
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)  # activation function
        # dot product of hidden layer (z2) and second set of 3x1 weights
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3)  # final activation function
        return o

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        # backward propgate through the network
        
        self.o_error = np.subtract(y.T, o)  # error in output

        # applying derivative of sigmoid to error
        self.o_delta = self.o_error*self.sigmoidPrime(o)

        # z2 error: how much our hidden layer weights contributed to output error
        self.z2_error = self.o_delta.dot(self.W2.T)
        # applying derivative of sigmoid to z2 error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)

        # adjusting first set (input --> hidden) weights
        self.W1 += X.T.dot(self.z2_delta)
        # adjusting second set (hidden --> output) weights
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)


def accuracy(o, Y):
    acc = 0
    for i, sample in enumerate(o):
        if (sample < 0.33) and (Y[0][i] == 0):
            acc += 1
        elif (sample > 0.66) and (Y[0][i] == 1):
            acc += 1
        elif (sample < 0.66) and (sample > 0.33 ) and (Y[0][i] == 0.5):
            acc += 1
    return (acc / len(o)) * 100

def run_NN():
    X, Y = get_data()
    X = X/np.amax(X, axis=0)  # maximum of X array
    testX, testY, trainX, trainY = split_data(X, Y)

    NN = Neural_Network()
    loss = []
    test_accuracy = []
    train_accuracy = []
    # Test and Train loop
    for i in range(1000):
        # mean sum squared loss
        train_accuracy.append(accuracy(NN.forward(trainX), trainY))
        test_accuracy.append(accuracy(NN.forward(testX), testY))
        NN.train(trainX, trainY)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(range(len(train_accuracy)), train_accuracy)

    # Produce Graph
    ax.set_title("Plot hw3 Iteration versus Train Accuracy")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Train Accuracy')
    fig.savefig('hw3_train.png')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(range(len(test_accuracy)), test_accuracy)

    # Produce Graph
    ax.set_title("Plot hw3 Iteration versus Test Accuracy")
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Test Accuracy')
    fig.savefig('hw3_test.png')

def run_keras_model():
    X, Y = get_data()
    X = X/np.amax(X, axis=0)  # maximum of X array
    testX, testY, trainX, trainY = split_data(X, Y)

    network = models.Sequential()
    network.add(layers.Dense(32, activation='relu', input_shape=(4,)))
    network.add(layers.Dense(16, activation='relu'))
    # network.add(layers.Dense(16, activation='relu'))
    network.add(layers.Dense(3, activation='softmax'))

    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # Transform our output into binary vectors
    train_labels = to_categorical(trainY[0], num_classes=3)
    test_labels = to_categorical(testY[0], num_classes=3)

    network.fit(trainX, train_labels, epochs=50)
    test_loss, test_acc = network.evaluate(testX, test_labels)
    print('test_acc:', test_acc)
    print('test_loss:', test_loss)

run_keras_model()