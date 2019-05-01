from sklearn.metrics import confusion_matrix, mean_squared_error

from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from nn_classification_plot import plot_hidden_layer_weights, plot_histogram_of_acc
import numpy as np

__author__ = 'bellec,subramoney'

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""


def ex_2_1(input2, target2):
    """
    Solution for exercise 2.1
    :param input2: The input from dataset2
    :param target2: The target from dataset2
    :return:
    """
    ## TODO
    pose = target2[:,1]
    nn = MLPClassifier(hidden_layer_sizes=(6,) ,activation='tanh', max_iter=200)
    nn.fit(input2, pose) 
    y_pred = nn.predict(input2)
    C = confusion_matrix(pose, y_pred, labels=None, sample_weight=None)
    plot_hidden_layer_weights(nn.coefs_[0])
    return C


def ex_2_2(input1, target1, input2, target2):
    list = []
    train_acc = np.zeros(10)
    test_acc = np.zeros(10)
    for i in range(10):
        nn = MLPClassifier(hidden_layer_sizes=(20,),activation='tanh', max_iter=1000, random_state=None)
        list.append(nn)
        nn.fit(input1, target1[:,0]) 
        train_acc[i] = nn.score(input1, target1[:,0])
        test_acc[i] = nn.score(input2,target2[:,0])
        i_best = np.where(test_acc == test_acc.min())[0][0]
    import pdb
    pdb.set_trace() 
    y_pred = list[i_best].predict(input2)
    C = confusion_matrix(target2[:,0], y_pred, labels=None, sample_weight=None)
    """
    Solution for exercise 2.2
    :param input1: The input from dataset1
    :param target1: The target from dataset1
    :param input2: The input from dataset2
    :param target2: The target from dataset2
    :return:
    """
    ## TODO
    return train_acc, test_acc, y_pred, C
