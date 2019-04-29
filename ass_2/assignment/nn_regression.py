import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt

from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, plot_learned_function, \
    plot_mse_vs_alpha, plot_bars_early_stopping_mse_comparison
    
import random

"""
Computational Intelligence TU - Graz
Assignment 2: Neural networks
Part 1: Regression with neural networks

This file contains functions to train and test the neural networks corresponding the the questions in the assignment,
as mentioned in comments in the functions.
Fill in all the sections containing TODO!
"""

__author__ = 'bellec,subramoney'


def calculate_mse(nn, x, y):
    """
    Calculate the mean squared error on the training and test data given the NN model used.
    :param nn: An instance of MLPRegressor or MLPClassifier that has already been trained using fit
    :param x: The data
    :param y: The targets
    :return: Training MSE, Testing MSE
    """
    ## TODO
    #mse = 0
    
    mse = mean_squared_error(y, nn.predict(x)) 
    return mse


def ex_1_1_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 a)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    #pass
    n_hidden = 5 # 2, 5, 50
    reg = MLPRegressor(hidden_layer_sizes=(n_hidden,8), activation='logistic', solver='lbfgs', alpha=0)

    
    reg.fit(x_train, y_train)
    y_pred_test = reg.predict(x_test)
    y_pred_train = reg.predict(x_train)
    
    plot_learned_function(n_hidden, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)


def ex_1_1_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    
    ## TODO
    #pass
    
    MSE_TEST = np.zeros(10, dtype=float)
    MSE_TRAIN = np.zeros(10, dtype=float)

    n_hidden = 5
    for j in range(10):
        reg = MLPRegressor(hidden_layer_sizes=(n_hidden,8), activation='logistic', solver='lbfgs', alpha=0,random_state=random.randint(0, 1000))
        reg.fit(x_train, y_train)
        mse = calculate_mse(reg, x_test, y_test)
        MSE_TEST[j] = mse
        
        mse = calculate_mse(reg, x_train, y_train)
        MSE_TRAIN[j] = mse
        
    mse_test_max = max(MSE_TEST)
    mse_test_min = min(MSE_TEST)
    mse_test_mean = np.mean(MSE_TEST)
    mse_test_std = np.std(MSE_TEST)
    
    print(mse_test_max, "/", mse_test_min, "/", mse_test_mean, "/", mse_test_std)
    
    mse_train_max = max(MSE_TRAIN)
    mse_train_min = min(MSE_TRAIN)
    mse_train_mean = np.mean(MSE_TRAIN)
    mse_train_std = np.std(MSE_TRAIN)
    
    print(mse_train_max, "/", mse_train_min, "/", mse_train_mean, "/", mse_train_std)



def ex_1_1_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 c)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    #pass
    N = 10
    n_hidden = [1,2,3,4,6,8,12,20,40]
    mse_train = np.zeros([np.size(n_hidden), N])
    mse_test = np.zeros([np.size(n_hidden), N])
      
    for j in range(np.size(n_hidden)):
        reg = MLPRegressor(hidden_layer_sizes=(1,n_hidden[j]), activation='logistic', solver='lbfgs', alpha=0,random_state=random.randint(0, 1000))
        for r in range(N):   
            
            reg.fit(x_train, y_train)
            
            mse_train[j, r] = calculate_mse(reg, x_train, y_train)
            mse_test[j, r] = calculate_mse(reg, x_test, y_test)
        
    # PLOT
    plot_mse_vs_neurons(mse_train, mse_test, n_hidden)
    """
    mse_test_mean = np.mean(mse_test, axis=1) 
    ind = np.argmin(mse_test_mean)
    """
    ind = np.unravel_index(np.argmin(mse_test), mse_test.shape)
    
    reg = MLPRegressor(hidden_layer_sizes=(n_hidden[ind[0]],), activation='logistic', solver='lbfgs', alpha=0,
                               random_state=random.randint(0, 1000))

    reg.fit(x_train, y_train)
    y_pred_test = reg.predict(x_test)
    y_pred_train = reg.predict(x_train)
    plot_learned_function(n_hidden[ind[0]], x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)

def ex_1_1_d(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.1 b)
    Remember to set alpha to 0 when initializing the model
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """

    ## TODO
    #pass
    N = 500
    n_hidden = [2,5,50]
    mse_train = np.zeros([np.size(n_hidden), N])
    mse_test = np.zeros([np.size(n_hidden), N])
      
    for j in range(np.size(n_hidden)):
        reg = MLPRegressor(hidden_layer_sizes=(n_hidden[j],), activation='logistic', solver='lbfgs', alpha=0,random_state=0, warm_start=True, max_iter=1)
        for r in range(N):
            reg.fit(x_train, y_train)
            
            mse_train[j, r] = calculate_mse(reg, x_train, y_train)
            mse_test[j, r] = calculate_mse(reg, x_test, y_test)
            

    # PLOT
    plot_mse_vs_neurons(mse_train, mse_test, n_hidden)
    #mse_test_mean = np.mean(mse_test, axis=1) 
    #ind = np.argmin(mse_test_mean)
    
    ind = np.unravel_index(np.argmin(mse_test), mse_test.shape)
    # geht es auch ohne den MLPRegressor nochmal zu initialisieren? Keine Ahnung obs anders besser geht
    reg = MLPRegressor(hidden_layer_sizes=(n_hidden[j],), activation='logistic', solver='lbfgs', alpha=0,
                               random_state=random.randint(0, 1000), max_iter=500)
    reg.fit(x_train, y_train)
   
    y_pred_test = reg.predict(x_test)
    y_pred_train = reg.predict(x_train)
    
    plot_learned_function(n_hidden[ind[0]], x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)

def ex_1_2_a(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 a)
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    #pass
    N = 5000
    n_hidden = 50
    alpha_ = np.power(10, np.linspace(-8, 2, 11))
    mse_train = np.zeros([np.size(alpha_), N])
    mse_test = np.zeros([np.size(alpha_), N])
    
    for j in range(np.size(alpha_)):
        reg = MLPRegressor(hidden_layer_sizes=(n_hidden,), activation='logistic', solver='lbfgs', alpha=alpha_[j],
                               random_state=0, warm_start=True, max_iter=1)
        for r in range(N):
            reg.fit(x_train, y_train)
            
            mse_train[j, r] = calculate_mse(reg, x_train, y_train)
            mse_test[j, r] = calculate_mse(reg, x_test, y_test)
            
    
    plot_mse_vs_alpha(mse_train, mse_test, alpha_)


def ex_1_2_b(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 b)
    :param x_train: The training dataset
    :param x_test: The testing dataset
    :param y_train: The training targets
    :param y_test: The testing targets
    :return:
    """
    ## TODO
    #pass
    Iterations = 100
    N = 10
    n_hidden = 50
    
    sequence = np.random.permutation(np.arange(0, np.size(y_train), 1))
    x_train = x_train[sequence]
    y_train = y_train[sequence]
    SIZE = int(np.ceil(np.size(y_train) / 3))

    x_val = x_train[:SIZE]
    y_val = y_train[:SIZE]

    x_train = x_train[SIZE:]
    y_train = y_train[SIZE:]


    mse_train = np.zeros(N)
    mse_test = np.zeros(N)
    
    
    for r in range(N):
        reg = MLPRegressor(hidden_layer_sizes=(n_hidden,), activation='logistic', solver='lbfgs', alpha=0,random_state=0, warm_start=True, max_iter=1)
        for iter in range(Iterations):
            reg.fit(x_train, y_train)
 
            mse_train[r] = calculate_mse(reg, x_train, y_train)
            mse_test[r] = calculate_mse(reg, x_test, y_test)
        
            if np.mod(r, 20) == 0:
                print('Troll')
 
                
    #plot_bars_early_stopping_mse_comparison(test_mse_end, test_mse_early_stopping, test_mse_ideal)

def ex_1_2_c(x_train, x_test, y_train, y_test):
    """
    Solution for exercise 1.2 c)
    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    ## TODO
    pass
