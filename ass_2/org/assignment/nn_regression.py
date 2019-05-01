import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network.multilayer_perceptron import MLPRegressor
import matplotlib.pyplot as plt

from nn_regression_plot import plot_mse_vs_neurons, plot_mse_vs_iterations, plot_learned_function, \
    plot_mse_vs_alpha, plot_bars_early_stopping_mse_comparison

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
    n_neur = 5 # 2, 5, 50
    reg = MLPRegressor(hidden_layer_sizes=(n_neur,), max_iter=5000, activation='logistic', solver='lbfgs', alpha=0)

    reg.fit(x_train, y_train)
    y_pred_test = reg.predict(x_test)
    y_pred_train = reg.predict(x_train)
    
    plot_learned_function(n_neur, x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)




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

    MSE_TEST = np.zeros(10, dtype=float)
    MSE_TRAIN = np.zeros(10, dtype=float)

    n_neur = 5
    for j in range(10):
        seed = np.random.randint(100)
        reg = MLPRegressor(hidden_layer_sizes=(n_neur,), max_iter=5000, activation='logistic', solver='lbfgs', alpha=0,random_state=seed)
        reg.fit(x_train, y_train)
        mse = calculate_mse(reg, x_test, y_test)
        MSE_TEST[j] = mse
        
        mse = calculate_mse(reg, x_train, y_train)
        MSE_TRAIN[j] = mse

    mse_test_max = max(MSE_TEST)
    mse_test_min = min(MSE_TEST)
    mse_test_mean = np.mean(MSE_TEST)
    mse_test_std = np.std(MSE_TEST)

    print("#####  Radom Seeds  #####")
    
    print("### Test Set ###")
    print(" MAX  /  MIN  /  MEAN  /  STD")
    print(round(mse_test_max,3), "/", round(mse_test_min,3), "/", round(mse_test_mean,3), "/", round(mse_test_std,3))
    print("\n",MSE_TEST,"n")
    
    mse_train_max = max(MSE_TRAIN)
    mse_train_min = min(MSE_TRAIN)
    mse_train_mean = np.mean(MSE_TRAIN)
    mse_train_std = np.std(MSE_TRAIN)
    
    print("\n### Training Set ###")
    print(" MAX  /  MIN  /  MEAN  /  STD")
    print(round(mse_train_max,3), "/", round(mse_train_min,3), "/", round(mse_train_mean,3), "/", round(mse_train_std,3))
    print("\n",MSE_TRAIN,"n")


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
    n_seeds = 10
    n_neur = [1,2,3,4,6,8,12,20,40]
    mse_train = np.zeros([np.size(n_neur), n_seeds])
    mse_test = np.zeros([np.size(n_neur), n_seeds])

    for h in range(np.size(n_neur)):
        for s in range(n_seeds): 
            seed = np.random.randint(100)
            reg = MLPRegressor(hidden_layer_sizes=(n_neur[h],), max_iter=5000, activation='logistic', solver='lbfgs', alpha=0,random_state=seed)         
            
            reg.fit(x_train, y_train)            
            mse_train[h, s] = calculate_mse(reg, x_train, y_train)
            mse_test[h, s] = calculate_mse(reg, x_test, y_test)

    plot_mse_vs_neurons(mse_train, mse_test, n_neur)
    sum_mse = mse_test.sum(axis=1)
    ind_min=sum_mse.argmin()
    
    reg = MLPRegressor(hidden_layer_sizes=(n_neur[ind_min],), max_iter=5000, activation='logistic', solver='lbfgs', alpha=0 , random_state=np.random.randint(100))

    reg.fit(x_train, y_train)
    y_pred_test = reg.predict(x_test)
    y_pred_train = reg.predict(x_train)
    plot_learned_function(n_neur[ind_min], x_train, y_train, y_pred_train, x_test, y_test, y_pred_test)


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
    total_iter = 50
    n_neur = [2,5,50]
    solvers = ['lbfgs','sgd','adam']
    counter_solv = 0
    mse_train = np.zeros([np.size(n_neur), total_iter, np.size(solvers)])
    mse_test = np.zeros([np.size(n_neur), total_iter, np.size(solvers)])
    for solv in solvers:
        for j in range(np.size(n_neur)):
            reg = MLPRegressor(hidden_layer_sizes=(n_neur[j],), activation='logistic', solver=solv, alpha=0, random_state=0, warm_start=True, max_iter=1)
            for r in range(total_iter):
                reg.fit(x_train, y_train)            
                mse_train[j, r, counter_solv] = calculate_mse(reg, x_train, y_train)
                mse_test[j, r, counter_solv] = calculate_mse(reg, x_test, y_test)
        counter_solv += 1
            

    # PLOT
    for s in range(np.size(solvers)):
        plot_mse_vs_neurons(mse_train[:,:,s], mse_test[:,:,s], n_neur)

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
    total_iter = 5000
    n_neuro = 50
    n_seeds = 10
    alph = np.power(10, np.linspace(-8, 2, 11))
    mse_train = np.zeros([np.size(alph),n_seeds])
    mse_test = np.zeros([np.size(alph),n_seeds])
    
    for j in range(np.size(alph)):
        for s in range(n_seeds): 
            seed = np.random.randint(100)
            reg = MLPRegressor(hidden_layer_sizes=(n_neuro,), activation='logistic', solver='lbfgs', alpha=alph[j], random_state=seed,  max_iter=total_iter)
            reg.fit(x_train, y_train)            
            mse_train[j, s] = calculate_mse(reg, x_train, y_train)
            mse_test[j, s] = calculate_mse(reg, x_test, y_test)
            
    
    plot_mse_vs_alpha(mse_train, mse_test, alph)



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
    total_iter = 1000
    epoch_iter = 20
    epochs = int(total_iter/epoch_iter)
    n_neuro = 50
    n_seeds = 10
    
    sequence = np.random.permutation(np.arange(0, np.size(y_train), 1))
    x_train = x_train[sequence]
    y_train = y_train[sequence]
    SIZE = int(np.ceil(np.size(y_train) / 3))

    x_val = x_train[:SIZE]
    y_val = y_train[:SIZE]

    x_train = x_train[SIZE:]
    y_train = y_train[SIZE:]

    mse_train = np.zeros([n_seeds,epochs])
    mse_val = np.zeros([n_seeds,epochs])
    
    
    for s in range(n_seeds):
        seed = np.random.randint(100)
        reg = MLPRegressor(hidden_layer_sizes=(n_neuro,), activation='logistic', solver='lbfgs', alpha=0, random_state=seed,  warm_start=True, max_iter=epoch_iter)
        for ep in range(epochs):            
            reg.fit(x_train, y_train) 
            mse_train[s,ep] = calculate_mse(reg, x_train, y_train)
            mse_val[s,ep] = calculate_mse(reg, x_val, y_val)

    ## Last MSE Value of Test Set
    last_test_error = mse_train[:,-1]

    ## MSE Value where validation set has minimum
    min_val_error = np.amin(mse_val, axis=1)
    test_error_min_val_error = mse_train[mse_val==min_val_error]

    ## MSE Value where test set has minimum
    min_test_error = np.amin(mse_train, axis=1)
    
    # Plot            
    plot_bars_early_stopping_mse_comparison(last_test_error, test_error_min_val_error, min_test_error)


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
