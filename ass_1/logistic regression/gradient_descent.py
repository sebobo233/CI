#!/usr/bin/env python
import numpy as np

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) and Adaptative gradient descent (GDad) (Logistic Regression)

This file contains generic implementation of gradient descent solvers
The functions are:
- TODO gradient_descent: for a given function with its gradient it finds the minimum with gradient descent
- TODO adaptative_gradient_descent: Same with adaptative learning rate
"""


def gradient_descent(f, df, theta0, learning_rate, max_iter):
    """
    Find the optimal solution of the function f(x) using gradient descent:
    Until the number of iteration is reached, decrease the parameter x by the gradient times the learning_rate.
    The function should return the minimal argument x and the list of errors at each iteration in a numpy array.

    :param f: function to minimize
    :param df: gradient of f
    :param theta0: initial point
    :param learning_rate:
    :param max_iter: maximal number of iterations
    :return: x (solution), E_list (array of errors over iterations)
    """
    ##############
    #
    # TODO
    #
    # Implement a gradient descent algorithm

    E_list = np.zeros(max_iter)
    theta = theta0
    
    for i in range(max_iter):
        theta -= learning_rate*df(theta)
        E_list[i] = f(theta)
        #Abbruch-bedingung ???        

    # END TODO
    ###########

    return theta, E_list


def adaptative_gradient_descent(f, df, theta0, initial_learning_rate, max_iter):
    """
    Find the optimal solution of the function f using an adaptative gradient descent:

    After every update check whether the cost increased or decreased.
        - If the cost increased, reject the update (go back to the
        previous parameter setting) and multiply the learning rate by 0.7.
        - If the cost decreased, accept the
        update and multiply the learning rate by 1.03.

    The iteration count should be increased after every iteration even if the update was rejected.

    :param f: function to minimize
    :param df: gradient of f
    :param theta0: initial point
    :param initial_learning_rate: initial learning rate
    :param max_iter: maximal number of iterations
    :return: x (solution), E_list (list of errors), lr_list (the list of learning rates)
    """

    ##############
    #
    # TODO
    #
    # Implement the adaptive gradient descent algorithm
    #

    E_list = np.empty(max_iter)
    E_list[:] = np.nan
    lr_list = np.empty(max_iter)
    theta = theta0
    old_theta = theta0
    adapt_rate = initial_learning_rate
    old_cost = f(theta)
    start_epoch = 5                      #first iteration for initial leraning_rate

    epoch_increase = start_epoch
    end_epoch = epoch_increase
    start_epoch = 0
    epoch_iter = 0
    

    while epoch_iter<(max_iter-1):
        for epoch_iter in range(start_epoch,end_epoch):
            theta -= adapt_rate*df(theta)
            E_list[epoch_iter] = f(theta)
            lr_list[epoch_iter] = adapt_rate
        if E_list[epoch_iter] <= old_cost:
            adapt_rate *= 1.03
            old_cost = E_list[epoch_iter]
            start_epoch = end_epoch
        else:
            adapt_rate *= 0.7
            theta = old_theta
        epoch_increase *= 2
        end_epoch += epoch_increase
        if end_epoch > max_iter:
            end_epoch = max_iter 
        

    # END TODO
    ###########

    return theta, E_list, lr_list
