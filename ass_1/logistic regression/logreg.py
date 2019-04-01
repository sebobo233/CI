#! python
import numpy as np

from logreg_toolbox import sig

__author__ = 'bellec, subramoney'

"""
Computational Intelligence TU - Graz
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) (Logistic Regression)
TODO Fill the cost function and the gradient
"""


def cost(theta, x, y):
    """
    Cost of the logistic regression function.

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: cost
    """
    #N, n = x.shape

    ##############
    #
    # TODO
    #
    # Write the cost of logistic regression as defined in the lecture
    # Hint:
    #   - use the logistic function sig imported from the file toolbox
    #   - prefer numpy vectorized operations over for loops
    # 
    # WARNING: If you run into instabilities during the exercise this
    #   could be due to the usage log(x) with x very close to 0. Some
    #   implementations are more or less sensible to this issue, you
    #   may try another one. A (dirty) trick is to replace log(x) with
    #   log(x + epsilon) with epsilon a very small number like 1e-20
    #   or 1e-10 but the gradients might not be exact anymore. This
    #   problem sometimes raises only when minizing the cost function
    #   with the scipy optimizer.
    m = x.shape[0]
    log_eps = 1e-20
    h_theta = sig(np.dot(x,theta))
    c = -(1/m)*(np.dot(y,np.log(h_theta+log_eps))+np.dot((1-y),np.log(1-h_theta+log_eps)))

    

    # END TODO
    ###########

    return c


def grad(theta, x, y):
    """

    Compute the gradient of the cost of logistic regression

    :param theta: parameter(s)
    :param x: sample(s)
    :param y: target(s)
    :return: gradient
    """
    #N, n = x.shape

    ##############
    #
    # TODO
    #
    #   - prefer numpy vectorized operations over for loops

    m = x.shape[0]
    h_theta = sig(np.dot(x,theta))
    g = (1/m)*(np.dot((h_theta-y),x))

    # END TODO
    ###########

    return g