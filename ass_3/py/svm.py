import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
#from sklearn.metrics import confusion_matrix

from svm_plot import plot_svm_decision_boundary, plot_score_vs_degree, plot_score_vs_gamma, plot_mnist, \
    plot_confusion_matrix

"""
Computational Intelligence TU - Graz
Assignment 3: Support Vector Machine, Kernels & Multiclass classification
Part 1: SVM, Kernels

TODOS are all contained here.
"""

__author__ = 'bellec,subramoney'


def ex_1_a(x, y):
    """
    Solution for exercise 1 a)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Train an 
    ## and plot the decision boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########
    #pass
    
    svc = svm.SVC(kernel ='linear', C = 1).fit(x, y) 
    plot_svm_decision_boundary(svc,x,y)

def ex_1_b(x, y):
    """
    Solution for exercise 1 b)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel
    ## and plot the decisSVM with a linear kernelion boundary and support vectors using 'plot_svm_decision_boundary' function
    ###########
    #pass
    x = np.append(x, [[4, 0]], axis=0)
    y = np.append(y, 1)

    svc = svm.SVC(kernel ='linear', C = 1).fit(x, y) 
    plot_svm_decision_boundary(svc,x,y)


def ex_1_c(x, y):
    """
    Solution for exercise 1 c)
    :param x: The x values
    :param y: The y values
    :return:
    """
    ###########
    ## TODO:
    ## Add a point (4,0) with label 1 to the data set and then
    ## train an SVM with a linear kernel with different values of C
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    Cs = [1e6, 1, 0.1, 0.001]
    
    x = np.append(x, [[4, 0]], axis=0)
    y = np.append(y, 1)
    
    for j in range(np.size(Cs)):
        svc = svm.SVC(kernel ='linear', C = Cs[j]).fit(x, y) 
        plot_svm_decision_boundary(svc,x,y)



def ex_2_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train an SVM with a linear kernel for the given dataset
    ## and plot the decision boundary and support vectors  for each using 'plot_svm_decision_boundary' function
    ###########
    #pass
    svc = svm.SVC(kernel ='linear', C = 1).fit(x_train, y_train) 
    score_svc = svc.score(x_test, y_test)
    print("Score for linear kernel SVC:", score_svc)
    plot_svm_decision_boundary(svc, x_train, y_train, x_test, y_test)

def ex_2_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with polynomial kernels for different values of the degree
    ## (Remember to set the 'coef0' parameter to 1)
    ## and plot the variation of the test and training scores with polynomial degree using 'plot_score_vs_degree' func.
    ## Plot the decision boundary and support vectors for the best value of degree
    ## using 'plot_svm_decision_boundary' function
    ###########
    
    degrees = range(1, 20)

    train_scores = np.zeros(np.size(degrees))
    test_scores = np.zeros(np.size(degrees))                   

    for j in range(np.size(degrees)):
        svc = svm.SVC(kernel ='poly', C = 1, coef0 = 1, degree = degrees[j]).fit(x_train, y_train)
        test_scores[j] = svc.score(x_test, y_test)
        train_scores[j] = svc.score(x_train, y_train)
    
    plot_score_vs_degree(train_scores, test_scores, degrees)
    
    acc_max = test_scores.argmax()
    svc = svm.SVC(kernel ='poly', C = 1, coef0 = 1, degree = degrees[acc_max]).fit(x_train, y_train)
    plot_svm_decision_boundary(svc, x_train, y_train, x_test, y_test)

def ex_2_c(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 2 c)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train SVMs with RBF kernels for different values of the gamma
    ## and plot the variation of the test and training scores with gamma using 'plot_score_vs_gamma' function.
    ## Plot the decision boundary and support vectors for the best value of gamma
    ## using 'plot_svm_decision_boundary' function
    ###########
    gammas = np.arange(0.01, 2, 0.02)
    
    train_scores = np.zeros(np.size(gammas))
    test_scores = np.zeros(np.size(gammas))                   

    for j in range(np.size(gammas)):
        svc = svm.SVC(kernel ='rbf', gamma  = gammas[j]).fit(x_train, y_train)
        test_scores[j] = svc.score(x_test, y_test)
        train_scores[j] = svc.score(x_train, y_train)

    plot_score_vs_gamma(train_scores, test_scores, gammas)
    
    acc_max = test_scores.argmax()
    svc = svm.SVC(kernel ='rbf', gamma  = gammas[acc_max]).fit(x_train, y_train)
    plot_svm_decision_boundary(svc, x_train, y_train, x_test, y_test)

def ex_3_a(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 a)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with one-versus-rest strategy with
    ## - linear kernel
    ## - rbf kernel with gamma going from 10**-5 to 10**-3
    ## - plot the scores with varying gamma using the function plot_score_versus_gamma
    ## - Mind that the chance level is not .5 anymore and add the score obtained with the linear kernel as optional argument of this function
    ###########

    gammas = np.power(10, np.linspace(5, -5, 11, endpoint=True))
    
    train_scores2 = np.zeros(np.size(gammas))
    test_scores2 = np.zeros(np.size(gammas))     

    svc1 = svm.SVC(kernel ='linear', decision_function_shape='ovr', C = 10).fit(x_train, y_train)
    test_scores1 = svc1.score(x_test, y_test)
    train_scores1 = svc1.score(x_train, y_train)
    
    for j in range(np.size(gammas)):
        svc2 = svm.SVC(kernel ='rbf', decision_function_shape = 'ovr', gamma = gammas[j], C = 10).fit(x_train, y_train)
        test_scores2[j] = svc2.score(x_test, y_test)
        train_scores2[j] = svc2.score(x_train, y_train)

    plot_score_vs_gamma(train_scores2, test_scores2, gammas, train_scores1, test_scores1,0.2)
     
def ex_3_b(x_train, y_train, x_test, y_test):
    """
    Solution for exercise 3 b)
    :param x_train: Training samples (2-dimensional)
    :param y_train: Training labels
    :param x_test: Testing samples (2-dimensional)
    :param y_test: Testing labels
    :return:
    """
    ###########
    ## TODO:
    ## Train multi-class SVMs with a LINEAR kernel
    ## Use the sklearn.metrics.confusion_matrix to plot the confusion matrix.
    ## Find the index for which you get the highest error rate.
    ## Plot the confusion matrix with plot_confusion_matrix.
    ## Plot the first 10 occurrences of the most misclassified digit using plot_mnist.
    ###########

    labels = range(1, 6)

    svc_ovo = svm.SVC(kernel ='linear', decision_function_shape='ovo').fit(x_train, y_train)

    y_pred = svc_ovo.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, labels)


    cp = cm
    np.fill_diagonal(cp,0)
    i = np.argmax(np.max(cp, axis=0))  # should be the label number corresponding the largest classification error
    sel_err = np.argwhere(np.not_equal(y_test, y_pred))  # Numpy indices to select images that are misclassified.

    plot_mnist(x_test[sel_err], y_pred[sel_err], labels=labels[i], k_plots=10, prefix='Real class')
