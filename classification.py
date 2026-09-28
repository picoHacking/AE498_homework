"""This module contains functions to fit classification models for HW 2 in AE 498 Computational Systems Engineering.
Homework submission by Amanda Yaklin
yaklin2@illinois.edu
"""

import numpy as np


def fit_knn(X, y, n_neighbors):
    """Function to fit a KNN classifier to a given dataset.

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor.
    y : array_like
        Output/response data (n elements), where each element is an observation.
    n_neighbors : int
        Number of k neighbors to use for each prediction.

    Returns
    -------
    y_predicted : array_like
        Predicted responses for input data (n elements).
    error : float
        Error rate for the input data, calculated as 1/n_obs * number of incorrect predictions.

    Notes
    -----
    The test functions assume the k nearest neighbors of observation x include x.

    """
    n = X.shape[0]
    y_predicted = np.array([])
    # for each observation:
    for x in X:
        distance = np.array([]) 
        # measure distance from observation x to all other sonar signals, including observation x
        for o in X:
            distance = np.append(distance,np.sum((o-x)**2)) # sum of squares as a measure of 'nearness' of neighbors
        # identify the indices of the k smallest distances
        nearest_ids = np.argsort(distance)[:n_neighbors] # argsort sorts in ascending order
        # find categories nearest to x
        classifier = 0
        item = 0 # default in case y is empty for some reason
        groups = {}
        count = {}
        for i in nearest_ids:
            item = y[i]
            groups.setdefault(item, []).append(item)
        for group in groups:
            count.setdefault(group,[]).append(len(groups[group]))
        for key, item in count.items():
            classifier = {key:np.multiply(item,(1/n_neighbors))}
            #print(item, classifier) # debug
            if classifier[key] > 0.5:
                prediction = key
        # calculate probability that x is in category i
        y_predicted = np.append(y_predicted, prediction)
    # compute error rate
    #print(y_predicted) # debug
    error = (1/n) * sum(list(y[i] != y_predicted[i] for i in range(len(y))))
    error = np.float32(error)
    print(error)
    
    return(error, y_predicted)
        
