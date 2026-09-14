# AE 498 HW1 AMANDA YAKLIN 
# YAKLIN2@ILLINOIS.EDU

"""This module contains functions to perform multiple linear regression for HW 1 in AE 498 Computational Systems Engineering.

"""


import numpy as np
import scipy
from scipy import stats


def normalize_data(X):
    """Function to normalize input data to [-1, 1].

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor. 
        The first column should be all ones.
    Returns
    -------
    X_normalized : array_like
        Input data (n x (p + 1) dimensions) normalized to the range [-1, 1].

    """

    n, n_columns = np.shape(X)

    # normalize each column
    X_normalized = np.ones((n, n_columns))
    for column in range(n_columns):
        x_column = X[:, column]
        x_max = max(x_column)
        x_min = min(x_column)
        if x_max != x_min:
            x_column_normalized = [
                (x - (x_max + x_min) / 2) / ((x_max - x_min) / 2) for x in x_column
            ]
            X_normalized[:, column] = x_column_normalized

    return X_normalized



def model_fit(X, y):
    """Function to fit a multiple linear regression model to a given dataset using ordinary least squares estimation.

    Parameters
    ----------
    X : array_like
            Input data square matrix (n x (p + 1) dimensions), where each row is an observation and each column is a predictor. 
            The first column should be all ones.
    y : array_like
        Output/response data (n elements), where each element is an observation.

    Returns
    -------
    coefficients : array_like
        Estimated model coefficients (1 + p elements).
    y_predicted : array_like
        Predicted responses for input data (n elements).

    """

    # coefficients
    B = np.linalg.inv(X.T @ X) @ X.T @ y
    # predictions
    y_predicted = X @ B
    # debug
    #print(B, y_predicted) 
    
    return B, y_predicted

def anova(y, y_predicted, p):
    """Function to perform an ANOVA F-test for a multiple linear regression model.

    Parameters
    ----------
    y : array_like
        Output/response data (n elements), where each element is an observation.
    y_predicted : array_like
        Predicted responses for input data (n elements).
    p : int
        Number of predictors.

    Returns
    -------
    p_value : float
        P-value for model F-statistic.
    f_statistic : float
        F-statistic for model F-test.

    """
    n = len(y)
    y_bar = np.mean(y)
    
    # error sum of squares
    ESS = np.sum((y_predicted - y_bar)**2)
    # regression sum of squares
    RSS = np.sum((y - y_predicted)**2) 
    
    f_statistic = (ESS / p)/(RSS / (n - 1- p)) 
    p_value = 1 - stats.f.cdf(f_statistic, p, n - 1 - p)
    # debug
    #print(p_value,f_statistic)
    
    return p_value, f_statistic

def coefficient_tests(X, coefficients, y, y_predicted):
    """Function to perform hypothesis t-tests for coefficients of a multiple linear regression model.

    Parameters
    ----------
    X : array_like
        Input data (n x (p + 1) dimensions), where each row is an observation and each column is a predictor. 
        The first column should be all ones.
    coefficients : array_like
        Estimated model coefficients (1 + p elements).
    y : array_like
        Output/response data (n elements), where each element is an observation.
    y_predicted : array_like
        Predicted responses for input data (n elements).

    Returns
    -------
    p_values : list_like
        P-values for coefficient t-statistics.
    t_statistics : list_like
        T-statistics for coefficient t-tests.

    """
    # no. of observations
    n = X.shape[0]
    # no. of predictors
    p = X.shape[1]
    # diagonal of (xTx)^-1 
    xTx_inv = np.linalg.inv(X.T @ X)
    v = np.diag(xTx_inv)  
                
    # standard error of model
    stderr_betas = np.sqrt(((np.sum((y_predicted - y)**2)) / (n - p)) * v)
    
    t_statistics = coefficients / stderr_betas
    
    # p values
    # since t-distribution is symmetric, multiply by 2
    # since the lowest p-value is very low (O(1e-19)), 
    # use stats.sf (survival function) to prevent floating point error
    p_values = 2 * stats.t.sf(np.abs(t_statistics), n - p) 
    
    return p_values, t_statistics
