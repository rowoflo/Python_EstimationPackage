#!/usr/bin/python
"""
.. module:: kalman
   :platform: Mac
   :synopsis: A collection of useful filtering and smoothing tools all derived from Kalman filtering techniques.

.. moduleauthor:: Rowland O'Flaherty <rowlandoflaherty.com>

"""

import numpy as np

def filter(x, P, Phi, H, W, V, z):
    """This function returns an optimal expected value of the state and covariance error matrix given an update and system parameters.

    :param x: **N x 1 numpy.ndarray** Current estimate of state at time t-1.
    :param P: **N x N numpy.ndarray** Current estimate of error covariance matrix at time t-1.
    :param Phi: **N x N numpy.ndarray** Current discrete time state transition matrix at time t-1.
    :param H: **M x N numpy.ndarray** Current observation model matrix at time t.
    :param W: **N x 1 or N x N numpy.ndarray** Process noise covariance at time t-1. If vector, diagonals of covariance.
    :param V: **M x 1 or M x M numpy.ndarray** Measurement noise covariance at time t. If vector, diagonals of covariance.
    :param z: **M x 1 numpy.ndarray** Measurement at time t.

    :returns: **(x, P, K, w, v, chi2) tuple**\n
        **x -- N x 1 numpy.ndarray** Updated estimate of state at time t.\n
        **P -- N x N numpy.ndarray** Updated estimate of error covariance matrix at time t.\n
        **K -- N x M numpy.ndarray** Kalman gain matrix at time t.\n
        **w -- N x 1 numpy.ndarray** Estimated process noise value at time t.\n
        **v -- M x 1 numpy.ndarray** Estimated measurement noise value at time t.\n
        **chi2 -- 1 x 1 number** Estimated chi-squared value at time t.\n

    """

    # Check inputs
    # Don't know how to do this yet

    [M,N] = np.shape(H)
    I = np.eye(N) # N x N identity matrix

    x_p = np.dot(Phi, x) # Prediction of estimated state vector
    P_p = np.dot(Phi, np.dot(P, Phi.T)) + W # Prediction of error covariance matrix
    S = np.dot(H, np.dot(P_p, H.T)) + V # Sum of error variances
    S_inv = np.linalg.inv(S) # Inverse of sum of error variances
    
    # iM = (S == np.inf) # Infinite mask
    # S_inv = np.zeros(np.shape(S)) #Inverse of sum of error variances
    # S_inv[~np.any(iM, axis = 0), ~np.any(iM, axis = 1)] = np.linalg.inv(S[~np.any(iM, axis = 0), ~np.any(iM, axis = 1)])

    K = np.dot(P_p, np.dot(H.T, S_inv)) # Kalman gain
    r = z - np.dot(H, x_p) # Prediction residual
    w = np.dot(-K, r) # Process error
    x = x_p - w # Update estimated state vector
    v = z - np.dot(H, x) # Measurement error
    P = np.dot((I - np.dot(K, H)), np.dot(P_p, (I - np.dot(K, H)).T)) + np.dot(K, np.dot(V, K.T)) # Updated error covariance matrix
    chi2 = np.dot(r.T, np.dot(S_inv, r)) # Chi-squared value

    return (x, P, K, w, v, chi2)
