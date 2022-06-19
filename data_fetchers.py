import numpy as np
from scipy.stats import zscore


def get_satellite():
    XY = np.genfromtxt('datasets/satellite.csv', delimiter=";", skip_header=1)
    Y = XY[:, -1] - 1
    X = XY[:, :-1]
    return zscore(X, axis=0), Y

def get_airfoil():
    XY = np.genfromtxt('datasets/airfoil_noise.csv', delimiter=";", skip_header=1)
    X = zscore(XY[:, :-1], axis=0)
    Y = XY[:, -1]
    return X, Y

def get_winequality():
    XY = np.genfromtxt('datasets/winequality-red.csv', delimiter=";", skip_header=1)
    X = zscore(XY[:, :-1], axis=0)
    Y = XY[:, -1]
    return X, Y
