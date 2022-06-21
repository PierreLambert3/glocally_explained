import numpy as np
from scipy.stats import zscore


def get_satellite():
    XY = np.genfromtxt('datasets/satellite.csv', delimiter=";", skip_header=1)
    Y = XY[:, -1] - 1
    X = XY[:, :-1]
    return zscore(X, axis=0), Y.astype(int), ["variable "+str(i) for i in range(X.shape[1])]

def get_airfoil():
    XY = np.genfromtxt('datasets/airfoil_noise.csv', delimiter=";", skip_header=1)
    X = zscore(XY[:, :-1], axis=0)
    Y = XY[:, -1]
    return X, Y.astype(float), ["Hz", "angle", "chord len", "free-stream v", "suc. displ. thickness"]

def get_winequality():
    XY = np.genfromtxt('datasets/winequality-red.csv', delimiter=";", skip_header=1)
    X = zscore(XY[:, :-1], axis=0)
    Y = XY[:, -1]
    return X, Y.astype(float), ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",\
                "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
