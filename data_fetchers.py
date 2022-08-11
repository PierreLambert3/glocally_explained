import numpy as np
from scipy.stats import zscore
import pandas as pd

def get_satellite():
    XY = np.genfromtxt('datasets/satellite.csv', delimiter=";", skip_header=1)
    Y = XY[:, -1] - 1
    X = XY[:, :-1]
    return zscore(X, axis=0), Y.astype(int), ["variable "+str(i) for i in range(X.shape[1])]

def get_RNA():
    XY = np.load('datasets/RNAseq_N3k.npy')
    X = XY[:, :-1]
    Y =  XY[:, -1]
    X = X[:, :8]
    return zscore(X, axis=0), Y.astype(int),  ["PC "+str(i) for i in range(X.shape[1])]

def get_countries():
    df = pd.read_csv("datasets/country_dataset_with_names.csv", index_col = 0)
    countries = df.index
    features = df.columns.values.tolist()
    Xhd = df.to_numpy()
    from sklearn.preprocessing import StandardScaler
    Xhd = StandardScaler().fit_transform(Xhd)
    X = Xhd
    Y = np.ones(Xhd.shape[0], dtype=int)
    return Xhd, Y, features


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
