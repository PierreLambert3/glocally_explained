import numpy as np
from scipy.stats import zscore
import pandas as pd

def get_satellite():
    XY = np.genfromtxt('datasets/satellite.csv', delimiter=";", skip_header=1)
    Y = XY[:, -1] - 1
    X = XY[:, :-1]
    return zscore(X, axis=0), Y.astype(int), ["variable "+str(i) for i in range(X.shape[1])]

def get_countries():
    df = pd.read_csv("datasets/country_dataset_with_names.csv", index_col = 0)


    countries = df.index

    features = df.columns.values.tolist()
    Xhd = df.to_numpy()
    from sklearn.preprocessing import StandardScaler
    Xhd = StandardScaler().fit_transform(Xhd)
    X = Xhd

    # from sklearn.manifold import TSNE
    # from sklearn.decomposition import PCA
    # from matplotlib import pyplot as plt
    # Xld = TSNE(n_components=2, init='pca', perplexity=4).fit_transform(Xhd)
    # # Xld = PCA(n_components=2).fit_transform(Xhd)
    # plt.scatter(Xld[:, 0], Xld[:, 1])
    # plt.show()
    #
    # 1/0

    Y = np.ones(Xhd.shape[0], dtype=int)
    for i, c in enumerate(countries):
        if "Indonesia" in c:
            Y[i] = 0
    # philipines a droite iran a gauche
    # Y = Xhd[:, np.where(np.array(features) == 'GDP')[0]]
    # Y = ((Y - np.min(Y)) / (np.max(Y) - np.min(Y))).reshape((-1,))
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
