#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 10:23:55 2022

@author: rmarion
"""

import BIOT
import Multi_BIOT
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans


# Import example dataset
# data = pd.read_csv("country_dataset.csv", index_col = 0)
data = pd.read_csv("datasets/country_dataset_with_names.csv", index_col = 0)
# data = pd.read_csv("datasets/airfoil_noise.csv")
# data = np.genfromtxt('datasets/airfoil_noise.csv', delimiter=";", skip_header=1)[:,:-1]
data = pd.DataFrame(data)

# Create t-SNE embedding
perplexity = 10
sc = StandardScaler()
data_norm = sc.fit_transform(data)
tsne = TSNE(n_components=2, verbose=1, random_state=123, perplexity = perplexity)
Y = tsne.fit_transform(data_norm) # embedding
X = data.copy() # features for explaining embedding (i.e. data used to generate embedding)

X = X[:200]
Y = Y[:200]

# Plot clusters for naive solution (kmeans applied to embedding)
K = 5
kmeans = KMeans(n_clusters=K).fit(Y)
clusters_init = kmeans.labels_
Multi_BIOT.plot_clusters(Y, clusters_init)

# Run Multi BIOT
max_lam = BIOT.calc_max_lam(X, Y)
n_lam = 10
lam_values = max_lam*(10**np.linspace(-1, 0, num=n_lam, endpoint=True, retstep=False, dtype=None))
lam_list = lam_values.tolist()

## Without initial clusters (cross-validation over candidate values for K and lambda: K_list and lam_list)
# K_list = [3, 4, 5, 6] # Number of clusters
# Yhat, W_list, w0_list, R_list, clusters = Multi_BIOT.CV_Multi_BIOT (X_train = X, X_test = X, Y_train = Y, lam_list = lam_list, K_list = K_list, clusters = None, rotation = True)
#
# Multi_BIOT.plot_clusters(Y, clusters)


K_list = [3]
## With initial clusters (cross-validation over candidate values for lambda: lam_list)
initial_clusters = clusters_init
Yhat, W_list, w0_list, R_list, clusters = Multi_BIOT.CV_Multi_BIOT (X_train = X, X_test = X, Y_train = Y, lam_list = lam_list, K_list = None, clusters = initial_clusters, rotation = True)

Multi_BIOT.plot_clusters(Y, initial_clusters) # Plot of initial clusters
Multi_BIOT.plot_clusters(Y, clusters) # Plot of final clusters
