#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:33:28 2022

@author: rmarion
original code: https://github.com/rebeccamarion/BIOT_Python_Package  (MIT licence)
"""

import BIOT
import pandas as pd
import numpy as np
import math
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.compose import TransformedTargetRegressor
from sklearn.neighbors import NearestNeighbors


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans

###############################################################################

# Functions and classes

def plot_clusters (Y, clusters):

    df = pd.DataFrame()
    df["cluster"] = clusters

    df["dim-1"] = Y[:,0]
    df["dim-2"] = Y[:,1]
    sns.scatterplot(x="dim-1", y="dim-2", hue = "cluster",
                    data=df).set(title="T-SNE projection")

def BIOT_Crit_i (X, Y, R, W, w0, lam, i):

    n = X.shape[0]

    W = check_array(W)

    diffs = (Y[i, :].T @ R) - (w0 + (X[i, :].T @ W))
    LS = np.linalg.norm(diffs)**2
    L1 = BIOT.Global_L1_Norm(W)

    err = ((1/(2*n)) * LS) # error term
    spars = (lam * L1) # sparsity term
    total = err + spars # full criterion value

    return err, spars, total

def Update_Cs (X, Y, R_list, W_list, w0_list, lam):

    n = X.shape[0]
    nb_models = len(R_list)

    Cs_list = list()
    final_crits = list()

    for i in range(n):

        err_list = list() # just prediction error
        crit_list = list() # full criterion
        for k in range(nb_models):
            err, spars, total = BIOT_Crit_i(X, Y, R_list[k], W_list[k], w0_list[k], lam, i)
            err_list.append(err)
            crit_list.append(total)

        new_cluster = np.argmin(np.array(err_list)) # assign cluster for instance i based on prediction error
        Cs_list.append(new_cluster)
        new_crit_value = crit_list[new_cluster] # criterion value for chosen cluster and instance i
        final_crits.append(new_crit_value)

    new_Cs = np.array(Cs_list) # new cluster assignments
    crit_value = np.sum(np.array(final_crits)) # new overall criterion value
    return new_Cs, crit_value

def Multi_BIOT (X, Y, lam, K = 3, clusters = None, fit_intercept = True, max_iter = 500, eps = 1e-6, rotation = False, feature_names = None):

    # If initial clusters are not provided
    if clusters is None :
        kmeans = KMeans(n_clusters=K).fit(Y)
        clusters = kmeans.labels_


    diff = math.inf
    iter_index = 0
    crit_list = [math.inf]

    # Main algorithm
    while iter_index < max_iter and diff > eps:

        R_list = list()
        W_list = list()
        w0_list = list()
        nb_models = (np.max(clusters) + 1).astype("int32")
        for cluster_index in range(nb_models):

            # Update BIOT model for each cluster
            R, W, w0 = BIOT.BIOT(X = X[clusters == cluster_index, : ], Y = Y[clusters == cluster_index, : ], lam = lam, fit_intercept = fit_intercept, rotation = rotation)
            W_named = pd.DataFrame(W, index = feature_names, copy = True)
            R_list.append(R)
            W_list.append(W_named)
            w0_list.append(w0)

        # Update clusters
        clusters, crit = Update_Cs (X, Y, R_list, W_list, w0_list, lam)

        # CHECK CONVERGENCE
        crit_list.append(crit)
        diff = np.absolute(crit_list[iter_index] - crit_list[iter_index + 1])

        iter_index += 1

    return R_list, W_list, w0_list, clusters

class MultiBIOTRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, lam = 1, K = 4, init_clusters = True, rotation = False, fit_intercept = True, feature_names = None):

        self.lam = lam # value of sparsity hyperparameter
        self.rotation = rotation # True if the orthogonal transformation must be a rotation
        self.feature_names = feature_names # feature names
        self.K = K # Number of clusters (irrelevant if initial clusters are provided)
        self.fit_intercept = fit_intercept # True if intercept should be learned
        self.init_clusters = init_clusters # True if initial clusters are provided (last column of X)

    def fit(self, X, Y):

        X = check_array(X)
        Y = check_array(Y)

        # If initial clusters are provided (in last column of X)
        if self.init_clusters:
            clusters = X[:, -1]
            X = X[:, :-1]
        else :
            clusters = None


        R_list, W_list, w0_list, clusters = Multi_BIOT(X = X, Y = Y, K = self.K, clusters = clusters, lam = self.lam, rotation = self.rotation, fit_intercept = self.fit_intercept, feature_names = self.feature_names)


        self.R_list_ = R_list
        self.W_list_ = W_list
        self.w0_list_ = w0_list
        self.clusters_ = clusters
        self.X_ = X
        self.Y_ = Y

        return self

    def predict(self, X):

        check_is_fitted(self)
        X = check_array(X)

        # If initial clusters are provided (in last column of X)
        if self.init_clusters:
            X = X[:, :-1]

        # Predict using model for the nearest neighbor in the training set X_
        knn_search = NearestNeighbors(n_neighbors=1)
        knn_search.fit(self.X_)

        dims = self.W_list_[0].shape[1]
        Y_pred = np.zeros((X.shape[0], dims))

        for i in range(X.shape[0]):

            nn_id = knn_search.kneighbors(X[i, None, :], return_distance=False)[0, 0]
            cluster_index = self.clusters_[nn_id]
            W = check_array(self.W_list_[cluster_index])
            R = check_array(self.R_list_[cluster_index])
            w0 = self.w0_list_[cluster_index]
            Y_pred[i, :] =  (w0.reshape(1, -1) + X[i, None, :] @ W) @ R.T

        return Y_pred


class myMultiPipe(Pipeline):

    def fit(self, X, Y):
        """Calls last elements .R_list_, .W_list_, .w0_list_, .clusters_, etc.
        Based on the sourcecode for decision_function(X).
        Link: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/pipeline.py
        ----------
        """

        super(myMultiPipe, self).fit(X, Y)

        #self.coef_=self.steps[-1][-1].coef_
        self.R_list_ = self.steps[-1][-1].R_list_
        self.W_list_ = self.steps[-1][-1].W_list_
        self.w0_list_ = self.steps[-1][-1].w0_list_
        self.clusters_ = self.steps[-1][-1].clusters_
        self.X_ = self.steps[-1][-1].X_
        self.Y_ = self.steps[-1][-1].Y_



        return


def CV_Multi_BIOT (X_train, X_test, Y_train, lam_list, K_list = None, clusters = None, fit_intercept = True, num_folds=10, random_state = 1, rotation = False, scoring = 'neg_mean_squared_error'):

   X_train_ = X_train.copy()
   X_test_ = X_test.copy()
   Y_train_ = Y_train.copy()

   d = X_train_.shape[1]

   # Create feature_names vector
   if isinstance(X_train_, pd.DataFrame):
       feature_names = X_train_.columns

   else:
       feature_names = range(X_train_.shape[0])

   # If initial clusters are provided
   if clusters is not None:
       init_clusters = True
   # If initial clusters are not provided
   else :
       init_clusters = False


   # define the model pipeline
   pipe = myMultiPipe([
        ('ct', ColumnTransformer([('sc', StandardScaler(), np.arange(0, d).tolist())], remainder='passthrough')),
        #('sc', StandardScaler()),
        ('MultiBIOT', MultiBIOTRegressor(init_clusters = init_clusters, rotation = rotation, fit_intercept = fit_intercept, feature_names = feature_names))
    ])

   # If initial clusters are provided
   if clusters is not None:
       K_list = [len(np.unique(clusters))]
       if isinstance(X_train_, pd.DataFrame):
           X_train_["clusters"] = clusters
           X_test_["clusters"] = np.zeros(X_test_.shape[0])

       else:
           X_train_ = np.c_[X_train_, clusters]
           X_test_ = np.c_[X_test_, np.zeros(X_test_.shape[0])]

   space = dict()
   space['regressor__MultiBIOT__lam'] = lam_list
   space['regressor__MultiBIOT__K'] = K_list

   # configure the cross-validation procedure
   cv = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
   # define search
   estimator = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler(with_std = False))
   search = GridSearchCV(estimator = estimator, param_grid = space, scoring=scoring, cv=cv, refit=True, verbose = 2)
   # execute search
   search.fit(X_train_, Y_train_)
   # get the best performing model fit on the whole training set
   best_model = search.best_estimator_
   # evaluate model on the hold out dataset
   Yhat = best_model.predict(X_test_)


   W_list = best_model.regressor_.W_list_
   w0_list = best_model.regressor_.w0_list_
   R_list = best_model.regressor_.R_list_
   clusters = best_model.regressor_.clusters_

   return Yhat, W_list, w0_list, R_list, clusters
