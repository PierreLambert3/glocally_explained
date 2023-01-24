#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 17:41:12 2022

@author: rmarion
original code: https://github.com/rebeccamarion/BIOT_Python_Package  (MIT licence)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
import math

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

###############################################################################

# Functions and classes

def Get_W_Lasso (X, Y, lam, fit_intercept = False):

    k = Y.shape[1]
    d = X.shape[1]
    W = np.zeros((d, k))
    w0 = np.zeros(k)
    # Fit Lasso for each dimension of Y
    for dim_index in range(k):
        model = Lasso(alpha = lam, fit_intercept = fit_intercept, max_iter = 5000)
        model.fit(X = X, y = Y[:, dim_index])
        W[:, dim_index] = model.coef_
        if fit_intercept:
            w0[dim_index] = model.intercept_

    return W, w0

def Global_L1_Norm (W):

    # Calculute L1 norm for each column of W
    k = W.shape[1]
    norm_val = 0
    for dim_index in range(k):
        norm_val += np.linalg.norm(W[:, dim_index], ord = 1)

    return norm_val

def BIOT_Crit (X, Y, R, W, w0, lam):

    n = X.shape[0]
    diffs = (Y @ R) - (np.tile(w0, (n, 1)) + (X @ W))
    LS = np.linalg.norm(diffs)**2
    L1 = Global_L1_Norm(W)

    crit = ((1/(2*n)) * LS) + (lam * L1)

    return crit

def BIOT (X, Y, lam, max_iter = 500, eps = 1e-6, rotation = False, R = None, fit_intercept = False):

    d = X.shape[1]
    n = X.shape[0]
    lam_norm = lam/np.sqrt(d)


    # If R is provided, get Lasso solution only
    if R is not None:
        YR = Y @ R
        W, w0 = Get_W_Lasso(X = X, Y = YR, lam = lam_norm, fit_intercept=fit_intercept)
    # Otherwise, run BIOT iterations
    else:
        # Init W
        W, w0 = Get_W_Lasso(X = X, Y = Y, lam = lam_norm, fit_intercept=fit_intercept)

        diff = math.inf
        iter_index = 0
        crit_list = [math.inf]

        while iter_index < max_iter and diff > eps:

            # UPDATE R
            u, s, v = np.linalg.svd((1/(2*n)) * Y.T @ (np.tile(w0, (n, 1)) + (X @ W)))

            # rotation matrix desired (counterclockwise)
            if rotation:
                sv = np.ones(len(s))
                which_smallest_s = np.argmin(s)
                sv[which_smallest_s] = np.sign(np.linalg.det(u @ v))
                R = u @ np.diag(sv) @ v
            # orthogonal transformation matrix desired
            else:
                R = u @ v

            # UPDATE W
            YR = Y @ R
            W, w0 = Get_W_Lasso(X = X, Y = YR, lam = lam_norm, fit_intercept = fit_intercept)

            # CHECK CONVERGENCE
            crit_list.append(BIOT_Crit(X = X, Y = Y, R = R, W = W, w0 = w0, lam = lam_norm))
            diff = np.absolute(crit_list[iter_index] - crit_list[iter_index + 1])

            iter_index += 1


    return R, W, w0

class BIOTRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, lam = 1, R = None, rotation = False, fit_intercept = False, feature_names = None):
        self.lam = lam
        self.R = R
        self.rotation = rotation
        self.feature_names = feature_names
        self.fit_intercept = fit_intercept

    def fit(self, X, Y):

        X = check_array(X)
        Y = check_array(Y)

        R, W, w0 = BIOT(X = X, Y = Y, lam = self.lam, rotation = self.rotation, R = self.R, fit_intercept = self.fit_intercept)
        self.R_ = R
        self.W_ = pd.DataFrame(W, index = self.feature_names)
        self.w0_ = w0

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        n = X.shape[0]
        W = check_array(self.W_)
        R = check_array(self.R_)
        intercept = np.tile(self.w0_, (n, 1))

        return (intercept + (X @ W)) @ R.T

def split_data (X, Y, train_ix, test_ix):
    X = pd.DataFrame(X, copy = True)
    Y = pd.DataFrame(Y, copy = True)
    X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
    Y_train, Y_test = Y.iloc[train_ix, :], Y.iloc[test_ix, :]
    return X_train, X_test, Y_train, Y_test

class myPipe(Pipeline):

    def fit(self, X, Y):
        """Calls last elements .R_ and .W_ method.
        Based on the sourcecode for decision_function(X).
        Link: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/pipeline.py
        ----------
        """

        super(myPipe, self).fit(X, Y)

        #self.coef_=self.steps[-1][-1].coef_
        self.R_ = self.steps[-1][-1].R_
        self.W_ = self.steps[-1][-1].W_
        self.w0_ = self.steps[-1][-1].w0_


        return

def CV_BIOT (X_train, X_test, Y_train, lam_list, fit_intercept = False, num_folds=10, random_state = 1, R = None, rotation = False, scoring = 'neg_mean_squared_error'):

   if isinstance(X_train, pd.DataFrame):
       feature_names = X_train.columns

   else:
       feature_names = range(X_train.shape[0])


   # define the model pipeline
   pipe = myPipe([
        ('sc', StandardScaler()),
        ('BIOT', BIOTRegressor(R = R, rotation = rotation, fit_intercept = fit_intercept, feature_names = feature_names))
    ])

   space = dict()
   space['regressor__BIOT__lam'] = lam_list

   # configure the cross-validation procedure
   cv = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
   # define search
   estimator = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler(with_std = False))
   search = GridSearchCV(estimator = estimator, param_grid = space, scoring=scoring, cv=cv, refit=True)
   # execute search
   search.fit(X_train, Y_train)
   # get the best performing model fit on the whole training set
   best_model = search.best_estimator_
   # evaluate model on the hold out dataset
   Yhat = best_model.predict(X_test)


   W = best_model.regressor_.W_
   R = best_model.regressor_.R_
   w0 = best_model.regressor_.w0_

   return Yhat, W, w0, R


def calc_max_lam (X, Y):
    n = X.shape[0]
    sc = StandardScaler()
    X_norm = sc.fit_transform(X)
    sc = StandardScaler(with_std=False)
    Y_norm = sc.fit_transform(Y)
    max_lam = np.max(np.absolute(X_norm.T @ Y_norm))/n

    return max_lam
