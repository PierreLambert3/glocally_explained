#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 16:19:23 2022

@author: rmarion
"""

# Example code for running BIOT

import BIOT
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
    
###############################################################################

# Import data
Y = pd.read_csv("embedding.csv", index_col = 0)
sc = StandardScaler(with_std = False)
Y_cent = sc.fit_transform(Y)
X = pd.read_csv("features.csv", index_col = 0)


# # Compare results to R outputs
# sc = StandardScaler()
# X_norm = sc.fit_transform(X)
# sc = StandardScaler(with_std=False)
# Y_norm = sc.fit_transform(Y)


# R, W, w0 = BIOT.BIOT(X_norm, Y_norm, lam = 0.01)

# Define candidate lambda values
max_lam = BIOT.calc_max_lam(X, Y)
n_lam = 10
lam_values = max_lam*(10**np.linspace(-2, 0, num=n_lam, endpoint=True, retstep=False, dtype=None))
lam_list = lam_values.tolist()

# Run BIOT with 10 fold CV to choose lambda

Yhat, W, w0, R = BIOT.CV_BIOT (X_train = X, X_test = X, Y_train = Y_cent, lam_list = lam_list, rotation = True, fit_intercept = False, num_folds=10, random_state = 1, scoring = 'neg_mean_squared_error')
YR = Y_cent @ R # rotated embedding
YhatR = Yhat @ R # Lasso predictions for rotated embedding

# counterclockwise rotation angle 
rotation_radians = math.acos(R[0,0])
rotation_degrees = math.degrees(rotation_radians) 
