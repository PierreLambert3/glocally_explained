import numpy as np
from numpy import mean, std, zeros, ones
from numpy.random import uniform, normal, exponential
from scipy.linalg import norm
import BIOT

class Local_explanation_wrapper():
    def __init__(self, sample_idx, Xld, Xhd, model='pca'):
        if model == 'pca':
            model = Dummy_model()
        else:
            model = BIOT_model()

        self.model = model
        self.sample_idx = sample_idx
        self.model.fit(Xld[sample_idx], Xhd[sample_idx])
        self.axis2d = model.axis2d
        self.center = model.center
        self.features_coeffs_ax1 = model.features_coeffs_ax1
        self.features_coeffs_ax2 = model.features_coeffs_ax2

        # target_direction = np.array([np.sqrt(2), np.sqrt(2)])
        # simi1 = np.dot(target_direction, self.axis2d[0])
        # simi2 = np.dot(target_direction, self.axis2d[1])
        # if simi1 < 0.:
        #     self.axis2d[0] *= -1.
        # if simi2 < 0.:
        #     self.axis2d[1] *= -1.

    def compute_errors(self, Xhd, Xld):
        Xld_hat = self.model.transform(Xhd)
        Xld_hat += self.model.center
        errors = np.sqrt(np.sum((Xld_hat - Xld)**2, axis=1))
        # errors = np.sqrt(np.mean((Xld_hat - Xld)**2, axis=1))
        return errors

class Expl_model():
    def __init__(self):
        self.center = None # shape: (2,)
        self.HDcenter = None # shape: (HDdimensionality,)
        self.axis2d = None # shape:(2, 2) # 2 axis, each one has 2 values for each variables in LD
        self.features_coeffs_ax1 = None # shape: (HDdimensionality,)
        self.features_coeffs_ax2 = None
        self.HDcenter = None

    def fit(self, Xld, Xhd):
        self.center = np.mean(Xld, axis=0)
        self.HDcenter = np.mean(Xhd, axis=0)
        self.axis2d = np.array([np.array([1., 0.]), np.array([0., 1.])])
        self.axis2d[0] /= norm(self.axis2d[0]) + 1e-12
        self.axis2d[1] /= norm(self.axis2d[1]) + 1e-12
        self.features_coeffs_ax1 = np.random.normal(size=Xhd.shape[1])
        self.features_coeffs_ax2 = np.random.normal(size=Xhd.shape[1])

    def transform(self, Xhd):
        if self.center is None:
            42/0
        return np.random.uniform(size=(Xhd.shape[0], 2))

class BIOT_model(Expl_model):
    def __init__(self):
        super(BIOT_model, self).__init__()
        self.W  = None
        self.w0 = None
        self.R  = None

    def transform(self, Xhd):
        if self.center is None:
            42/0
        N, M = Xhd.shape
        intercept = np.tile(self.w0, (N, 1))

        # print(self.center)
        # print(self.HDcenter)
        # print(self.axis2d)
        # print(self.features_coeffs_ax1)
        # print(self.features_coeffs_ax2)
        # print(self.center.shape)
        # print(self.HDcenter.shape)
        # print(self.axis2d.shape)
        # print(self.features_coeffs_ax1.shape)
        # print(self.features_coeffs_ax2.shape)

        return ((intercept + (Xhd @ self.W)) @ self.R.T)


    def fit(self, Xld, Xhd):
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        self.center = np.mean(Xld, axis=0)
        self.HDcenter = np.mean(Xhd, axis=0)

        # Xhd_centered = Xhd - self.HDcenter
        Xld_centered = Xld - self.center


        # sc = StandardScaler(with_std = False)
        # Xld = sc.fit_transform(Xld)

        Xhd = pd.DataFrame(Xhd)

        max_lam = BIOT.calc_max_lam(Xhd, Xld)
        n_lam = 10
        lam_values = max_lam*(10**np.linspace(-2, 0, num=n_lam, endpoint=True, retstep=False, dtype=None))
        lam_list = lam_values.tolist()


        Yhat, W, w0, R = BIOT.CV_BIOT(X_train = Xhd, X_test = Xhd, Y_train = Xld_centered, lam_list = lam_list, rotation = True, fit_intercept = False, num_folds=10, random_state = 1, scoring = 'neg_mean_squared_error')

        Yhat = Yhat
        W = W.to_numpy() # ugh
        w0 = w0
        R = R

        self.W = W
        self.w0 = w0
        self.R = R
        self.axis2d = R
        self.features_coeffs_ax1 = W[:,0]
        self.features_coeffs_ax2 = W[:,1]



class Dummy_model(Expl_model):
    def __init__(self):
        super(Dummy_model, self).__init__()
        self.linreg1 = None
        self.linreg2 = None

    def transform(self, Xhd):
        if self.center is None:
            42/0
        centered_HD = Xhd - self.HDcenter
        Xld1_hat = self.linreg1.predict(centered_HD)
        Xld2_hat = self.linreg2.predict(centered_HD)

        # print(self.center)
        # print(self.HDcenter)
        # print(self.axis2d)
        # print(self.features_coeffs_ax1)
        # print(self.features_coeffs_ax2)
        # print(self.center.shape)
        # print(self.HDcenter.shape)
        # print(self.axis2d.shape)
        # print(self.features_coeffs_ax1.shape)
        # print(self.features_coeffs_ax2.shape)

        return np.hstack((Xld1_hat.reshape((-1,1)), Xld2_hat.reshape((-1,1))))


    def fit(self, Xld, Xhd):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression
        self.center = np.mean(Xld, axis=0)
        self.HDcenter = np.mean(Xhd, axis=0)
        local_pca = PCA(n_components=2).fit(Xld-self.center)
        self.axis2d = local_pca.components_

        projected = np.dot((Xld-self.center), self.axis2d.T)
        centered_HD = Xhd - self.HDcenter
        self.linreg1 = LinearRegression().fit(centered_HD, projected[:, 0])
        self.linreg2 = LinearRegression().fit(centered_HD, projected[:, 1])
        self.features_coeffs_ax1 = self.linreg1.coef_
        self.features_coeffs_ax2 = self.linreg2.coef_
