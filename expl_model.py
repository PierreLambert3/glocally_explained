import numpy as np
from numpy import mean, std, zeros, ones
from numpy.random import uniform, normal, exponential
from scipy.linalg import norm
import BIOT

class Local_explanation_wrapper():
    def __init__(self, sample_idx, Xld, Xhd, method='biot', fit=True):
        if method == 'pca':
            model = Dummy_model()
        else:
            model = BIOT_model()

        self.model = model
        self.sample_idx = sample_idx
        self.center_LD = np.mean(Xld[sample_idx], axis=0)
        if fit:
            self.model.fit(Xld[sample_idx], Xhd[sample_idx], center_LD = self.center_LD, center_HD = np.mean(Xhd[sample_idx], axis=0))
        self.axis2d = model.axis2d # shape (2, 2)

    def get_features_coeff(self):
        return self.model.features_coeffs_ax1, self.model.features_coeffs_ax2

    def compute_errors(self, Xhd, Xld):
        Xld_hat = self.model.transform(Xhd)
        errors = np.mean(np.abs(Xld_hat - Xld), axis=1)
        return errors

class Expl_model():
    def __init__(self):
        self.center_LD = None # shape: (2,)
        self.center_HD = None # shape: (HDdimensionality,)
        self.axis2d = None # shape:(2, 2) # 2 axis, each one has 2 values for each variables in LD
        self.features_coeffs_ax1 = None # shape: (HDdimensionality,)
        self.features_coeffs_ax2 = None

    def fit(self, Xld, Xhd):
        self.center_LD = np.mean(Xld, axis=0)
        self.center_HD = np.mean(Xhd, axis=0)
        self.axis2d = np.array([np.array([1., 0.]), np.array([0., 1.])])
        self.axis2d[0] /= norm(self.axis2d[0]) + 1e-12
        self.axis2d[1] /= norm(self.axis2d[1]) + 1e-12
        self.features_coeffs_ax1 = np.random.normal(size=Xhd.shape[1])
        self.features_coeffs_ax2 = np.random.normal(size=Xhd.shape[1])

    def transform(self, Xhd):
        if self.center_HD is None:
            42/0

class BIOT_model(Expl_model):
    def __init__(self):
        super(BIOT_model, self).__init__()
        self.W  = None
        self.w0 = None
        self.R  = None

    def transform(self, Xhd, HD_is_centered=False, uncenter=True):
        assert self.center_HD is not None and self.center_LD is not None
        N, M = Xhd.shape
        intercept = np.tile(self.w0, (N, 1))
        if not HD_is_centered:
            Xhd = Xhd - self.center_HD
        if uncenter:
            return ((intercept + (Xhd @ self.W)) @ self.R.T) + self.center_LD
        else:
            return ((intercept + (Xhd @ self.W)) @ self.R.T)


    def fit(self, Xld, Xhd, center_LD=None, center_HD=None):
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        self.center_HD = np.mean(Xhd, axis=0)
        self.center_LD = np.mean(Xld, axis=0)
        if center_LD is not None:
            self.center_LD = center_LD
        if center_HD is not None:
            self.center_HD = center_HD

        Xhd_centered = Xhd - self.center_HD
        Xld_centered = Xld - self.center_LD

        Xhd_centered = pd.DataFrame(Xhd_centered)
        max_lam = BIOT.calc_max_lam(Xhd_centered, Xld_centered)
        # max_lam = 0.99
        n_lam = 10
        lam_values = max_lam*(10**np.linspace(-2, 0, num=n_lam, endpoint=True, retstep=False, dtype=None))
        lam_list = lam_values.tolist()

        # higher lamdas = more L1 = more sparsity in explanations at risk of full 0s
        # max_lam = max(0.3, max_lam)
        # lam_list = [max_lam]
        # for i in range(10):
        #     e = lam_list[-1] * 0.92
        #     if e > 0.1:
        #         lam_list.append(e)
        # print(lam_list)
        # 1/0
        # lam_list = [max_lam, 0.5,0.6,0.75]
        # print(max_lam, " ____")
        # # lam_list = [max_lam]

        Yhat, W, w0, R = BIOT.CV_BIOT(X_train = Xhd_centered, X_test = Xhd_centered, Y_train = Xld_centered, lam_list = lam_list, rotation = True, fit_intercept = False, num_folds=4, random_state = 1, scoring = 'neg_mean_squared_error')
        W = W.to_numpy()

        print(" \n DONE BIOT \n")

        self.W = W
        self.w0 = w0
        self.R = R
        self.axis2d = self.R
        self.features_coeffs_ax1 = W[:,0]
        self.features_coeffs_ax2 = W[:,1]



class Dummy_model(Expl_model):
    def __init__(self):
        super(Dummy_model, self).__init__()
        self.linreg1 = None
        self.linreg2 = None

    def transform(self, Xhd):
        assert self.center_HD is not None and self.center_LD is not None

        centered_HD = Xhd - self.center_HD
        Xld1_hat = self.linreg1.predict(centered_HD)
        Xld2_hat = self.linreg2.predict(centered_HD)

        return np.hstack((Xld1_hat.reshape((-1,1)), Xld2_hat.reshape((-1,1)))) + self.center_LD


    def fit(self, Xld, Xhd, center_HD=None, center_LD=None):
        self.center_HD = center_HD
        self.center_LD = center_LD
        if center_HD is None:
            self.center_HD = np.mean(Xhd, axis=0)
        if center_LD is None:
            self.center_LD = np.mean(Xld, axis=0)

        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression

        local_pca = PCA(n_components=2).fit(Xld-self.center_LD)
        self.axis2d = local_pca.components_

        projected = np.dot((Xld-self.center_LD), self.axis2d.T)
        centered_HD = Xhd - self.center_HD
        self.linreg1 = LinearRegression().fit(centered_HD, projected[:, 0])
        self.linreg2 = LinearRegression().fit(centered_HD, projected[:, 1])
        self.features_coeffs_ax1 = self.linreg1.coef_
        self.features_coeffs_ax2 = self.linreg2.coef_
