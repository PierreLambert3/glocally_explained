import numpy as np
from numpy import mean, std, zeros, ones
from numpy.random import uniform, normal, exponential
from scipy.linalg import norm

class Local_explanation_wrapper():
    def __init__(self, sample_idx, Xld, Xhd, model=None):
        if model is None:
            model = Dummy_model()
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
        return -errors

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

        # import matplotlib.pyplot as plt
        # plt.scatter(Xld[:, 0], Xld[:, 1])
        # plt.show()
        # trsnaforped = self.transform(Xhd)+self.center
        # plt.scatter(trsnaforped[:, 0], trsnaforped[:, 1])
        # plt.show()










4
