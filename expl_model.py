import numpy as np
import scipy.linalg.norm as norm


class Local_explanation():
    def __init__(self, sample, center=None):
        Ns, M = sample.shape
        if center is None:
            center = np.mean(sample, axis=0)
        self.components = np.ones((2, M)) / norm(np.ones((2, M)))
