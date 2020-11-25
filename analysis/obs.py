import numpy as np
# numpy 1.17.0 or later
#from numpy.random import default_rng
#rng = default_rng()
from numpy import random

class Obs():
    def __init__(self, operator, sigma):
        self.operator = operator
        self.sigma = sigma

    def get_op(self):
        return self.operator

    def get_sig(self):
        return self.sigma

    def set_r(self, nx):
        r = np.diag(np.ones(nx)*self.sigma*self.sigma)
        rmat = np.diag(np.ones(nx) / self.sigma)
        rinv = rmat.transpose() @ rmat
        return r, rmat, rinv

    def h_operator(self, x):
        if self.operator == "linear":
            return x
        elif self.operator == "quadratic":
            return x**2
        elif self.operator == "cubic":
            return x**3
        elif self.operator == "quartic":
            return x**4 
        elif self.operator == "quadratic-nodiff":
            return np.where(x >= 0.5, x**2, -x**2)
        elif self.operator == "cubic-nodiff":
            return np.where(x >= 0.5, x**3, -x**3)
        elif self.operator == "quartic-nodiff":
            return np.where(x >= 0.5, x**4, -x**4)
        elif self.operator == "test":
            return 0.5*x*(1.0+0.1*np.abs(x))

    def dhdx(self, x):
        if self.operator == "linear":
            return np.diag(np.ones(x.size))
        elif self.operator == "quadratic":
            return np.diag(2 * x)
        elif self.operator == "cubic":
            return np.diag(3 * x**2)
        elif self.operator == "quartic":
            return np.diag(4 * x**3)
        elif self.operator == "quadratic-nodiff":
            return np.diag(np.where(x >= 0.5, 2*x, -2*x))
        elif self.operator == "cubic-nodiff":
            return np.diag(np.where(x >= 0.5, 3*x**2, -3*x**2))
        elif self.operator == "quartic-nodiff":
            return np.diag(np.where(x >= 0.5, 4*x**3, -4*x**3))
        elif self.operator == "test":
            return np.diag(0.5+0.1*np.abs(x))

    def add_noise(self, x):
# numpy 1.17.0 or later
#    return x + rng.normal(0, mu=sigma, size=x.size)
        #np.random.seed(514)
        return x + random.normal(0, scale=self.sigma, size=x.size).reshape(x.shape)
