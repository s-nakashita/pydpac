import numpy as np
import numpy.linalg as la
# numpy 1.17.0 or later
#from numpy.random import default_rng
#rng = default_rng()
from numpy import random


def h_operator(x, operator="linear"):
    if operator == "linear":
        return x
    elif operator == "quadratic":
        return x**2
    elif operator == "cubic":
        return x**3
    elif operator == "quartic":
        return x**4 
    elif operator == "quadratic-nodiff":
        return np.where(x >= 0.5, x**2, -x**2)
    elif operator == "cubic-nodiff":
        return np.where(x >= 0.5, x**3, -x**3)
    elif operator == "quartic-nodiff":
        return np.where(x >= 0.5, x**4, -x**4)
    elif operator == "test":
        return 0.5*x*(1.0+0.1*np.abs(x))

def dhdx(x, operator="linear"):
    if operator == "linear":
        return np.diag(np.ones(x.size))
    elif operator == "quadratic":
        return np.diag(2 * x)
    elif operator == "cubic":
        return np.diag(3 * x**2)
    elif operator == "quartic":
        return np.diag(4 * x**3)
    elif operator == "quadratic-nodiff":
        return np.diag(np.where(x >= 0.5, 2*x, -2*x))
    elif operator == "cubic-nodiff":
        return np.diag(np.where(x >= 0.5, 3*x**2, -3*x**2))
    elif operator == "quartic-nodiff":
        return np.diag(np.where(x >= 0.5, 4*x**3, -4*x**3))
    elif operator == "test":
        return np.diag(0.5+0.1*np.abs(x))


def add_noise(x, sigma):
# numpy 1.17.0 or later
#    return x + rng.normal(0, mu=sigma, size=x.size)
    #np.random.seed(514)
    return x + random.normal(0, scale=sigma, size=x.size).reshape(x.shape)
