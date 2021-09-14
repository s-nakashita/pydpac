import numpy as np
# numpy 1.17.0 or later
#from numpy.random import default_rng
#rng = default_rng()
from numpy import random
import math
import logging
from logging.config import fileConfig

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')

class Obs():
    def __init__(self, operator, sigma):
        self.operator = operator
        self.sigma = sigma
        self.gamma = 3
        logger.info(f"operator={self.operator}, sigma={self.sigma}, gamma={self.gamma}")

    def get_op(self):
        return self.operator

    def get_sig(self):
        return self.sigma

    def set_r(self, obsloc):
        p = obsloc.size
        r = np.diag(np.ones(p)*self.sigma*self.sigma)
        rmat = np.diag(np.ones(p) / self.sigma)
        rinv = rmat.transpose() @ rmat
        return r, rmat, rinv

    def h_operator(self, obsloc, x):
        #logger.debug(f"x={x}")
        hxf = self.hx(x)
        #logger.debug(f"hx={hxf}")
        nobs = obsloc.size
        logger.debug(f"nobs={nobs}")
        if hxf.ndim == 1:
            obs = np.zeros(nobs)
            for k in range(nobs):
                obs[k] = self.itpl1d(obsloc[k], hxf)
        else:
            nens = hxf.shape[1]
            obs = np.zeros((nobs, nens))
            for k in range(nobs):
                for j in range(nens):
                    obs[k,j] = self.itpl1d(obsloc[k], hxf[:,j])
        return obs

    def hx(self, x):
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
            return 0.5*x*(1.0+np.power((0.1*np.abs(x)), (self.gamma-1)))
        elif self.operator == "abs":
            return np.abs(x)
        elif self.operator == "hint":
            x_int = np.zeros_like(x)
            for i in range(-3, 4):
                x_int += np.roll(x, i, axis=0)
            x_int /= 7
            return x_int

    def dh_operator(self, obsloc, x):
        nobs = obsloc.size
        nx = x.size
        jach = np.zeros((nobs, nx))
        itpl_mat = np.zeros((nobs, nx))
        dhdxf = self.dhdx(x)
        for k in range(nobs):
            ri = obsloc[k]
            i = math.floor(ri)
            ai = ri - float(i)
            if i < nx-1:
                #jach[k,i] = (1.0 - ai)*dhdxf[i]
                itpl_mat[k,i] = (1.0 - ai)
                #jach[k,i+1] = ai*dhdxf[i+1]
                itpl_mat[k,i+1] = ai
            else:
                #jach[k,i] = (1.0 - ai)*dhdxf[i]
                itpl_mat[k,i] = (1.0 - ai)
                #jach[k,0] = ai*dhdxf[0]
                itpl_mat[k,0] = ai
        jach = itpl_mat @ dhdxf
        return jach

    def dhdx(self, x):
        if self.operator == "linear":
            return np.diag(np.ones(x.size))
            #return np.ones(x.size)
        elif self.operator == "quadratic":
            return np.diag(2 * x)
            #return 2 * x
        elif self.operator == "cubic":
            return np.diag(3 * x**2)
            #return 3 * x**2
        elif self.operator == "quartic":
            return np.diag(4 * x**3)
            #return 4 * x**3
        elif self.operator == "quadratic-nodiff":
            return np.diag(np.where(x >= 0.5, 2*x, -2*x))
            #return np.where(x >= 0.5, 2*x, -2*x)
        elif self.operator == "cubic-nodiff":
            return np.diag(np.where(x >= 0.5, 3*x**2, -3*x**2))
            #return np.where(x >= 0.5, 3*x**2, -3*x**2)
        elif self.operator == "quartic-nodiff":
            return np.diag(np.where(x >= 0.5, 4*x**3, -4*x**3))
            #return np.where(x >= 0.5, 4*x**3, -4*x**3)
        elif self.operator == "test":
            return np.diag(0.5+0.5*self.gamma*np.power((0.1*np.abs(x)), (self.gamma-1)))
            #return 0.5+0.5*self.gamma*np.power((0.1*np.abs(x)), (self.gamma-1))
        elif self.operator == "abs":
            return np.diag(x/np.abs(x))
            #return x/np.abs(x)
        elif self.operator == "hint":
            dhdx = np.zeros((x.size, x.size))
            val = 1.0/7.0
            for i in range(dhdx.shape[0]):
                for j in range(i-3, i+4):
                    if j < 0:
                        jj = j + x.size
                        dhdx[i, jj] = val
                    elif j >= x.size:
                        jj = j - x.size
                        dhdx[i, jj] = val
                    else:
                        dhdx[i, j] = val
            return dhdx
    def add_noise(self, x):
# numpy 1.17.0 or later
#    return x + rng.normal(0, mu=sigma, size=x.size)
        #np.random.seed(514)
        return x + random.normal(0, scale=self.sigma, size=x.size).reshape(x.shape)

    def itpl1d(self, ri, x):
        i = math.floor(ri)
        ai = ri - float(i)
        if i < len(x) - 1:
            return (1.0 - ai)*x[i] + ai*x[i+1]
        else:
            return (1.0 - ai)*x[i] + ai*x[0]