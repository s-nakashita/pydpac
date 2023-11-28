import numpy as np
# numpy 1.17.0 or later
from numpy.random import default_rng
rng = default_rng()
#from numpy import random
import math
import logging
from logging.config import fileConfig

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')

class Obs():
    def __init__(self, operator, sigma, nvars=1, ndims=1, ni=0, nj=0, ix=None, jx=None, icyclic=True, jcyclic=True):
        self.operator = operator
        self.sigma = sigma
        self.gamma = 3
        logger.info(f"operator={self.operator}, obserr={self.sigma}")#, gamma={self.gamma}")
        self.nvars = nvars
        logger.info(f"nvars={self.nvars}")
        self.ndims = ndims
        self.ix = ix
        self.icyclic = icyclic
        if self.ndims == 2:
            self.ni, self.nj = ni,nj
            self.jx = jx
            self.jcyclic = jcyclic
            logger.info(f"ni={self.ni} nj={self.nj}")

    def get_op(self):
        return self.operator

    def get_sig(self):
        return self.sigma

    def set_r(self, obsloc):
        if self.ndims==1:
            p = obsloc.size
        else:
            p = obsloc.shape[0]
        r = np.diag(np.ones(p)*self.sigma*self.sigma)
        rmat = np.diag(np.ones(p) / self.sigma)
        rinv = rmat.transpose() @ rmat
        return r, rmat, rinv

    def h_operator(self, obsloc, x):
        #logger.debug(f"x={x}")
        if self.ndims==1:
            nobs = obsloc.size
        else:
            nobs = obsloc.shape[0]
        #logger.debug(f"nobs={nobs}")
        if x.ndim == 1:
            if self.nvars==1:
                hxf = self.hx(x)
            else:
                hxf = self.hx(x.reshape(self.nvars,-1))
            #logger.debug(f"hx={hxf}")
            nx = int(hxf.size / self.nvars)
            obs = np.zeros(nobs)
            for k in range(nobs):
                if self.nvars==1:
                    if self.ndims==1:
                        obs[k] = self.itpl1d(obsloc[k], hxf)
                    elif self.ndims==2:
                        obs[k] = self.itpl2d(obsloc[k,0],obsloc[k,1],hxf)
                else:
                    ivar = int(obsloc[k,0])
                    if self.ndims==1:
                        obs[k] = self.itpl1d(obsloc[k,1], hxf[ivar,:])
                    elif self.ndims==2:
                        obs[k] = self.itpl2d(obsloc[k,1],obsloc[k,2],hxf[ivar,:])
        else:
            nens = x.shape[1]
            if self.nvars==1:
                hxf = self.hx(x)
            else:
                hxf = self.hx(x.reshape(self.nvars,-1,nens))
            #logger.debug(f"hx={hxf}")
            nx = int(hxf.shape[0] / self.nvars)
            obs = np.zeros((nobs, nens))
            for k in range(nobs):
                for j in range(nens):
                    if self.nvars == 1:
                        if self.ndims==1:
                            obs[k,j] = self.itpl1d(obsloc[k], hxf[:,j])
                        elif self.ndims==2:
                            obs[k,j] = self.itpl2d(obsloc[k,0],obsloc[k,1], hxf[:,j])
                    else:
                        ivar = int(obsloc[k,0])
                        if self.ndims==1:
                            obs[k,j] = self.itpl1d(obsloc[k,1], hxf[ivar,:,j])
                        elif self.ndims==2:
                            obs[k,j] = self.itpl2d(obsloc[k,1],obsloc[k,2], hxf[ivar,:,j])
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
        if self.ndims==1:
            nobs = obsloc.size
        else:
            nobs = obsloc.shape[0]
        if self.ndims == 1:
            nx = x.size
        else:
            nx = int(x.size/self.nvars)
        jach = np.zeros((nobs, nx*self.nvars))
        itpl_mat = np.zeros((nobs, nx*self.nvars))
        dhdxf = self.dhdx(x)
        if self.ndims==1:
            for k in range(nobs):
                if self.nvars==1:
                    ivar=0
                    ri = obsloc[k]
                else:
                    ivar = int(obsloc[k,0])
                    ri = obsloc[k,1]
                i = math.floor(ri)
                ai = ri - float(i)
                if i < nx-1:
                    #jach[k,i] = (1.0 - ai)*dhdxf[i]
                    itpl_mat[k,ivar*nx+i] = (1.0 - ai)
                    #jach[k,i+1] = ai*dhdxf[i+1]
                    itpl_mat[k,ivar*nx+i+1] = ai
                else:
                    #jach[k,i] = (1.0 - ai)*dhdxf[i]
                    itpl_mat[k,ivar*nx+i] = (1.0 - ai)
                    #jach[k,0] = ai*dhdxf[0]
                    itpl_mat[k,ivar*nx] = ai
        elif self.ndims==2:
            for k in range(nobs):
                if self.nvars==1:
                    ivar=0
                    ri, rj = obsloc[k,:]
                else:
                    ivar = int(obsloc[k,0])
                    ri, rj = obsloc[k,1:]
                i = math.floor(ri)
                j = math.floor(rj)
                ai = ri - float(i)
                aj = rj - float(j)
                ij = self.nj*i + j
                itpl_mat[k,ivar*nx+ij] = (1.0-ai)*(1.0-aj)
                itpl_mat[k,ivar*nx+ij+1] = ai*(1.0-aj)
                itpl_mat[k,ivar*nx+ij+self.ni] = (1.0-ai)*aj
                itpl_mat[k,ivar*nx+ij+self.ni+1] = ai*aj
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
        return x + rng.normal(0, scale=self.sigma, size=x.size).reshape(x.shape)
        #np.random.seed(514)
        #return x + random.normal(0, scale=self.sigma, size=x.size).reshape(x.shape)

    def itpl1d(self, ri, x):
        if self.ix is None:
            ix = np.arange(len(x))
        else:
            ix = self.ix
        dx = float(ix[1]) - float(ix[0])
        ii = math.floor(ri)
        i = np.argmin(np.abs(ix - ii))
        if ix[i] > ii:
            i = i - 1
        ai = (ri - float(ix[i]))/dx
        #logger.debug(f"ri={ri} i={i} ai={ai}")
        if i < len(x) - 1:
            return (1.0 - ai)*x[i] + ai*x[i+1]
        else:
            if self.icyclic:
                return (1.0 - ai)*x[i] + ai*x[0]
            else:
                return (1.0 - ai)*x[i]

    def itpl2d(self, ri, rj, x):
        x2d = x.reshape(self.ni,self.nj)
        if self.ix is None:
            ix = np.arange(self.ni)
        else:
            ix = self.ix
        if self.jx is None:
            jx = np.arange(self.nj)
        else:
            jx = self.jx
        ii = math.floor(ri)
        jj = math.floor(rj)
        ai = ri - float(ii)
        aj = rj - float(jj)
        i = np.argmin(np.abs(ix - ii))
        j = np.argmin(np.abs(jx - jj))
        logger.debug(f"ri={ri} i={i} ai={ai}")
        logger.debug(f"rj={rj} j={j} aj={aj}")
        if i+1>=self.ni and j+i>=self.nj:
            y = x2d[i,j]
        elif i+1>=self.ni:
            y = (1.0-aj)*x2d[i,j]+aj*x2d[i,j+1]
        elif j+1>=self.nj:
            y = (1.0-ai)*x2d[i,j]+ai*x2d[i+1,j]
        else:
            y =  (1.0-ai)*(1.0-aj)*x2d[i,j] \
                +     ai *(1.0-aj)*x2d[i+1,j] \
                +(1.0-ai)*     aj *x2d[i,j+1] \
                +     ai *     aj *x2d[i+1,j+1]
        return y
