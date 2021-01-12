import sys
import logging
from logging.config import fileConfig
import random
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('anl')

class EnKF():

    def __init__(self, da, obs, infl, lsig, model="model"):
        self.da = da # DA type (ETKF, PO, SRF, or LETKF)
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.infl_parm = infl # inflation parameter
        self.lsig = lsig # localization parameter
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.da} op={self.op} sig={self.sig} infl_parm={self.infl_parm} l_sig={self.lsig}")
        
    def __call__(self, xb, pf, y, save_hist=False, save_dh=False, \
        infl=False, loc=False, tlm=True, \
        icycle=0):
        xf = xb[:, 1:]
        xf_ = np.mean(xf, axis=1)
        JH = self.obs.dhdx(xf_)
        R, dum1, dum2 = self.obs.set_r(y.size)
        nmem = xf.shape[1]
        dxf = xf - xf_[:,None]
        alpha = self.infl_parm # inflation parameter
        if infl:
            if self.da != "letkf":
                logger.info("==inflation==, alpha={}".format(self.infl_parm))
                dxf *= alpha
        pf = dxf @ dxf.T / (nmem-1)
        if tlm:
            dy = JH @ dxf
        else:
            dy = self.obs.h_operator(xf) - self.obs.h_operator(xf_)[:, None]
        d = y - self.obs.h_operator(xf_)
        if save_dh:
            np.save("{}_dh_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dy)
            np.save("{}_d_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), d)
        logger.info("save_dh={} cycle{}".format(save_dh, icycle))

        #if loc: # B-localization
        #    if da == "etkf":
        #        print("==B-localization==")
        #        dist, l_mat = loc_mat(sigma=2.0, nx=xf_.size, ny=xf_.size)
        #        pf = pf * l_mat
        if self.da == "etkf":
            K1 = dxf @ dy.T / (nmem-1)
            K2 = dy @ dy.T / (nmem-1) + R
            #K = dxf @ dy.T @ la.inv(dy @ dy.T + (nmem-1)*R)
            eigK, vK = la.eigh(K2)
            logger.info("eigenvalues of K2")
            logger.info(eigK)
            K2inv = la.inv(K2)
            K = K1 @ K2inv
            np.save("{}_K_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K)
            if loc: # K-localization
                logger.info("==K-localization==, lsig={}".format(self.lsig))
                dist, l_mat = self.loc_mat(sigma=self.lsig, nx=xf_.size, ny=y.size)
                K = K * l_mat
                np.save("{}_Kloc_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K)
            xa_ = xf_ + K @ d
            if save_dh:
                np.save("{}_dx_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K@d)
        
            TT = la.inv( np.eye(nmem) + dy.T @ la.inv(R) @ dy / (nmem-1) )
            lam, v = la.eigh(TT)
            D = np.diag(np.sqrt(lam))
            T = v @ D

            dxa = dxf @ T
            xa = dxa + xa_[:,None]
            if save_dh:
                ua = np.zeros((xa_.size,nmem+1))
                ua[:,0] = xa_
                ua[:,1:] = xa
                np.save("{}_ua_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), ua)
            pa = dxa@dxa.T/(nmem-1)

        elif self.da=="po":
            Y = np.zeros((y.size,nmem))
            #err = np.random.normal(0, scale=self.sig, size=Y.size)
            mu = np.zeros(y.size)
            sigr = np.eye(y.size)*self.sig
            err = np.random.multivariate_normal(mu,sigr,nmem).T
            #err_ = np.mean(err.reshape(Y.shape), axis=1)
            #Y = y[:,None] + err.reshape(Y.shape)
            Y = y[:,None] + err
            #d_ = y + err_ - self.obs.h_operator(xf_)
            d_ = y - self.obs.h_operator(xf_)
            K1 = dxf @ dy.T / (nmem-1)
            K2 = dy @ dy.T / (nmem-1) + R
            eigK, vK = la.eigh(K2)
            logger.info("eigenvalues of K2")
            logger.info(eigK)
            K2inv = la.inv(K2)
            K = K1 @ K2inv
            #K = dxf @ dy.T @ la.inv(dy @ dy.T + (nmem-1)*R)
            if loc:
                logger.info("==localization==")
                dist, l_mat = self.loc_mat(sigma=self.lsig, nx=xf_.size, ny=y.size)
                K = K * l_mat
            xa_ = xf_ + K @ d_
            if tlm:
                xa = xf + K @ (Y - JH @ xf)
                pa = (np.eye(xf_.size) - K @ JH) @ pf
            else:
                HX = self.obs.h_operator(xf)
                xa = xf + K @ (Y - HX)
                pa = pf - K @ dy @ dxf.T / (nmem-1)

        elif self.da=="srf":
            I = np.eye(xf_.size)
            p0 = np.zeros_like(pf)
            p0 = pf[:,:]
            dx0 = np.zeros_like(dxf)
            dx0 = dxf[:,:]
            x0_ = np.zeros_like(xf_)
            x0_ = xf_[:]
            if loc:
                logger.info("==localization==")
                dist, l_mat = self.loc_mat(sigma=self.lsig, nx=xf_.size, ny=y.size)
            for i in range(y.size):
                hrow = JH[i].reshape(1,-1)
                d1 = hrow @ p0 @ hrow.T + self.sig*self.sig
                k1 = p0 @ hrow.T /d1
                k1_ = k1 / (1.0 + self.sig/np.sqrt(d1))
                if loc:
                    k1_ = k1_ * l_mat[:,i].reshape(k1_.shape)
                xa_ = x0_.reshape(k1_.shape) + k1_ * (y[i] - hrow@x0_)
                dxa = (I - k1_@hrow) @ dx0
                pa = dxa@dxa.T/(nmem-1)

                x0_ = xa_[:]
                dx0 = dxa[:,:]
                p0 = pa[:,:]
            xa = dxa + xa_
            xa_ = np.squeeze(xa_)

        elif self.da=="letkf":
            sigma = 7.5
            r0 = 100.0 # all
            if loc:
                logger.info("==localized r0==")
                r0 = 5.0
            nx = xf_.size
            dist, l_mat = self.loc_mat(sigma, nx, ny=y.size)
            print(dist[0])
            xa = np.zeros_like(xf)
            xa_ = np.zeros_like(xf_)
            dxa = np.zeros_like(dxf)
            E = np.eye(nmem)
            hx = self.obs.h_operator(xf_)
            if infl:
                logger.info("==inflation==")
                E /= alpha
            for i in range(nx):
                far = np.arange(y.size)
                far = far[dist[i]>r0]
                logger.debug("number of assimilated obs.={}".format(y.size - len(far)))
                yi = np.delete(y,far)
                if tlm:
                    Hi = np.delete(JH,far,axis=0)
                    di = yi - Hi @ xf_
                    dyi = Hi @ dxf
                else:
                    hxi = np.delete(hx,far)
                    di = yi - hxi
                    dyi = np.delete(dy,far,axis=0)
                Ri = np.delete(R,far,axis=0)
                Ri = np.delete(Ri,far,axis=1)
                if loc:
                    logger.info("==localization==")
                    diagR = np.diag(Ri)
                    l = np.delete(l_mat[i],far)
                    Ri = np.diag(diagR/l)
                R_inv = la.inv(Ri)
            
                A = (nmem-1)*E + dyi.T @ R_inv @ dyi
                lam,v = la.eigh(A)
                D_inv = np.diag(1.0/lam)
                pa_ = v @ D_inv @ v.T
            
                xa_[i] = xf_[i] + dxf[i] @ pa_ @ dyi.T @ R_inv @ di
                sqrtPa = v @ np.sqrt(D_inv) @ v.T * np.sqrt(nmem-1)
                dxa[i] = dxf[i] @ sqrtPa
                xa[i] = np.full(nmem,xa_[i]) + dxa[i]
            pa = dxa@dxa.T/(nmem-1)
        
        if save_dh:
            np.save("{}_pa_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), pa)

        innv = y - self.obs.h_operator(xa_)
        p = innv.size
        G = JH @ pf @ JH.T + R 
        chi2 = innv.T @ la.inv(G) @ innv / p
        ds = self.dof(dy,nmem)
        logger.info("dof={}".format(ds))
        
        u = np.zeros_like(xb)
        u[:, 0] = xa_
        u[:, 1:] = xa
        return u, pa, chi2, ds

    def loc_mat(self, sigma, nx, ny):
        dist = np.zeros((nx,ny))
        l_mat = np.zeros_like(dist)
        for j in range(nx):
            for i in range(ny):
                dist[j,i] = min(abs(j-i),nx-abs(j-i))
        d0 = 2.0 * np.sqrt(10.0/3.0) * sigma
        l_mat = np.exp(-dist**2/(2.0*sigma**2))
        l_mat[dist>d0] = 0
        return dist, l_mat 

    def dof(self, dy, nmem):
        zmat = dy / self.sig
        u, s, vt = la.svd(zmat)
        ds = np.sum(s**2/(1.0+s**2))#/(nmem-1)
        return ds