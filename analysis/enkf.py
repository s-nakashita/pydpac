import sys
import logging
from logging.config import fileConfig
import random
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
from .chi_test import Chi

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('anl')

class EnKF():

    def __init__(self, da, obs, infl, lsig,
                 linf, lloc, ltlm, model="model"):
        self.da = da # DA type (ETKF, PO, SRF, or LETKF)
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.infl_parm = infl # inflation parameter
        self.lsig = lsig # localization parameter
        self.linf = linf # True->Apply inflation False->Not apply
        self.lloc = lloc # True->Apply localization False->Not apply
        self.ltlm = ltlm # True->Use tangent linear approximation False->Not use
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.da} op={self.op} sig={self.sig} infl_parm={self.infl_parm} lsig={self.lsig}")
        logger.info(f"linf={self.linf} lloc={self.lloc} ltlm={self.ltlm}")
        
    def __call__(self, xb, pf, y, yloc, save_hist=False, save_dh=False, icycle=0):
        xf = xb[:]
        xf_ = np.mean(xf, axis=1)
        #JH = self.obs.dhdx(xf_)
        JH = self.obs.dh_operator(yloc, xf_)
        R, rmat, rinv = self.obs.set_r(y.size)
        nmem = xf.shape[1]
        chi2_test = Chi(y.size, nmem, rmat)
        dxf = xf - xf_[:,None]
        xloc = np.arange(xf_.size)
        alpha = self.infl_parm # inflation parameter
        if self.linf and self.da != "letkf" and self.da != "etkf":
            logger.info("==inflation==, alpha={}".format(self.infl_parm))
            dxf *= alpha
            xf = xf_[:, None] + dxf
        pf = dxf @ dxf.T / (nmem-1)
        logger.info("pf max={} min={}".format(np.max(pf),np.min(pf)))
        if save_dh:
            np.save("{}_pf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), pf)
            np.save("{}_spf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxf)
            #if self.lloc and self.da != "letkf":
            #    logger.info("==B-localization==, lsig={}".format(self.lsig))
            #    dxf = self.pfloc(dxf, save_dh, icycle)
            #    pf = dxf @ dxf.T / (nmem-1)
            #    xf = xf_[:, None] + dxf
        if self.ltlm:
            dy = JH @ dxf
        else:
            #dy = self.obs.h_operator(xf) - self.obs.h_operator(xf_)[:, None]
            #dy = self.obs.h_operator(xf) - np.mean(self.obs.h_operator(xf), axis=1)[:, None]
            dy = self.obs.h_operator(yloc, xf) - np.mean(self.obs.h_operator(yloc, xf), axis=1)[:, None]
        #d = y - self.obs.h_operator(xf_)
        #d = y - np.mean(self.obs.h_operator(xf), axis=1)
        d = y - np.mean(self.obs.h_operator(yloc, xf), axis=1)
        if save_dh:
            np.save("{}_dh_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dy)
            np.save("{}_d_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), d)
        logger.info("save_dh={} cycle{}".format(save_dh, icycle))

        if self.da == "etkf":
            """
            K1 = dxf @ dy.T / (nmem-1)
            K2 = dy @ dy.T / (nmem-1) + R
            #K = dxf @ dy.T @ la.inv(dy @ dy.T + (nmem-1)*R)
            eigK, vK = la.eigh(K2)
            logger.info("eigenvalues of K2")
            logger.info(eigK)
            K2inv = la.inv(K2)
            K = K1 @ K2inv
            if save_dh:
                np.save("{}_K_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K)
            if self.lloc: # K-localization
                logger.info("==K-localization==, lsig={}".format(self.lsig))
                dist, l_mat = self.loc_mat(sigma=self.lsig, nx=xf_.size, ny=y.size)
                K = K * l_mat
                if save_dh:
                    np.save("{}_Kloc_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K)
            
            #xa_ = xf_ + K @ d
            
            if save_dh:
                np.save("{}_dx_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K@d)
            """
            E = np.eye(nmem)
            if self.linf:
                logger.info("==inflation==, alpha={}".format(alpha))
                E /= alpha
            A = (nmem-1)*E + dy.T @ rinv @ dy
            #TT = la.inv( np.eye(nmem) + dy.T @ rinv @ dy / (nmem-1) )
            lam, v = la.eigh(A)
            Dinv = np.diag(1.0/lam)
            TT = v @ Dinv @ v.T
            T = np.sqrt(nmem-1) * v @ np.sqrt(Dinv) @ v.T

            K = dxf @ TT @ dy.T @ rinv
            if save_dh:
                np.save("{}_K_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K)
                np.save("{}_dx_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K@d)
            if self.lloc: # K-localization
                logger.info("==K-localization==, lsig={}".format(self.lsig))
                dist, l_mat = self.k_loc(sigma=self.lsig, obsloc=yloc, xloc=xloc)
                K = K * l_mat
                if save_dh:
                    np.save("{}_Kloc_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K)
                    np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), l_mat)
            xa_ = xf_ + K @ d
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
            np.random.seed(514)
            mu = np.zeros(y.size)
            sigr = np.eye(y.size)*self.sig
            err = np.random.multivariate_normal(mu,sigr,nmem).T
            err_ = np.mean(err, axis=1)
            #Y = y[:,None] + err.reshape(Y.shape)
            Y = y[:,None] + err
            #d_ = y + err_ - self.obs.h_operator(xf_)
            #d_ = y - self.obs.h_operator(xf_)
            d_ = d + err_
            K1 = dxf @ dy.T / (nmem-1)
            K2 = dy @ dy.T / (nmem-1) + R
            eigK, vK = la.eigh(K2)
            logger.info("eigenvalues of K2")
            logger.info(eigK)
            K2inv = la.inv(K2)
            K = K1 @ K2inv
            #K = dxf @ dy.T @ la.inv(dy @ dy.T + (nmem-1)*R)
            if self.lloc:
                logger.info("==K-localization== lsig={}".format(self.lsig))
                dist, l_mat = self.k_loc(sigma=self.lsig, obsloc=yloc, xloc=xloc)
                K = K * l_mat
                if save_dh:
                    np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), l_mat)
            xa_ = xf_ + K @ d_
            if self.ltlm:
                xa = xf + K @ (Y - JH @ xf)
                #dxa = xa - xa_[:, None]
                #pa = (np.eye(xf_.size) - K @ JH) @ pf
            else:
                #HX = self.obs.h_operator(xf)
                HX = self.obs.h_operator(yloc, xf)
                xa = xf + K @ (Y - HX)
                #dxa = xa - xa_[:, None]
                #pa = pf - K @ dy @ dxf.T / (nmem-1)
            dxa = xa - xa_[:, None]
            pa = dxa @ dxa.T /(nmem-1)

        elif self.da=="srf":
            I = np.eye(xf_.size)
            p0 = np.zeros_like(pf)
            p0 = pf[:,:]
            dx0 = np.zeros_like(dxf)
            dx0 = dxf[:,:]
            dy0 = np.zeros_like(dy)
            dy0 = dy[:,:]
            x0_ = np.zeros_like(xf_)
            x0_ = xf_[:]
            d0 = np.zeros_like(d)
            d0 = d[:]
            if self.lloc:
                logger.info("==K-localization== lsig={}".format(self.lsig))
                dist, l_mat = self.k_loc(sigma=self.lsig, obsloc=yloc, xloc=xloc)
                if save_dh:
                    np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), l_mat)
            for i in range(y.size):
                #hrow = JH[i].reshape(1,-1)
                dyi = dy0[i].reshape(1,-1)
                #d1 = hrow @ p0 @ hrow.T + self.sig*self.sig
                d1 = dyi @ dyi.T + self.sig*self.sig
                #k1 = p0 @ hrow.T /d1
                k1 = dx0 @ dyi.T /d1
                k1_ = k1 / (1.0 + self.sig/np.sqrt(d1))
                if self.lloc: # K-localization
                    k1_ = k1_ * l_mat[:,i].reshape(k1_.shape)
                #xa_ = x0_.reshape(k1_.shape) + k1_ * (y[i] - hrow@x0_)
                xa_ = x0_.reshape(k1_.shape) + k1_ * d0[i]
                #dxa = (I - k1_@hrow) @ dx0
                dxa = dx0 - k1_@ dyi
                pa = dxa@dxa.T/(nmem-1)

                x0_ = xa_[:]
                dx0 = dxa[:,:]
                #dy0 = self.obs.h_operator(x0_+dx0) - self.obs.h_operator(x0_)
                #d0 = y[:, None] - self.obs.h_operator(x0_)
                x0 = x0_ + dx0
                #dy0 = self.obs.h_operator(x0) - np.mean(self.obs.h_operator(x0), axis=1)[:, None]
                dy0 = self.obs.h_operator(yloc, x0) - np.mean(self.obs.h_operator(yloc, x0), axis=1)[:, None]
                #d0 = y - np.mean(self.obs.h_operator(x0), axis=1)
                d0 = y - np.mean(self.obs.h_operator(yloc, x0), axis=1)
                p0 = pa[:,:]
            xa = dxa + xa_
            xa_ = np.squeeze(xa_)

        elif self.da=="letkf":
            #sigma = 7.5
            sigma = self.lsig
            #r0 = 100.0 # all
            #r0 = 10.0 # half
            #if self.lloc:
            #    logger.info("==localized r0==")
            #    r0 = 5.0
            nx = xf_.size
            #dist, l_mat = self.loc_mat(sigma, nx, ny=y.size)
            #print(dist[0])
            xa = np.zeros_like(xf)
            xa_ = np.zeros_like(xf_)
            dxa = np.zeros_like(dxf)
            E = np.eye(nmem)
            #hx = self.obs.h_operator(xf_)
            if self.linf:
                logger.info("==inflation==, alpha={}".format(alpha))
                E /= alpha
            for i in range(nx):
                far, Rwf_loc = self.r_loc(sigma, nx, yloc, float(i))
                #far = np.arange(y.size)
                #far = far[dist[i]>r0]
                logger.debug("number of assimilated obs.={}".format(y.size - len(far)))
                yi = np.delete(y,far)
                #if self.ltlm:
                #    Hi = np.delete(JH,far,axis=0)
                #    di = yi - Hi @ xf_
                #    #dyi = Hi @ dxf
                #else:
                #    hxi = np.delete(hx,far)
                #    di = yi - hxi
                di = np.delete(d,far)
                dyi = np.delete(dy,far,axis=0)
                if self.lloc:
                    logger.info("==R-localization==, lsig={}".format(self.lsig))
                    diagR = np.diag(R)
                    Ri = np.diag(diagR/Rwf_loc)
                else:
                    Ri = R[:,:]
                Ri = np.delete(Ri,far,axis=0)
                Ri = np.delete(Ri,far,axis=1)
                #if self.lloc:
                #    logger.info("==R-localization==")
                #    diagR = np.diag(Ri)
                #    l = np.delete(l_mat[i],far)
                #    Ri = np.diag(diagR/l)
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

        if self.ltlm:
            #dh = self.obs.dhdx(xa_) @ dxf / np.sqrt(nmem-1)
            dh = self.obs.dh_operator(yloc, xa_) @ dxf / np.sqrt(nmem-1)
        else:
            #dy = self.obs.h_operator(xa) - self.obs.h_operator(xa_)[:, None]
            x1 = xa_[:, None] + dxf / np.sqrt(nmem-1)
            #dh = self.obs.h_operator(x1) - self.obs.h_operator(xa_)[:, None]
            dh = self.obs.h_operator(yloc, x1) - self.obs.h_operator(yloc, xa_)[:, None]
            #dh = self.obs.h_operator(x1) - np.mean(self.obs.h_operator(x1), axis=1)[:, None]
        zmat = rmat @ dh
        #d = y - self.obs.h_operator(xa_)
        d = y - self.obs.h_operator(yloc, xa_)
        #d = y - np.mean(self.obs.h_operator(xa))
        innv, chi2 = chi2_test(zmat, d)
        ds = self.dof(dy,nmem)
        logger.info("dof={}".format(ds))
        
        u = np.zeros_like(xb)
        u = xa[:,:]
        #u[:, 0] = xa_
        #u[:, 1:] = xa
        return u, pa, innv, chi2, ds

    def b_loc(self, sigma, nx, ny):
        if sigma < 0.0:
            loc_scale = 1.0
        else:
            loc_scale = sigma
        dist = np.zeros((nx,ny))
        l_mat = np.zeros_like(dist)
        # distance threshold
        dist0 = loc_scale * np.sqrt(10.0/3.0) * 2.0
        logger.debug(dist0)
        for j in range(nx):
            for i in range(ny):
                dist[j,i] = min(abs(j-i),nx-abs(j-i))
        l_mat = np.exp(-0.5*(dist/loc_scale)**2)
        logger.debug(dist[dist>dist0])
        l_mat[dist>dist0] = 0
        return dist, l_mat

    def k_loc(self, sigma, obsloc, xloc):
        if sigma < 0.0:
            loc_scale = 1.0
        else:
            loc_scale = sigma
        nx = xloc.size
        nobs = obsloc.size
        dist = np.zeros((nx, nobs))
        # distance threshold
        dist0 = loc_scale * np.sqrt(10.0/3.0) * 2.0
        logger.debug(dist0)
        for j in range(nx):
            for i in range(nobs):
                dist[j, i] = min(abs(obsloc[i]-j), nx-abs(obsloc[i]-j))
        l_mat = np.exp(-0.5*(dist/loc_scale)**2)
        logger.debug(dist[dist>dist0])
        l_mat[dist>dist0] = 0
        return dist, l_mat        

    def r_loc(self, sigma, nx, obsloc, xloc):
        if sigma < 0.0:
            loc_scale = 1.0
        else:
            loc_scale = sigma
        nobs = obsloc.size
        far = np.arange(nobs)
        Rwf_loc = np.ones(nobs)

        # distance threshold
        dist0 = loc_scale * np.sqrt(10.0/3.0) * 2.0
        logger.debug(dist0)

        dist = np.zeros(nobs)
        for k in range(nobs):
            dist[k] = min(abs(obsloc[k] - xloc), nx-abs(obsloc[k] - xloc))
        far = far[dist>dist0]
        logger.debug(far)
        Rwf_loc = np.exp(-0.5*(dist/loc_scale)**2)
        return far, Rwf_loc

    def pfloc(self, dxf, save_dh, icycle):
        nmem = dxf.shape[1]
        pf = dxf @ dxf.T / (nmem - 1)
        #if save_dh:
        #    np.save("{}_pf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), pf)
        #    np.save("{}_spf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), sqrtpf)
        dist, l_mat = self.b_loc(self.lsig, pf.shape[0], pf.shape[1])
        if save_dh:
            np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), l_mat)
        pf = pf * l_mat
        logger.info("lpf max={} min={}".format(np.max(pf),np.min(pf)))
        if save_dh:
            np.save("{}_lpf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), pf)
        lam, v = la.eigh(pf)
        lam = lam[::-1]
        v = v[:,::-1]
        #lam = lam[::-1]
        if save_dh:
            np.save("{}_lpfeig_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), lam)
        #lam[nmem:] = 0.0
        logger.info("eigen value = {}".format(lam))
        pf = v[:,:nmem] @ np.diag(lam[:nmem]) @ v[:,:nmem].T
        dxf = v[:,:nmem] @ np.diag(np.sqrt(lam[:nmem])) * np.sqrt(nmem-1)
        #spf0 = v @ np.diag(np.sqrt(lam)) @ v.T
        #spf = spf0[:,:nmem]
        logger.info("pf - spf@spf.T={}".format(np.mean(pf - dxf@dxf.T/(nmem-1))))
        if save_dh:
            np.save("{}_lpfr_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), pf)
            np.save("{}_lspf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxf)
        return dxf

    def dof(self, dy, nmem):
        zmat = dy / self.sig
        u, s, vt = la.svd(zmat)
        ds = np.sum(s**2/(1.0+s**2))#/(nmem-1)
        return ds