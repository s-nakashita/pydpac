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

class EnKS():
# Hunt et al. 2004: "Four-dimensional ensemble Kalman filtering," Tellus, 56A, 273-277
# Fertig et al. 2007: "A comparative study of 4D-VAR and a 4D ensemble Kalman filter: perfect model simulations with Lorenz-96," Tellus, 59A, 96-100
    def __init__(self, da, state_size, nmem, obs, step, nt, window_l, 
                linf=False, infl_parm=1.0, 
                iloc=None, lsig=-1.0, calc_dist=None, calc_dist1=None,
                ltlm=False, model="model"):
        # necessary parameters
        self.da = da[2:] # DA type (prefix 4d + ETKF, PO, SRF, or LETKF)
        self.ndim = state_size # state size
        self.nmem = nmem # ensemble size
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.step = step # forward model
        self.nt = nt     # assimilation interval
        self.window_l = window_l # assimilation window length
        # optional parameters
        # inflation
        self.linf = linf # True->Apply inflation False->Not apply
        self.infl_parm = infl_parm # inflation parameter
        # localization
        self.iloc = iloc # iloc = None->No localization
                         #      = 0   ->R-localization
                         #      = 1   ->Eigen value decomposition of localized Pf
                         #      = 2   ->Modulated ensemble
        self.lsig = lsig # localization parameter
        if calc_dist is None:
            def calc_dist(self, i):
                dist = np.zeros(self.ndim)
                for j in range(self.ndim):
                    dist[j] = min(abs(j-i),self.ndim-abs(j-i))
                return dist
        else:
            self.calc_dist = calc_dist # distance calculation routine
        if calc_dist1 is None:
            def calc_dist1(self, i, j):
                return min(abs(j-i),self.ndim-abs(j-i))
        else:
            self.calc_dist1 = calc_dist1 # distance calculation routine
        # tangent linear
        self.ltlm = ltlm # True->Use tangent linear approximation False->Not use
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.da} op={self.op} sig={self.sig} infl_parm={self.infl_parm} lsig={self.lsig}")
        logger.info(f"linf={self.linf} iloc={self.iloc} ltlm={self.ltlm}")
        if self.iloc == 1 or self.iloc == 2:
            self.l_mat, self.l_sqrt, self.nmode, self.enswts\
                = self.b_loc(self.lsig, self.ndim, self.ndim)
            np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), self.l_mat)
        logger.info(f"nt={self.nt} window_l={self.window_l}")

    def calc_pf(self, xf, pa, cycle):
        dxf = xf - np.mean(xf,axis=1)[:, None]
        pf = dxf @ dxf.transpose() / (self.nmem-1)
        return pf
        
    def __call__(self, xf, pf, y, yloc, save_hist=False, save_dh=False, icycle=0):
        #xf = xb[:]
        xf_ = np.mean(xf, axis=1)
        logger.debug(f"obsloc={yloc.shape}")
        logger.debug(f"obssize={y.shape}")
        R, rmat, rinv = self.obs.set_r(yloc[0])
        nmem = xf.shape[1]
        chi2_test = Chi(y.size, nmem, rmat)
        dxf = xf - xf_[:,None]
        logger.debug(xf.shape)
        xloc = np.arange(xf_.size)

        pf = dxf @ dxf.T / (nmem-1)
        logger.info("pf max={} min={}".format(np.max(pf),np.min(pf)))
        if save_dh:
            np.save("{}_pf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), pf)
            np.save("{}_spf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxf)
        dxf_orig = dxf.copy()
        xf_orig = xf.copy()
        if (self.iloc == 1 or self.iloc == 2) and self.da != "letkf":
            logger.info("==B-localization==, lsig={}".format(self.lsig))
            if self.iloc == 1:
                dxf = self.pfloc(dxf_orig, save_dh, icycle)
            elif self.iloc == 2:
                dxf = self.pfmod(dxf_orig, save_dh, icycle)
            nmem2 = dxf.shape[1]
            pf = dxf @ dxf.T / (nmem2-1)
            xf = xf_[:, None] + dxf
            logger.info("dxf.shape={}".format(dxf.shape))
        else:
            nmem2 = nmem
        
        xb = np.zeros_like(xf)
        xb = xf[:,:] 
        xb_ = np.zeros_like(xf_)
        xb_ = xf_[:] 
        dxb = np.zeros_like(dxf)
        dxb = dxf[:,:]
        if (self.iloc == 1 or self.iloc == 2):
            xb_orig = np.zeros_like(xf_orig)
            xb_orig = xf_orig[:,:]
            dxb_orig = np.zeros_like(dxf_orig)
            dxb_orig = dxf_orig[:,:] 
        bg = [] # background mean states
        #bge = [] # background ensemble states (no modulation)
        dy = [] # observation perturbations
        if (self.iloc == 1 or self.iloc == 2):
            dy_orig = [] # observation perturvations (no modulation)
        d  = [] # innovation vectors
        for l in range(min(self.window_l, y.shape[0])):
            bg.append(xb_)
            #if (self.iloc == 1 or self.iloc == 2):
            #    bge.append(xb_orig)
            #else:
            #    bge.append(xb)
            if self.ltlm:
                JH = self.obs.dh_operator(yloc[l], bg[l])
                dy.append(JH @ dxb)
                if (self.iloc == 1 or self.iloc == 2):
                    dy_orig.append(JH @ dxb_orig)
            else:
                dy.append(self.obs.h_operator(yloc[l], xb) - np.mean(self.obs.h_operator(yloc[l], xb), axis=1)[:, None])
                if (self.iloc == 1 or self.iloc == 2):
                    dy_orig.append(self.obs.h_operator(yloc[l], xb_orig) - np.mean(self.obs.h_operator(yloc[l], xb_orig), axis=1)[:, None])
            d.append(y[l] - np.mean(self.obs.h_operator(yloc[l], xb), axis=1))
            for k in range(self.nt):
                xb = self.step(xb)
                if (self.iloc == 1 or self.iloc == 2):
                    xb_orig = self.step(xb_orig)
            xb_ = np.mean(xb, axis=1)
            dxb = xb - xb_[:, None]
            if (self.iloc == 1 or self.iloc == 2):
                dxb_orig = xb_orig - np.mean(xb_orig, axis=1)[:, None]
        logger.debug(f"bg {len(bg)}")
        logger.debug(f"d {len(d)}")
        if save_dh:
            np.save("{}_dxf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxf)
            np.save("{}_dh_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dy)
            np.save("{}_d_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), d)
        logger.info("save_dh={} cycle{}".format(save_dh, icycle))

        alpha = self.infl_parm # inflation parameter
        if self.linf:
            if self.da != "letkf" and self.da != "etkf":
                logger.info("==inflation==, alpha={}".format(self.infl_parm))
                dxf *= alpha
            #xf = xf_[:, None] + dxf
        
        if self.da == "etkf":
            if self.linf:
                logger.info("==inflation==, alpha={}".format(alpha))
                A = np.eye(nmem2) * (nmem2-1) / alpha
            else:
                A = np.eye(nmem2) * (nmem2-1)
            for l in range(len(dy)):
                logger.debug(f"dy {dy[l].shape}")
                A = A + dy[l].T @ rinv @ dy[l]
            lam, v = la.eigh(A)
            Dinv = np.diag(1.0/lam)
            TT = v @ Dinv @ v.T
            T = v @ np.sqrt((nmem2-1)*Dinv) @ v.T
            xa_ = xf_
            for l in range(len(dy)):
                K = dxf @ TT @ dy[l].T @ rinv
            #if save_dh:
            #    np.save("{}_K_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K)
            #    np.save("{}_dx_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K@d)
                if self.iloc == 0: # K-localization
                    logger.info("==K-localization==, lsig={}".format(self.lsig))
                    l_mat = self.k_loc(sigma=self.lsig, obsloc=yloc[l], xloc=xloc)
                    K = K * l_mat
                #if save_dh:
                #    np.save("{}_Kloc_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K)
                #    np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), l_mat)
                xa_ = xa_ + K @ d[l]
            dxa = dxf @ T
            if self.iloc == 1 or self.iloc == 2:
                # random sampling
                ptrace = np.sum(np.diag(dxa @ dxa.T / (nmem2-1)))
                rvec = np.random.randn(nmem2, nmem)
                rvec_mean = np.mean(rvec, axis=0)
                rvec = rvec - rvec_mean[None, :]
                rvec_stdv = np.sqrt((rvec**2).sum(axis=0)/(nmem2-1))
                rvec = rvec / rvec_stdv[None, :]
                logger.debug("rvec={}".format(rvec[:, 0]))
                dxa = dxf @ T @ rvec / np.sqrt(nmem2-1)
                trace = np.sum(np.diag(dxa @ dxa.T / (nmem-1)))
                logger.info("standard deviation ratio = {}".format(np.sqrt(ptrace / trace)))
                if np.sqrt(ptrace / trace) > 1.0:
                    dxa *= np.sqrt(ptrace / trace)
            xa = dxa + xa_[:,None]

        elif self.da=="po":
            dxa = np.zeros_like(dxf_orig)
            dxa = dxf_orig[:,:]
            xa_ = np.zeros_like(xf_)
            xa_ = xf_[:]
            for l, obsloc, obs, dl, dyl \
                in zip(np.arange(len(dy)), yloc, y, d, dy):
                #Y = np.zeros((y[l].size,nmem))
                rs = np.random.RandomState()
                err = rs.standard_normal(size=(obs.size,nmem))
                err_var = np.sum((err - np.mean(err, axis=1)[:, None])**2, axis=1)/(nmem-1)
                err = self.sig * err / np.sqrt(err_var).reshape(-1,1)
                #mu = np.zeros(y[l].size)
                #sigr = np.eye(y[l].size)*self.sig
                #err = np.random.multivariate_normal(mu,sigr,nmem).T
                #err_ = np.mean(err, axis=1)
                #Y = y[:,None] + err.reshape(Y.shape)
                #Y = obs.reshape(-1,1) + err
                #d_ = d[l] + err_
                K1 = dxf @ dyl.T / (nmem2-1)
                K2 = dyl @ dyl.T / (nmem2-1) + R
                if logger.getEffectiveLevel() == logging.DEBUG:
                    eigK, vK = la.eigh(K2)
                    logger.debug("eigenvalues of K2")
                    logger.debug(eigK)
                K2inv = la.inv(K2)
                K = K1 @ K2inv
            #K = dxf @ dy.T @ la.inv(dy @ dy.T + (nmem-1)*R)
                if self.iloc == 0:
                    logger.info("==K-localization== lsig={}".format(self.lsig))
                    l_mat = self.k_loc(sigma=self.lsig, obsloc=obsloc, xloc=xloc)
                    K = K * l_mat
                    if save_dh:
                        np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), l_mat)
                logger.info(f"{l}th K max={np.max(K)} min={np.min(K)}")
                xa_ = xa_ + K @ dl
                #if self.ltlm:
                #    JH = self.obs.dh_operator(yloc[l], bg[l])
                #    xa = xa + K @ (Y - JH @ xf_orig)
                #    xa = xa + K @ (Y - JH @ bge[l])
                #else:
                #    HX = self.obs.h_operator(yloc[l], bge[l])
                #    xa = xa + K @ (Y - HX)
                if (self.iloc == 1 or self.iloc == 2):
                    dxa = dxa - K @ (dy_orig[l] - err)
                else:
                    dxa = dxa - K @ (dyl - err)
            xa = dxa + xa_[:, None]

        elif self.da=="srf":
            xa = xf
            xa_ = np.mean(xa, axis=1)
            dxa = dxf
            x0 = xa.copy()
            dx0 = dxa.copy()
            for l, obsloc, obs in zip(np.arange(len(dy)), yloc, y):
                if self.iloc == 0:
                    logger.info("==K-localization== lsig={}".format(self.lsig))
                    l_mat = self.k_loc(sigma=self.lsig, obsloc=obsloc, xloc=xloc)
                    if save_dh:
                        np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), l_mat)
                for i, obloc, ob in zip(np.arange(obs.size), obsloc, obs):
                    x0 = xa.copy()
                    for k in range(self.nt*l):
                        x0 = self.step(x0)
                    if self.ltlm:
                        dx0 = x0 - np.mean(x0, axis=1)[:, None]
                        dyi = self.obs.dh_operator(np.atleast_1d(obloc), x0) @ dx0
                    else:
                        dyi = self.obs.h_operator(np.atleast_1d(obloc), x0) - np.mean(self.obs.h_operator(np.atleast_1d(obloc), x0), axis=1)[:, None]
                    logger.debug(f"dyi.shape={dyi.shape}")
                    dymean = np.mean(self.obs.h_operator(np.atleast_1d(obloc), x0), axis=1)
                    d1 = dyi @ dyi.T / (nmem2-1) + self.sig*self.sig 
                    k1 = dxf @ dyi.T / (nmem2-1) / d1
                    logger.debug(f"k1.shape={k1.shape}")
                    if self.iloc == 0: # K-localization
                        k1 = k1 * l_mat[:,i].reshape(k1.shape)
                    k1_ = k1 * np.sqrt(d1) / (np.sqrt(d1) + self.sig)
                    xa_ = xa_.reshape(k1.shape) + k1 * (ob - dymean)
                    dxa = dxa - k1_@ dyi
                    xa = xa_ + dxa
                    
            if self.iloc == 1 or self.iloc == 2:
                # random sampling
                ptrace = np.sum(np.diag(dxa @ dxa.T / (nmem2-1)))
                rvec = np.random.randn(nmem2, nmem)
                rvec_mean = np.mean(rvec, axis=0)
                rvec = rvec - rvec_mean[None, :]
                rvec_stdv = np.sqrt((rvec**2).sum(axis=0))
                rvec = rvec / rvec_stdv[None, :]
                logger.debug("rvec={}".format(rvec[:, 0]))
                dxa = dxa @ rvec
                trace = np.sum(np.diag(dxa @ dxa.T / (nmem-1)))
                logger.info("standard deviation ratio = {}".format(np.sqrt(ptrace / trace)))
                #if np.sqrt(ptrace / trace) > 1.0:
                #    dxa *= np.sqrt(ptrace / trace)
                xa = xa_ + dxa
            xa_ = np.squeeze(xa_)
        elif self.da=="letkf":
            #sigma = 7.5
            sigma = self.lsig
            nx = xf_.size
            xa = np.zeros_like(xf)
            xa_ = np.zeros_like(xf_)
            dxa = np.zeros_like(dxf)
            for i in range(nx):
                A = np.eye(nmem)*(nmem-1)
                if self.linf:   
                    logger.info("==inflation==, alpha={}".format(alpha))
                    A /= alpha
                w = np.zeros(nmem)
                for l in range(len(dy)):
                    far, Rwf_loc = self.r_loc(sigma, nx, yloc[l], float(i))
                    logger.info("number of assimilated obs.={}".format(y[l].size - len(far)))
                    yi = np.delete(y[l],far)
                    di = np.delete(d[l],far)
                    dyi = np.delete(dy[l],far,axis=0)
                    if self.iloc is not None:
                        logger.info("==R-localization==, lsig={}".format(self.lsig))
                        diagR = np.diag(R)
                        Ri = np.diag(diagR/Rwf_loc)
                    else:
                        Ri = R[:,:]
                    Ri = np.delete(Ri,far,axis=0)
                    Ri = np.delete(Ri,far,axis=1)
                    R_inv = la.inv(Ri)
            
                    A = A + dyi.T @ R_inv @ dyi
                    w = w + dyi.T @ R_inv @ di
                lam,v = la.eigh(A)
                D_inv = np.diag(1.0/lam)
                pa_ = v @ D_inv @ v.T
            
                xa_[i] = xf_[i] + dxf[i] @ pa_ @ w
                sqrtPa = v @ np.sqrt(D_inv) @ v.T * np.sqrt(nmem-1)
                dxa[i] = dxf[i] @ sqrtPa
                xa[i] = np.full(nmem,xa_[i]) + dxa[i]
        pa = dxa@dxa.T/(nmem-1)
        spa = dxa / np.sqrt(nmem-1)
        if save_dh:
            ua = np.zeros((xa_.size,nmem+1))
            ua[:,0] = xa_
            ua[:,1:] = xa
            np.save("{}_ua_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), ua)
            np.save("{}_pa_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), pa)

#        if self.ltlm:
#            dh = self.obs.dh_operator(yloc, xa_) @ dxf / np.sqrt(nmem-1)
#        else:
#            x1 = xa_[:, None] + dxf / np.sqrt(nmem-1)
#            dh = self.obs.h_operator(yloc, x1) - np.mean(self.obs.h_operator(yloc, x1), axis=1)[:, None]
#        zmat = rmat @ dh
#        d = y - np.mean(self.obs.h_operator(yloc, xa))
#        innv, chi2 = chi2_test(zmat, d)
        ds = self.dof(dy,nmem)
        logger.info("dof={}".format(ds))
        
        #u = np.zeros_like(xb)
        #u = xa[:,:]
        return xa, pa, ds

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
        #for j in range(nx):
        #    for i in range(ny):
        #        dist[j,i] = min(abs(j-i),nx-abs(j-i))
        for i in range(ny):
            dist[:, i] = self.calc_dist(float(i))
        l_mat = np.exp(-0.5*(dist/loc_scale)**2)
        logger.debug(dist[dist>dist0])
        l_mat[dist>dist0] = 0

        lam, v = la.eigh(l_mat)
        lam = lam[::-1]
        lam[lam < 1.e-10] = 1.e-10
        lamsum = np.sum(lam)
        v = v[:,::-1]
        nmode = 1
        thres = 0.99
        frac = 0.0
        while frac < thres:
            frac = np.sum(lam[:nmode]) / lamsum
            nmode += 1
        nmode = min(nmode, nx, ny)
        logger.info("contribution rate = {}".format(np.sum(lam[:nmode])/np.sum(lam)))
        l_sqrt = v[:,:nmode] @ np.diag(np.sqrt(lam[:nmode]/frac))
        return l_mat, l_sqrt, nmode, np.sqrt(lam[:nmode])

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
        #for j in range(nx):
        #    for i in range(nobs):
        #        dist[j, i] = min(abs(obsloc[i]-j), nx-abs(obsloc[i]-j))
        for i in range(nobs):
            dist[:, i] = self.calc_dist(obsloc[i])
        l_mat = np.exp(-0.5*(dist/loc_scale)**2)
        #logger.debug(dist[dist>dist0])
        l_mat[dist>dist0] = 0
        return l_mat        

    def r_loc(self, sigma, nx, obsloc, xloc):
        if sigma < 0.0:
            loc_scale = 1.0e5
        else:
            loc_scale = sigma
        nobs = obsloc.size
        far = np.arange(nobs)
        Rwf_loc = np.ones(nobs)

        # distance threshold
        dist0 = loc_scale * np.sqrt(10.0/3.0) * 2.0
        #dist0 = 6.5
        logger.debug(dist0)

        dist = np.zeros(nobs)
        #for k in range(nobs):
        #    dist[k] = min(abs(obsloc[k] - xloc), nx-abs(obsloc[k] - xloc))
        for k in range(nobs):
            dist[k] = self.calc_dist1(xloc, obsloc[k])
        far = far[dist>dist0]
        logger.debug(far)
        Rwf_loc = np.exp(-0.5*(dist/loc_scale)**2)
        return far, Rwf_loc

    def pfloc(self, dxf_orig, save_dh, icycle):
        nmem = dxf_orig.shape[1]
        nmode = min(100, self.ndim)
        pf = dxf_orig @ dxf_orig.T / (nmem - 1)
        pf = pf * self.l_mat
        logger.info("lpf max={} min={}".format(np.max(pf),np.min(pf)))
        if save_dh:
            np.save("{}_lpf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), pf)
        lam, v = la.eigh(pf)
        lam = lam[::-1]
        lam[lam < 0.0] = 0.0
        v = v[:,::-1]
        if save_dh:
            np.save("{}_lpfeig_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), lam)
        logger.info("pf eigen value = {}".format(lam))
        pf = v[:,:nmode] @ np.diag(lam[:nmode]) @ v[:,:nmode].T
        dxf = v[:,:nmode] @ np.diag(np.sqrt(lam[:nmode])) * np.sqrt(nmode-1)
        logger.info("pf - spf@spf.T={}".format(np.mean(pf - dxf@dxf.T/(nmode-1))))
        if save_dh:
            np.save("{}_lpfr_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), pf)
            np.save("{}_lspf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxf)
        return dxf

    def pfmod(self, dxf_orig, save_dh, icycle):
        nmem = dxf_orig.shape[1]
        logger.info(f"== modulated ensemble, nmode={self.nmode} ==")
        
        dxf = np.empty((self.ndim, nmem*self.nmode), dxf_orig.dtype)
        for l in range(self.nmode):
            for k in range(nmem):
                m = l*nmem + k
                dxf[:, m] = self.l_sqrt[:, l]*dxf_orig[:, k]
        dxf = dxf * np.sqrt((nmem*self.nmode-1)/(nmem-1))
        if save_dh:
            np.save("{}_lspf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxf)
            fullpf = dxf @ dxf.T / (nmem*self.nmode - 1)
            np.save("{}_lpf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), fullpf)
        return dxf

    def dof(self, dy, nmem):
        zmat = np.sum(np.array(dy), axis=0) / self.sig
        logger.debug(f"zmat max={np.max(zmat)} min={np.min(zmat)}")
        u, s, vt = la.svd(zmat)
        ds = np.sum(s**2/(1.0+s**2))/(nmem-1)
        return ds