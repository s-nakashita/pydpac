import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
from .chi_test import Chi

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('anl')

class EnKF():

    def __init__(self, da, state_size, nmem, obs, infl, lsig,
                 linf, iloc, ltlm, calc_dist, calc_dist1, model="model"):
        self.da = da # DA type (ETKF, PO, SRF, or LETKF)
        self.ndim = state_size # state size
        self.nmem = nmem # ensemble size
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.infl_parm = infl # inflation parameter
        self.lsig = lsig # localization parameter
        self.linf = linf # True->Apply inflation False->Not apply
        self.iloc = iloc # iloc = None->No localization
                         #      = 0   ->R-localization
                         #      = 1   ->Eigen value decomposition of localized Pf
                         #      = 2   ->Modulated ensemble
        self.ltlm = ltlm # True->Use tangent linear approximation False->Not use
        self.calc_dist = calc_dist # distance calculation routine
        self.calc_dist1 = calc_dist1 # distance calculation routine
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.da} op={self.op} sig={self.sig} infl_parm={self.infl_parm} lsig={self.lsig}")
        logger.info(f"linf={self.linf} iloc={self.iloc} ltlm={self.ltlm}")
        if self.iloc == 1 or self.iloc == 2:
            self.l_mat, self.l_sqrt, self.nmode, self.enswts\
                = self.b_loc(self.lsig, self.ndim, self.ndim)
            np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), self.l_mat)


    def calc_pf(self, xf, pa, cycle):
        dxf = xf - np.mean(xf,axis=1)[:, None]
        pf = dxf @ dxf.transpose() / (self.nmem-1)
        return pf
        
    def __call__(self, xf, pf, y, yloc, save_hist=False, save_dh=False, icycle=0):
        #xf = xb[:]
        xf_ = np.mean(xf, axis=1)
        logger.debug(f"obsloc={yloc}")
        logger.debug(f"obssize={y.size}")
        JH = self.obs.dh_operator(yloc, xf_)
        R, rmat, rinv = self.obs.set_r(y.size)
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
        if (self.iloc == 1 or self.iloc == 2) and self.da != "letkf":
            logger.info("==B-localization==, lsig={}".format(self.lsig))
            dxf_orig = dxf.copy()
            xf_orig = xf.copy()
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
            xf_orig = xf.copy()
        
        if self.ltlm:
            dy = JH @ dxf
        else:
            dy = self.obs.h_operator(yloc, xf) - np.mean(self.obs.h_operator(yloc, xf), axis=1)[:, None]
        d = y - np.mean(self.obs.h_operator(yloc, xf), axis=1)
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
                A = np.eye(nmem2) / alpha
            else:
                A = np.eye(nmem2)
            A = (nmem2-1)*A + dy.T @ rinv @ dy
            lam, v = la.eigh(A)
            #logger.info("eigen values={}".format(lam))
            Dinv = np.diag(1.0/lam)
            TT = v @ Dinv @ v.T
            T = v @ np.sqrt((nmem2-1)*Dinv) @ v.T

            K = dxf @ TT @ dy.T @ rinv
            if save_dh:
                np.save("{}_K_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K)
                np.save("{}_dx_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K@d)
            if self.iloc == 0: # K-localization
                logger.info("==K-localization==, lsig={}".format(self.lsig))
                l_mat = self.k_loc(sigma=self.lsig, obsloc=yloc, xloc=xloc)
                K = K * l_mat
                if save_dh:
                    np.save("{}_Kloc_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K)
                    np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), l_mat)
            #dumu, dums, dumvt = la.svd(K)
            #logger.info("singular values of K={}".format(dums))
            #print(f"rank(kmat)={dums[np.abs(dums)>1.0e-10].shape[0]}")
            xa_ = xf_ + K @ d
            dxa = dxf @ T
            if self.iloc == 1 or self.iloc == 2:
                # random sampling
                ptrace = np.sum(np.diag(dxa @ dxa.T / (nmem2-1)))
                rvec = np.random.randn(nmem2, nmem)
                rvec_mean = np.mean(rvec, axis=0)
                rvec = rvec - rvec_mean[None, :]
                rvec_stdv = np.sqrt((rvec**2).sum(axis=0))
                rvec = rvec / rvec_stdv[None, :]
                logger.debug("rvec={}".format(rvec[:, 0]))
                dxa = dxf @ T @ rvec * np.sqrt((nmem2-1)/(nmem-1))
                trace = np.sum(np.diag(dxa @ dxa.T / (nmem-1)))
                logger.info("standard deviation ratio = {}".format(np.sqrt(ptrace / trace)))
                if np.sqrt(ptrace / trace) > 1.0:
                    dxa *= np.sqrt(ptrace / trace)
            xa = dxa + xa_[:,None]
            
        elif self.da=="po":
            Y = np.zeros((y.size,nmem))
            #np.random.seed(514)
            #err = np.random.normal(0, scale=self.sig, size=Y.size)
            #err = err.reshape(Y.shape)
            mu = np.zeros(y.size)
            sigr = np.eye(y.size)*self.sig
            err = np.random.multivariate_normal(mu,sigr,nmem).T
            err_ = np.mean(err, axis=1)
            #Y = y[:,None] + err.reshape(Y.shape)
            Y = y[:,None] + err
            d_ = d + err_
            K1 = dxf @ dy.T / (nmem2-1)
            K2 = dy @ dy.T / (nmem2-1) + R
            eigK, vK = la.eigh(K2)
            logger.info("eigenvalues of K2")
            logger.info(eigK)
            K2inv = la.inv(K2)
            K = K1 @ K2inv
            #K = dxf @ dy.T @ la.inv(dy @ dy.T + (nmem-1)*R)
            if self.iloc == 0:
                logger.info("==K-localization== lsig={}".format(self.lsig))
                l_mat = self.k_loc(sigma=self.lsig, obsloc=yloc, xloc=xloc)
                K = K * l_mat
                if save_dh:
                    np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), l_mat)
            xa_ = xf_ + K @ d_
            if self.ltlm:
                xa = xf_orig + K @ (Y - JH @ xf_orig)
            else:
                HX = self.obs.h_operator(yloc, xf_orig)
                xa = xf_orig + K @ (Y - HX)
            dxa = xa - xa_[:, None]

        elif self.da=="srf":
            I = np.eye(xf_.size)
            dx0 = np.zeros_like(dxf)
            dx0 = dxf[:,:]
            dy0 = np.zeros_like(dy)
            dy0 = dy[:,:]
            x0_ = np.zeros_like(xf_)
            x0_ = xf_[:]
            d0 = np.zeros_like(d)
            d0 = d[:]
            if self.iloc == 0:
                logger.info("==K-localization== lsig={}".format(self.lsig))
                l_mat = self.k_loc(sigma=self.lsig, obsloc=yloc, xloc=xloc)
                if save_dh:
                    np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), l_mat)
            for i in range(y.size):
                dyi = dy0[i].reshape(1,-1)
                d1 = dyi @ dyi.T + self.sig*self.sig * (nmem2-1)
                k1 = dx0 @ dyi.T /d1
                k1_ = k1 / (1.0 + self.sig/np.sqrt(d1/(nmem2-1)))
                if self.iloc == 0: # K-localization
                    k1_ = k1_ * l_mat[:,i].reshape(k1_.shape)
                xa_ = x0_.reshape(k1.shape) + k1 * d0[i]
                dxa = dx0 - k1_@ dyi
            
                x0_ = xa_[:]
                dx0 = dxa[:,:]
                x0 = x0_ + dx0
                dy0 = self.obs.h_operator(yloc, x0) - np.mean(self.obs.h_operator(yloc, x0), axis=1)[:, None]
                d0 = y - np.mean(self.obs.h_operator(yloc, x0), axis=1)
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
                if np.sqrt(ptrace / trace) > 1.0:
                    dxa *= np.sqrt(ptrace / trace)
            xa = dxa + xa_
            xa_ = np.squeeze(xa_)

        elif self.da=="letkf":
            #sigma = 7.5
            sigma = self.lsig
            nx = xf_.size
            xa = np.zeros_like(xf)
            xa_ = np.zeros_like(xf_)
            dxa = np.zeros_like(dxf)
            E = np.eye(nmem)
            if self.linf:
                logger.info("==inflation==, alpha={}".format(alpha))
                E /= alpha
            for i in range(nx):
                far, Rwf_loc = self.r_loc(sigma, yloc, float(i))
                logger.info("number of assimilated obs.={}".format(y.size - len(far)))
                di = np.delete(d,far)
                dyi = np.delete(dy,far,axis=0)
                if self.iloc==0:
                    logger.info("==R-localization==, lsig={}".format(self.lsig))
                    diagR = np.diag(R)
                    Ri = np.diag(diagR/Rwf_loc)
                else:
                    Ri = R[:,:]
                Ri = np.delete(Ri,far,axis=0)
                Ri = np.delete(Ri,far,axis=1)
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
        spa = dxa / np.sqrt(nmem-1)
        if save_dh:
            ua = np.zeros((xa_.size,nmem+1))
            ua[:,0] = xa_
            ua[:,1:] = xa
            np.save("{}_ua_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), ua)
            np.save("{}_pa_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), pa)

        if self.ltlm:
            dh = self.obs.dh_operator(yloc, xa_) @ dxf / np.sqrt(nmem2-1)
        else:
            x1 = xa_[:, None] + dxf / np.sqrt(nmem2-1)
            dh = self.obs.h_operator(yloc, x1) - np.mean(self.obs.h_operator(yloc, x1), axis=1)[:, None]
        zmat = rmat @ dh
        d = y - np.mean(self.obs.h_operator(yloc, xa))
        innv, chi2 = chi2_test(zmat, d)
        ds = self.dof(dy,nmem)
        logger.info("dof={}".format(ds))
        
        #u = np.zeros_like(xb)
        #u = xa[:,:]
        return xa, pa, spa, innv, chi2, ds

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
        logger.debug(dist[dist>dist0])
        l_mat[dist>dist0] = 0
        return l_mat        

    def r_loc(self, sigma, obsloc, xloc):
        if sigma < 0.0:
            loc_scale = 1.0e5
        else:
            loc_scale = sigma
        nobs = obsloc.size
        far = np.arange(nobs)
        Rwf_loc = np.ones(nobs)

        # distance threshold
        #if self.model == "l96":
        #    dist0 = 6.5
        #else:
        dist0 = loc_scale * np.sqrt(10.0/3.0) * 2.0
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
        if save_dh:
            np.save("{}_lspf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxf)
            fullpf = dxf @ dxf.T / (nmem*self.nmode - 1)
            np.save("{}_lpf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), fullpf)
        return dxf

    def dof(self, dy, nmem):
        zmat = dy / self.sig
        u, s, vt = la.svd(zmat)
        ds = np.sum(s**2/(1.0+s**2))/(nmem-1)
        return ds