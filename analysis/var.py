import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
from .obs import Obs
from .minimize import Minimize

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')
zetak = []
alphak = []

class Var():
    def __init__(self, obs, nx, sigb=1.0, lb=-1.0, bmat=None, calc_dist=None, cyclic=True, model="model"):
        self.pt = "var" # DA type 
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.nx = nx # state size
        # climatological background error covariance
        self.sigb = sigb # error variance
        self.lb = lb # error correlation length (< 0.0 : diagonal)
        self.bmat = bmat # prescribed background error covariance
        self.cyclic = cyclic # boundary treatment
        if calc_dist is None:
            self.calc_dist = self._calc_dist
        else:
            self.calc_dist = calc_dist # distance function
        self.model = model
        self.verbose = True
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig}")
        logger.info(f"sigb={self.sigb} lb={self.lb}")
        logger.info(f"bmat in={self.bmat is not None}")

    def _calc_dist(self, ix):
        dist = np.zeros(self.nx)
        for j in range(self.nx):
            if self.cyclic:
                dist[j] = np.abs(self.nx/np.pi*np.sin(np.pi*(i-j)/self.nx))
            else:
                dist[j] = np.abs(self.nx/np.pi/2*np.sin(np.pi*(i-j)/self.nx/2))
        return dist

    def calc_pf(self, xf, pa, cycle):
        if cycle == 0:
            if self.bmat is None:
                if self.lb < 0:
                    self.bmat = self.sigb**2*np.eye(self.nx)
                else:
                    dist = np.eye(self.nx)
                    for i in range(self.nx):
                        dist[i,:] = self.calc_dist(i)
                    #for i in range(nx):
                    #    for j in range(nx):
                    #        if self.cyclic:
                    #            dist[i,j] = np.abs(nx/np.pi*np.sin(np.pi*(i-j)/nx))
                    #        else:
                    #            dist[i,j] = np.abs(nx/np.pi/2*np.sin(np.pi*(i-j)/nx/2))
                    self.bmat = self.sigb**2 * np.exp(-0.5*(dist/self.lb)**2)
            else:
                # use only the correlation structure
                diag = np.diag(self.bmat)
                dsqrtinv = np.diag(1.0/np.sqrt(diag))
                cmat = dsqrtinv @ self.bmat @ dsqrtinv
                self.bmat = np.diag(np.full(cmat.shape[0],self.sigb)) @ cmat @ np.diag(np.full(cmat.shape[0],self.sigb))
            if self.verbose:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(ncols=2)
                xaxis = np.arange(self.nx+1)
                mappable = ax[0].pcolor(xaxis, xaxis, self.bmat, cmap='Blues')
                fig.colorbar(mappable, ax=ax[0],shrink=0.6,pad=0.01)
                ax[0].set_title(r"$\mathrm{cond}(\mathbf{B})=$"+f"{la.cond(self.bmat):.3e}")
                ax[0].invert_yaxis()
                ax[0].set_aspect("equal")
                binv = la.inv(self.bmat)
                mappable = ax[1].pcolor(xaxis, xaxis, binv, cmap='Blues')
                fig.colorbar(mappable, ax=ax[1],shrink=0.6,pad=0.01)
                ax[1].set_title(r"$\mathbf{B}^{-1}$")
                ax[1].invert_yaxis()
                ax[1].set_aspect("equal")
                fig.tight_layout()
                fig.savefig("Bv{:.1f}l{:d}_{}.png".format(self.sigb,int(self.lb),self.model))
                plt.close()
        return self.bmat

    def callback(self, xk, alpha=None):
        global zetak, alphak
        logger.debug("xk={}".format(xk))
        zetak.append(xk)
        if alpha is not None:
            alphak.append(alpha)

    def prec(self,w,first=False):
        global bsqrt
        if first:
            eival, eivec = la.eigh(self.bmat)
            eival[eival<1.0e-16] = 0.0
            npos = np.sum(eival>=1.0e-16)
            logger.info(f"#positive eigenvalues in bmat={npos}")
            bsqrt = np.dot(eivec,np.diag(np.sqrt(eival)))
        return np.dot(bsqrt,w), bsqrt

    def calc_j(self, w, *args):
        JH, rinv, ob = args
        jb = 0.5 * np.dot(w,w)
        x, _ = self.prec(w)
        d = JH @ x - ob
        jo = 0.5 * d.T @ rinv @ d
        return jb + jo

    def calc_grad_j(self, w, *args):
        JH, rinv, ob = args
        x, bsqrt = self.prec(w)
        d = JH @ x - ob
        return w + bsqrt.T @ JH.T @ rinv @ d

    def calc_hess(self, w, *args):
        JH, rinv, ob = args
        _, bsqrt = self.prec(w)
        return np.eye(w.size) + bsqrt.T @ JH.T @ rinv @ JH @ bsqrt

    def __call__(self, xf, pf, y, yloc, method="CG", cgtype=1,
        gtol=1e-6, maxiter=None,\
        disp=False, save_hist=False, save_dh=False, icycle=0,
        evalout=False):
        global zetak, alphak, bsqrt
        zetak = []
        alphak = []
        _, rsqrtinv, rinv = self.obs.set_r(yloc)
        JH = self.obs.dh_operator(yloc, xf)
        ob = y - self.obs.h_operator(yloc,xf)
        nobs = ob.size

        w0 = np.zeros_like(xf)
        x0, bsqrt = self.prec(w0,first=True)
        args_j = (JH, rinv, ob)
        iprint = np.zeros(2, dtype=np.int32)
        options = {'gtol':gtol, 'disp':disp, 'maxiter':maxiter}
        minimize = Minimize(w0.size, self.calc_j, jac=self.calc_grad_j, hess=self.calc_hess,
                            args=args_j, iprint=iprint, method=method, cgtype=cgtype,
                            maxiter=maxiter)
        logger.info(f"save_hist={save_hist} cycle={icycle}")
        if save_hist:
            w, flg = minimize(w0, callback=self.callback)
            jh = np.zeros(len(zetak))
            gh = np.zeros(len(zetak))
            for i in range(len(zetak)):
                jh[i] = self.calc_j(np.array(zetak[i]), *args_j)
                g = self.calc_grad_j(np.array(zetak[i]), *args_j)
                gh[i] = np.sqrt(g.transpose() @ g)
            np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), jh)
            np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), gh)
            np.savetxt("{}_alpha_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), alphak)
        else:
            w, flg = minimize(w0)
        
        x, _ = self.prec(w)
        xa = xf + x
        innv = np.zeros_like(ob)
        fun = self.calc_j(w, *args_j)
        chi2 = fun / nobs

        pai = self.calc_hess(w, *args_j)
        lam, v = la.eigh(pai)
        dfs = xf.size - np.sum(1.0/lam)
        spa = bsqrt @ v @ np.diag(1.0/np.sqrt(lam)) @ v.transpose()
        pa = np.dot(spa,spa.T)
        #spf = la.cholesky(pf)

        if evalout:
            tmp = np.dot(np.dot(rsqrtinv,JH),spa)
            infl_mat = np.dot(tmp,tmp.T)
            eval, _ = la.eigh(infl_mat)
            logger.debug("eval={}".format(eval))
            return xa, pa, spa, innv, chi2, dfs, eval[::-1]
        else:
            return xa, pa, spa, innv, chi2, dfs