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
    def __init__(self, obs, sigb=1.0, lb=-1.0, model="model"):
        self.pt = "var" # DA type 
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        # climatological background error
        self.sigb = sigb # error variance
        self.lb = lb # error correlation length (< 0.0 : diagonal)
        self.model = model
        self.verbose = False
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig} lb={self.lb}")

    def calc_pf(self, xf, pa, cycle):
        global bmat
        if cycle == 0:
            nx = xf.size
            if self.lb < 0:
                bmat = self.sigb**2*np.eye(nx)
            else:
                dist = np.eye(nx)
                for i in range(nx):
                    for j in range(nx):
                        dist[i,j] = np.abs(nx/np.pi*np.sin(np.pi*(i-j)/nx))
                bmat = self.sigb**2 * np.exp(-0.5*(dist/self.lb)**2)
            if self.verbose:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(ncols=2)
                xaxis = np.arange(nx+1)
                mappable = ax[0].pcolor(xaxis, xaxis, bmat, cmap='Blues')
                fig.colorbar(mappable, ax=ax[0])
                ax[0].set_title(r"$\mathbf{B}$")
                ax[0].invert_yaxis()
                ax[0].set_aspect("equal")
                binv = la.inv(bmat)
                mappable = ax[1].pcolor(xaxis, xaxis, binv, cmap='Blues')
                fig.colorbar(mappable, ax=ax[1])
                ax[1].set_title(r"$\mathbf{B}^{-1}$")
                ax[1].invert_yaxis()
                ax[1].set_aspect("equal")
                fig.tight_layout()
                fig.savefig("Bv{:.1f}l{:d}.png".format(self.sigb,int(self.lb)))
        return bmat

    def callback(self, xk, alpha=None):
        global zetak, alphak
        logger.debug("xk={}".format(xk))
        zetak.append(xk)
        if alpha is not None:
            alphak.append(alpha)

    def prec(self,w,bmat,first=False):
        global bsqrt
        if first:
            eval, evec = la.eigh(bmat)
            eval[eval<1.0e-16] = 0.0
            bsqrt = np.dot(evec,np.diag(np.sqrt(eval)))
        return np.dot(bsqrt,w), bsqrt

    def calc_j(self, w, *args):
        bmat, JH, rinv, ob = args
        jb = 0.5 * np.dot(w,w)
        x, _ = self.prec(w,bmat)
        d = JH @ x - ob
        jo = 0.5 * d.T @ rinv @ d
        return jb + jo

    def calc_grad_j(self, w, *args):
        bmat, JH, rinv, ob = args
        x, bsqrt = self.prec(w,bmat)
        d = JH @ x - ob
        return w + bsqrt.T @ JH.T @ rinv @ d

    def calc_hess(self, w, *args):
        bmat, JH, rinv, ob = args
        _, bsqrt = self.prec(w,bmat)
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
        x0, bsqrt = self.prec(w0,pf,first=True)
        args_j = (pf, JH, rinv, ob)
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
        
        x, _ = self.prec(w,pf)
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