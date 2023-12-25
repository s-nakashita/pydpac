#
# Nesting variational assimilation 
# Reference : Guidard and Fischer (2008, QJRMS), Dahlgren and Gustafsson (2012, Tellus A)
#
import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
from scipy.interpolate import interp1d
from .obs import Obs
from .minimize import Minimize

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')
zetak = []
alphak = []

class Var_nest():
    def __init__(self, obs, ix_gm, \
        sigb=1.0, lb=-1.0, bmat=None, \
        sigv=1.0, lv=-1.0, vmat=None, \
        crosscov=False, ekbmat=None, ebkmat=None, \
        model="model"):
        self.pt = "var_nest" # DA type 
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.ix_gm = ix_gm # GM grid
        # LAM background error covariance
        self.sigb = sigb # error variance
        self.lb = lb # error correlation length (< 0.0 : diagonal)
        self.bmat = bmat # prescribed background error covariance
        # GM background error covariance in LAM space
        self.sigv = sigv # error variance
        self.lv = lv # error correlation length (< 0.0 : diagonal)
        self.vmat = vmat # prescribed background error covariance
        # correlation between GM and LAM
        self.crosscov = crosscov # whether correlation is considered or not
        self.ekbmat = ekbmat
        self.ebkmat = ebkmat
        #
        self.model = model
        self.verbose = True
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig} lb={self.lb}")
        logger.info(f"bmat in={self.bmat is not None}")
        logger.info(f"sigv={self.sigv} lv={self.lv}")
        logger.info(f"vmat in={self.vmat is not None}")
        logger.info(f"crosscov={self.crosscov}")

    def calc_pf(self, xf, pa, cycle):
        if cycle == 0:
            nx = xf.size
            if self.bmat is None:
                if self.lb < 0:
                    self.bmat = self.sigb**2*np.eye(nx)
                else:
                    dist = np.eye(nx)
                    for i in range(nx):
                        for j in range(nx):
                            dist[i,j] = np.abs(nx/np.pi*np.sin(np.pi*(i-j)/nx))
                    self.bmat = self.sigb**2 * np.exp(-0.5*(dist/self.lb)**2)
            if self.vmat is None:
                if self.lv < 0:
                    self.vmat = self.sigv**2*np.eye(nx)
                else:
                    dist = np.eye(nx)
                    for i in range(nx):
                        for j in range(nx):
                            dist[i,j] = np.abs(nx/np.pi*np.sin(np.pi*(i-j)/nx))
                    self.vmat = self.sigv**2 * np.exp(-0.5*(dist/self.lv)**2)
            if self.verbose:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(nrows=2,ncols=2)
                xaxis = np.arange(nx+1)
                mappable = ax[0,0].pcolor(xaxis, xaxis, self.bmat, cmap='Blues')
                fig.colorbar(mappable, ax=ax[0,0],shrink=0.6,pad=0.01)
                ax[0,0].set_title(r"$\mathbf{B}$")
                ax[0,0].invert_yaxis()
                ax[0,0].set_aspect("equal")
                binv = la.inv(self.bmat)
                mappable = ax[0,1].pcolor(xaxis, xaxis, binv, cmap='Blues')
                fig.colorbar(mappable, ax=ax[0,1],shrink=0.6,pad=0.01)
                ax[0,1].set_title(r"$\mathbf{B}^{-1}$")
                ax[0,1].invert_yaxis()
                ax[0,1].set_aspect("equal")
                mappable = ax[1,0].pcolor(xaxis, xaxis, self.vmat, cmap='Blues')
                fig.colorbar(mappable, ax=ax[1,0],shrink=0.6,pad=0.01)
                ax[1,0].set_title(r"$\mathbf{V}$")
                ax[1,0].invert_yaxis()
                ax[1,0].set_aspect("equal")
                vinv = la.inv(self.vmat)
                mappable = ax[1,1].pcolor(xaxis, xaxis, vinv, cmap='Blues')
                fig.colorbar(mappable, ax=ax[1,1],shrink=0.6,pad=0.01)
                ax[1,1].set_title(r"$\mathbf{V}^{-1}$")
                ax[1,1].invert_yaxis()
                ax[1,1].set_aspect("equal")
                fig.tight_layout()
                fig.savefig("Bv{:.1f}l{:d}+Vv{:.1f}l{:d}_{}.png".format(self.sigb,int(self.lb),self.sigv,int(self.lv),self.model))
                plt.close()
        return self.bmat

    def callback(self, xk, alpha=None):
        global zetak, alphak
        logger.debug("xk={}".format(xk))
        zetak.append(xk)
        if alpha is not None:
            alphak.append(alpha)

    def prec(self,w,first=False):
        global bsqrt, vsqrtinv
        if first:
            eival, eivec = la.eigh(self.bmat)
            eival = eival[::-1]
            eivec = eivec[:,::-1]
            #eival[eival<1.0e-16] = 0.0
            npos = np.sum(eival>=1.0e-16)
            logger.info(f"#positive eigenvalues in bmat={npos}")
            bsqrt = np.dot(eivec[:,:npos],np.diag(np.sqrt(eival[:npos])))
            eival, eivec = la.eigh(self.vmat)
            eival = eival[::-1]
            eivec = eivec[:,::-1]
            #eival[eival<1.0e-16] = 1.0e-16
            npos = np.sum(eival>=1.0e-16)
            logger.info(f"#positive eigenvalues in vmat={npos}")
            vsqrtinv = np.dot(np.diag(1.0/np.sqrt(eival[:npos])),eivec[:,:npos].T)
        return np.dot(bsqrt,w), bsqrt, vsqrtinv

    def calc_j(self, w, *args):
        JH, rinv, ob, dk, JH2 = args
        jb = 0.5 * np.dot(w,w)
        x, _, vsqrtinv = self.prec(w)
        d = JH @ x - ob
        jo = 0.5 * d.T @ rinv @ d
        dktmp = vsqrtinv @ (JH2@x-dk)
        jk = 0.5 * np.dot(dktmp,dktmp)
        return jb + jo + jk

    def calc_grad_j(self, w, *args):
        JH, rinv, ob, dk, JH2 = args
        x, bsqrt, vsqrtinv = self.prec(w)
        d = JH @ x - ob
        dktmp = vsqrtinv @ (JH2@x-dk)
        return w + bsqrt.T @ JH.T @ rinv @ d + bsqrt.T @ vsqrtinv.T @ dktmp

    def calc_hess(self, w, *args):
        JH, rinv, ob, dk, JH2 = args
        _, bsqrt, vsqrtinv = self.prec(w)
        return np.eye(w.size) + bsqrt.T @ JH.T @ rinv @ JH @ bsqrt + bsqrt.T @ JH2.T @ vsqrtinv.T @ vsqrtinv @ JH2 @ bsqrt

    def __call__(self, xf, pf, y, yloc, xg, ix_lam, method="CG", cgtype=1,
        gtol=1e-6, maxiter=30,\
        disp=False, save_hist=False, save_dh=False, icycle=0,
        evalout=False):
        global zetak, alphak, bsqrt
        zetak = []
        alphak = []
        _, rsqrtinv, rinv = self.obs.set_r(yloc)
        JH = self.obs.dh_operator(yloc, xf)
        ob = y - self.obs.h_operator(yloc,xf)
        nobs = ob.size
        i0 = np.argmin(np.abs(self.ix_gm-ix_lam[0]))
        if self.ix_gm[i0]<ix_lam[0]: i0+=1
        i1 = np.argmin(np.abs(self.ix_gm-ix_lam[-1]))
        if self.ix_gm[i1]>ix_lam[-1]: i1-=1
        x_lam2gm = interp1d(ix_lam,xf)
        dk = xg[i0:i1+1] - x_lam2gm(self.ix_gm[i0:i1+1])
        tmp_lam2gm = interp1d(ix_lam,np.eye(ix_lam.size),axis=0)
        JH2 = tmp_lam2gm(self.ix_gm[i0:i1+1])

        w0 = np.zeros_like(xf)
        x0, bsqrt, vsqrtinv = self.prec(w0,first=True)
        args_j = (JH, rinv, ob, dk, JH2)
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
        
        x, _, _ = self.prec(w)
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