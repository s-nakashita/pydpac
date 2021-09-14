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
    def __init__(self, pt, obs, lb, model="model"):
        self.pt = pt # DA type 
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.lb = lb # correlation length (< 0.0 : diagonal)
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig} lb={self.lb}")

    def calc_pf(self, xf, pa, cycle):
        import matplotlib.pyplot as plt
        if cycle == 0:
            if self.model == "l96" or self.model == "hs00":
                sigb = np.sqrt(0.2)
                nx = xf.size
                if self.lb < 0:
                    bmat = sigb**2*np.eye(nx)
                else:
                    dist = np.eye(nx)
                    for i in range(nx):
                        for j in range(nx):
                            dist[i,j] = np.abs(nx/np.pi*np.sin(np.pi*(i-j)/nx))
                    bmat = sigb**2 * np.exp(-0.5*(dist/self.lb)**2)
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
                fig.savefig("B{}.png".format(self.lb))
                return bmat

            elif self.model == "z08":
                return np.eye(xf.size)*0.1
            elif self.model == "tc87":
                return np.eye(xf.size)*1.0
        else:
            return pa

    def callback(self, xk, alpha=None):
        global zetak, alphak
        logger.debug("xk={}".format(xk))
        zetak.append(xk)
        if alpha is not None:
            alphak.append(alpha)

    def calc_j(self, x, *args):
        binv, JH, rinv, ob = args
        jb = 0.5 * x.T @ binv @ x
        d = JH @ x - ob
        jo = 0.5 * d.T @ rinv @ d
        return jb + jo

    def calc_grad_j(self, x, *args):
        binv, JH, rinv, ob = args
        d = JH @ x - ob
        return binv @ x + JH.T @ rinv @ d

    def calc_hess(self, x, *args):
        binv, JH, rinv, ob = args
        return binv + JH.T @ rinv @ JH

    def __call__(self, xf, pf, y, yloc, method="CGF", cgtype=1,
        gtol=1e-6, maxiter=None,\
        disp=False, save_hist=False, save_dh=False, icycle=0):
        global zetak, alphak
        zetak = []
        alphak = []
        _, _, rinv = self.obs.set_r(yloc)
        JH = self.obs.dh_operator(yloc, xf)
        ob = y - self.obs.h_operator(yloc,xf)
        nobs = ob.size

        x0 = np.zeros_like(xf)
        binv = la.inv(pf)
        args_j = (binv, JH, rinv, ob)
        iprint = np.zeros(2, dtype=np.int32)
        options = {'gtol':gtol, 'disp':disp, 'maxiter':maxiter}
        minimize = Minimize(x0.size, self.calc_j, jac=self.calc_grad_j, hess=self.calc_hess,
                            args=args_j, iprint=iprint, method=method, cgtype=cgtype,
                            maxiter=maxiter)
        logger.info(f"save_hist={save_hist} cycle={icycle}")
        if save_hist:
            x, flg = minimize(x0, callback=self.callback)
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
            x, flg = minimize(x0)
        
        xa = xf + x
        innv = np.zeros_like(ob)
        fun = self.calc_j(x, *args_j)
        chi2 = fun / nobs

        pai = self.calc_hess(x, *args_j)
        lam, v = la.eigh(pai)
        spa = v @ np.diag(1.0/np.sqrt(lam)) @ v.transpose()
        #spf = la.cholesky(pf)

        return xa, pf, spa, innv, chi2, 0.0