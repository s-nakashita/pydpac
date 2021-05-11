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
    def __init__(self, pt, obs, model="model"):
        self.pt = pt # DA type 
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig}")

    def calc_pf(self, xf, pa, cycle):
        if cycle == 0:
            if self.model == "l96" or self.model == "hs00":
                return np.eye(xf.size)*0.2*self.window_l
            elif self.model == "z08":
                return np.eye(xf.size)*0.1
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
        dum1, dum2, rinv = self.obs.set_r(y.size)
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

        #spf = la.cholesky(pf)

        return xa, pf, None, innv, chi2, 0.0