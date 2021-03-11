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
            if self.model == "l96":
                return np.eye(xf.size)*0.2
            elif self.model == "z08":
                return np.eye(xf.size)*0.02
        else:
            return pa

    def callback(self, xk):
        global zetak
        zetak.append(xk)

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

    def __call__(self, xf, pf, y, yloc, method="BFGS", gtol=1e-6, maxiter=None,\
        disp=False, save_hist=False, save_dh=False, icycle=0):
        global zetak
        zetak = []
        dum1, dum2, rinv = self.obs.set_r(y.size)
        JH = self.obs.dh_operator(yloc, xf)
        ob = y - self.obs.h_operator(yloc,xf)
        nobs = ob.size

        x0 = np.zeros_like(xf)
        binv = la.inv(pf)
        args_j = (binv, JH, rinv, ob)
        iprint = np.zeros(2, dtype=np.int32)
        options = {'gtol':gtol, 'disp':disp, 'maxiter':maxiter}
        minimize = Minimize(x0.size, 7, self.calc_j, self.calc_grad_j, 
                            args_j, iprint, method, options)
        logger.info(f"save_hist={save_hist} cycle={icycle}")
        if save_hist:
            x = minimize(x0, callback=self.callback)
            jh = np.zeros(len(zetak))
            gh = np.zeros(len(zetak))
            for i in range(len(zetak)):
                jh[i] = self.calc_j(np.array(zetak[i]), *args_j)
                g = self.calc_grad_j(np.array(zetak[i]), *args_j)
                gh[i] = np.sqrt(g.transpose() @ g)
            np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), jh)
            np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), gh)
        else:
            x = minimize(x0)
        
        xa = xf + x
        innv = np.zeros_like(ob)
        fun = self.calc_j(x, *args_j)
        chi2 = fun / nobs

        return xa, pf, innv, chi2, 0.0