import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
from .obs import Obs

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')
zetak = []

class Var():
    def __init__(self, pt, obs, model):
        self.pt = pt # DA type (MLEF or GRAD)
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig}")

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

    def __call__(self, xf, pf, y, gtol=1e-6,\
        disp=False, save_hist=False, save_dh=False,
        infl=False, loc = False, tlm = False, icycle=0):
        global zetak
        zetak = []
        dum1, dum2, rinv = self.obs.set_r(y.size)
        JH = self.obs.dhdx(xf)
        ob = y - self.obs.h_operator(xf)

        x0 = np.zeros_like(xf)
        binv = la.inv(pf)
        args_j = (binv, JH, rinv, ob)
        logger.info(f"save_hist={save_hist} cycle={icycle}")
        if save_hist:
            res = spo.minimize(self.calc_j, x0, args=args_j, method='BFGS',\
                jac=self.calc_grad_j,options={'gtol':gtol, 'disp':disp}, callback=self.callback)
            jh = np.zeros(len(zetak))
            gh = np.zeros(len(zetak))
            for i in range(len(zetak)):
                jh[i] = self.calc_j(np.array(zetak[i]), *args_j)
                g = self.calc_grad_j(np.array(zetak[i]), *args_j)
                gh[i] = np.sqrt(g.transpose() @ g)
            np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), jh)
            np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), gh)
        else:
            res = spo.minimize(self.calc_j, x0, args=args_j, method='BFGS',\
                jac=self.calc_grad_j,options={'gtol':gtol, 'disp':disp})
        logger.info("success={} message={}".format(res.success, res.message))
        logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
            res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))
    
        xa = xf + res.x

        return xa, pf, 0.0