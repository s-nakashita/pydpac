import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
from .minimize import Minimize

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('anl')
zetak = []

class Var4d():
    def __init__(self, pt, obs, step, nt, window_l, model="model"):
        self.pt = pt # DA type (MLEF or GRAD)
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.step = step
        self.step_t = step.step_t
        self.step_adj = step.step_adj
        self.nt = nt
        self.window_l = window_l
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig}")
        logger.info(f"nt={self.nt} window_l={self.window_l}")

    def callback(self, xk):
        global zetak
        zetak.append(xk)

    def calc_j(self, x, *args):
        binv, JH, rinv, ob, TM, AM = args
        window_l = len(TM)
        jb = 0.5 * x.T @ binv @ x
        jo = 0
        for k in range(window_l):
            Mk = TM[k]
            d = JH @ Mk @ x - ob[k]
            jo = jo + 0.5 * d.T @ rinv @ d
        return jb + jo

    def calc_grad_j(self, x, *args):
        binv, JH, rinv, ob, TM, AM = args
        window_l = len(TM)
        djb = binv @ x
        djo = 0
        for k in range(window_l):
            Mk = TM[k]
            MkT = AM[k]
            d = JH @ Mk @ x - ob[k]
            djo = djo + MkT @ JH.T @ rinv @ d
        return djb + djo

    def __call__(self, xf, pf, y, method="LBFGS", gtol=1e-6, maxiter=None,\
        disp=False, save_hist=False, save_dh=False, icycle=0):
        global zetak
        zetak = []
        JH = self.obs.dhdx(xf)
        dum1, dum2, rinv = self.obs.set_r(np.array(y).shape[1])
        xb = xf
        bg = [xb] # background state
        for k in range(self.window_l-1):
            for l in range(self.nt):
                xb = self.step(xb)
            bg.append(xb)
        TM = [np.eye(xb.size)] # tangent linear model
        AM = [np.eye(xb.size)] # adjoint model
        E = np.eye(xb.size)
        for k in range(self.window_l-1):
            xk = bg[k]
            M = np.eye(xk.size)
            MT = np.eye(xk.size)
            for l in range(self.nt):
                Mk = self.step_t(xk[:, None], E)
                M = Mk @ M
                MkT = self.step_adj(xk[:, None], E)
                MT = MT @ MkT
                xk = self.step(xk)
            TM.append(M@TM[k])
            AM.append(AM[k]@MT)
        logger.debug("Assimilation window size = {}".format(len(TM)))
        ob = [] # innovation
        for k in range(len(bg)):
            ob.append(y[k] - self.obs.h_operator(bg[k]))

        x0 = np.zeros_like(xf)
        binv = la.inv(pf)

        args_j = (binv, JH, rinv, ob, TM, AM)
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

        return xa, pf, 0.0