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
alphak = []

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

    def calc_pf(self, xf, pa, cycle):
        if cycle == 0:
            if self.model == "l96" or self.model == "hs00":
                return np.eye(xf.size)*0.2
            elif self.model == "z08":
                return np.eye(xf.size)*0.02
        else:
            return pa
    
    def callback(self, xk, alpha=None):
        global zetak, alphak
        logger.debug("xk={}".format(xk))
        zetak.append(xk)
        if alpha is not None:
            alphak.append(alpha)

    def calc_j(self, x, *args):
        binv, JH, rinv, ob, TM, AM = args
        window_l = len(TM)
        jb = 0.5 * x.T @ binv @ x
        jo = 0
        for k in range(window_l):
            Mk = TM[k]
            d = JH[k] @ Mk @ x - ob[k]
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
            d = JH[k] @ Mk @ x - ob[k]
            djo = djo + MkT @ JH[k].T @ rinv @ d
        return djb + djo

    def calc_hess(self, x, *args):
        binv, JH, rinv, ob, TM, AM = args
        window_l = len(TM)
        djb = binv
        djo = np.zeros(x.size)
        for k in range(window_l):
            Mk = TM[k]
            MkT = AM[k]
            d = JH[k] @ Mk 
            djo = djo + MkT @ JH[k].T @ rinv @ d
        return djb + djo

    def __call__(self, xf, pf, y, yloc, method="CGF", cgtype=1,
        gtol=1e-6, maxiter=None,\
        disp=False, save_hist=False, save_dh=False, icycle=0):
        global zetak, alphak
        zetak = []
        alphak = []
        #JH = self.obs.dh_operator(yloc, xf)
        dum1, dum2, rinv = self.obs.set_r(np.array(y).shape[1])
        xb = xf
        bg = [xb] # background state
        for k in range(self.window_l-1):
            for l in range(self.nt):
                xb = self.step(xb)
            bg.append(xb)
        TM = [np.eye(xb.size)] # tangent linear model
        AM = [np.eye(xb.size)] # adjoint model
        JH = [self.obs.dh_operator(yloc[0], xf)] # tangent linear observation operator
        E = np.eye(xb.size)
        for k in range(self.window_l-1):
            xk = bg[k]
            M = np.eye(xk.size)
            MT = np.eye(xk.size)
            Hk = self.obs.dh_operator(yloc[k+1], bg[k+1])
            for l in range(self.nt):
                Mk = self.step_t(xk[:, None], E)
                M = Mk @ M
                MkT = self.step_adj(xk[:, None], E)
                MT = MT @ MkT
                xk = self.step(xk)
            TM.append(M@TM[k])
            AM.append(AM[k]@MT)
            JH.append(Hk)
        logger.debug("Assimilation window size = {}".format(len(TM)))
        ob = [] # innovation
        for k in range(len(bg)):
            ob.append(y[k] - self.obs.h_operator(yloc[k],bg[k]))

        x0 = np.zeros_like(xf)
        binv = la.inv(pf)

        args_j = (binv, JH, rinv, ob, TM, AM)
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

        return xa, pf, 0.0