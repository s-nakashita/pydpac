import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
#from . import obs
from model.lorenz import step, step_t, step_adj

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('anl')
#logger = logging.getLogger().getchild("var4d")
zetak = []

class Var4d():
    def __init__(self, pt, obs):
        self.pt = pt # DA type (MLEF or GRAD)
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig}")

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
        #R_inv = np.eye(np.array(ob).shape[1])/sig/sig
        djo = 0
        for k in range(window_l):
            Mk = TM[k]
            MkT = AM[k]
            d = JH @ Mk @ x - ob[k]
            djo = djo + MkT @ JH.T @ rinv @ d
        return djb + djo

    def analysis(self, xf, pf, y, params, gtol=1e-6,\
        disp=False, save_hist=False, model="z08", icycle=0):
        global zetak
        zetak = []
        h, F, nt, window_l = params
        JH = self.obs.dhdx(xf)
        dum1, dum2, rinv = self.obs.set_r(np.array(y).shape[1])
        xb = xf
        bg = [xb] # background state
        for k in range(window_l-1):
            for l in range(nt):
                xb = step(xb, h, F)
            bg.append(xb)
        TM = [np.eye(xb.size)] # tangent linear model
        AM = [np.eye(xb.size)] # adjoint model
        E = np.eye(xb.size)
        for k in range(window_l-1):
            xk = bg[k]
            M = np.eye(xk.size)
            MT = np.eye(xk.size)
            for l in range(nt):
                Mk = step_t(xk[:, None], E, h, F)
                M = Mk @ M
                MkT = step_adj(xk[:, None], E, h, F)
                MT = MT @ MkT
                xk = step(xk, h, F)
            TM.append(M@TM[k])
            AM.append(AM[k]@MT)
        logger.debug("Assimilation window size = {}".format(len(TM)))
        ob = [] # innovation
        for k in range(len(bg)):
            ob.append(y[k] - self.obs.h_operator(bg[k]))

        x0 = np.zeros_like(xf)
        binv = la.inv(pf)

        args_j = (binv, JH, rinv, ob, TM, AM)
    #print(f"save_hist={save_hist} cycle={icycle}")
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
            np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(model, self.op, self.pt, icycle), jh)
            np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(model, self.op, self.pt, icycle), gh)
        else:
            res = spo.minimize(self.calc_j, x0, args=args_j, method='BFGS',\
                jac=self.calc_grad_j,options={'gtol':gtol, 'disp':disp})
        logger.info("success={} message={}".format(res.success, res.message))
    #    print("success={} message={}".format(res.success, res.message))
        logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
                res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))
#    print("J={:7.3e} dJ={:7.3e} nit={}".format( \
#            res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))
    
        xa = xf + res.x

        return xa