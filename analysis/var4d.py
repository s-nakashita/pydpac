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
    def __init__(self, obs, step, nt, window_l, sigb=1.0, lb=-1.0, model="model"):
        self.pt = "4dvar" # DA type
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.step = step
        self.step_t = step.step_t
        self.step_adj = step.step_adj
        self.nt = nt
        self.window_l = window_l
        # climatological background error
        self.sigb = sigb # error variance
        self.lb = lb # error correlation length (< 0.0 : diagonal)
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig}")
        logger.info(f"nt={self.nt} window_l={self.window_l}")
        from .var import Var
        self.var = Var(obs,sigb=self.sigb,lb=self.lb,model=self.model)

    def calc_pf(self, xf, **kwargs):
        return self.var.calc_pf(xf,**kwargs)
    
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
        bmat, JH, rinv, ob, TM, AM = args
        window_l = len(TM)
        jb = 0.5 * np.dot(w,w)
        x, _ = self.prec(w,bmat)
        jo = 0
        for k in range(window_l):
            Mk = TM[k]
            d = JH[k] @ Mk @ x - ob[k]
            jo = jo + 0.5 * d.T @ rinv @ d
        return jb + jo

    def calc_grad_j(self, w, *args):
        bmat, JH, rinv, ob, TM, AM = args
        window_l = len(TM)
        x, bsqrt = self.prec(w,bmat)
        djb = w
        djo = 0
        for k in range(window_l):
            Mk = TM[k]
            MkT = AM[k]
            d = JH[k] @ Mk @ x - ob[k]
            djo = djo + bsqrt.T @ MkT @ JH[k].T @ rinv @ d
        return djb + djo

    def calc_hess(self, w, *args):
        bmat, JH, rinv, ob, TM, AM = args
        window_l = len(TM)
        _, bsqrt = self.prec(w,bmat)
        djb = np.eye(w.size)
        djo = np.zeros((w.size,w.size))
        for k in range(window_l):
            Mk = TM[k]
            MkT = AM[k]
            d = JH[k] @ Mk @ bsqrt
            djo = djo + bsqrt.T @ MkT @ JH[k].T @ rinv @ d
        return djb + djo

    def __call__(self, xf, pf, y, yloc, method="CG", cgtype=1,
        gtol=1e-6, maxiter=None,\
        disp=False, save_hist=False, save_dh=False, icycle=0):
        global zetak, alphak
        zetak = []
        alphak = []
        #JH = self.obs.dh_operator(yloc, xf)
        _, _, rinv = self.obs.set_r(yloc[0])
        a_window = min(len(y),self.window_l)
        xb = xf
        bg = [xb] # background state
        for k in range(a_window-1):
            for l in range(self.nt):
                xb = self.step(xb)
            bg.append(xb)
        TM = [np.eye(xb.size)] # tangent linear model
        AM = [np.eye(xb.size)] # adjoint model
        JH = [self.obs.dh_operator(yloc[0], xf)] # tangent linear observation operator
        E = np.eye(xb.size)
        for k in range(a_window-1):
            xk = bg[k]
            M = np.eye(xk.size)
            MT = np.eye(xk.size)
            Hk = self.obs.dh_operator(yloc[k+1], bg[k+1])
            for l in range(self.nt):
                Mk = self.step_t(xk[:, None], E)
                M = Mk @ M
                MkT = self.step_adj(xk[:, None], E)
                MT = MT @ MkT
            TM.append(M@TM[k])
            AM.append(AM[k]@MT)
            JH.append(Hk)
        logger.debug("Assimilation window size = {}".format(len(TM)))
        ob = [] # innovation
        for k in range(len(bg)):
            ob.append(y[k] - self.obs.h_operator(yloc[k],bg[k]))

        w0 = np.zeros_like(xf)
        x0, bsqrt = self.prec(w0,pf,first=True)

        args_j = (pf, JH, rinv, ob, TM, AM)
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
        fun = self.calc_j(w,*args_j)
        nobs = np.array(ob).size
        self.innv = np.zeros(nobs)
        self.chi2 = fun / nobs

        pai = self.calc_hess(w,*args_j)
        eval, evec = la.eigh(pai)
        self.ds = xf.size - np.sum(1.0/eval)
        spa = bsqrt @ evec @ np.diag(1.0/np.sqrt(eval)) @ evec.T
        pa = np.dot(spa,spa.T)

        return xa, pa #, spa, innv, chi2, dfs
