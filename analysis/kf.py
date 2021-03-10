import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('anl')

class Kf():

    def __init__(self, pt, obs, infl, linf, step):
        self.pt = pt # DA type (MLEF or GRAD)
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.infl_parm = infl # inflation parameter
        self.linf = linf
        self.step = step
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig} infl_parm={self.infl_parm} linf={self.linf}")

    def __call__(self, xf, pf, y, yloc, save_hist=False, save_dh=False, icycle=0):
        JH = self.obs.dh_operator(yloc,xf)
        R, dum1, dum2 = self.obs.set_r(y.size)

        if self.linf:
            logger.info("==inflation==")
            pf *= self.infl_parm
        # Kalman gain
        K = pf @ JH.T @ la.inv(JH @ pf @ JH.T + R)

        ob = y - self.obs.h_operator(yloc,xf)
        xa = xf + K @ ob

        pa = (np.eye(xf.size) - K @ JH) @ pf

        innv, chi2 = self.chi2(pf, JH, R, ob)
        ds = self.dof(K, JH)
        logger.info("dof={}".format(ds))

        return xa, pa, innv, chi2, ds

    def get_linear(self, xa, Mb):
        eps = 1e-5
        nx = xa.size
        E = np.eye(nx)*eps
        M = np.zeros((nx,nx))

        xf = self.step(xa)
        M[:,:] = (self.step(xa[:,None]+E) - xf[:,None])/eps
        return M @ Mb

    def dof(self, K, JH):
        A = K @ JH
        ds = np.sum(np.diag(A))
        logger.info(ds)
        return ds

    def chi2(self, pf, JH, R, d):
        nobs = d.size
        s = JH @ pf @ JH.T + R
        sinv = la.inv(s)
        l, v = la.eigh(s)
        smat = v @ np.diag(np.sqrt(l)) @ v.T
        chi2 = d.T @ sinv @ d / nobs
        innv = smat @ d / np.sqrt(nobs)
        logger.debug(innv.T @ innv - chi2)
        return innv, chi2
