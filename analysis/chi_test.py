# chi2-test for ensemble DA
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')

class Chi():

    def __init__(self, nobs, nens, rmat):
        self.nobs = nobs # number of observations
        self.nens = nens # ensemble size
        self.rmat = rmat # square root of observation error covariance
    
    def calc_gmat(self, zmat, tol=1e-5):
        Iobs = np.eye(self.nobs)
        cmat = zmat.T @ zmat
        try:
            lam, v = la.eigh(cmat)
        except la.LinAlgError:
            return Iobs, Iobs
        logger.debug("lam shape={}".format(lam.shape))
        logger.debug("v shape={}".format(v.shape))
        phi0 = - 1.0 / (1.0+lam)
        ginv = Iobs + zmat @ v @ np.diag(phi0) @ v.T @ zmat.T

        sig = np.zeros_like(phi0)
        gam = np.zeros_like(phi0)
        for k in range(100):
            tmp = sig[:]
            sig = 0.5 * (sig - gam + phi0 - phi0*lam*gam)
            gam = sig / (1.0 + sig*lam)
            nrm = np.mean((sig-tmp)**2) / np.mean(sig**2)
            if nrm < tol:
                logger.info("converge gmat, iter_num={}, norm={}".format(k, nrm))
                break
        gmat = Iobs + zmat @ v @ np.diag(sig) @ v.T @ zmat.T
        logger.info("err={}".format(np.sum(gmat.T@gmat - ginv)))
        return ginv, gmat
    
    def __call__(self, zmat, d):
        ginv, gmat = self.calc_gmat(zmat)
        dnrm = self.rmat @ d
        innv = gmat @ dnrm / np.sqrt(self.nobs)
        chi2 = innv.T @ innv
        return innv, chi2