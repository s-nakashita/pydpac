import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
#from .obs import Obs

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('anl')

class Kf():

    def __init__(self, pt, obs, infl):
        self.pt = pt # DA type (MLEF or GRAD)
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.infl_parm = infl # inflation parameter
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig} infl_parm={self.infl_parm}")

    def analysis(self, xf, pf, y, infl=False):
        JH = self.obs.dhdx(xf)
        #R  = np.eye(y.size)*sig*sig
        R, dum1, dum2 = self.obs.set_r(y.size)

        if infl:
            logger.info("==inflation==")
            pf *= self.infl_parm
        # Kalman gain
        K = pf @ JH.T @ la.inv(JH @ pf @ JH.T + R)

        ob = y - self.obs.h_operator(xf)
        xa = xf + K @ ob

        pa = (np.eye(xf.size) - K @ JH) @ pf

        return xa, pa

    def get_linear(self, xa, h, F, Mb, step):
        eps = 1e-5
        nx = xa.size
        E = np.eye(nx)*eps
        M = np.zeros((nx,nx))

        xf = step(xa, h, F)
        M[:,:] = (step(xa[:,None]+E, h, F) - xf[:,None])/eps
        return M @ Mb