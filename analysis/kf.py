import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
from . import obs

#logging.config.fileConfig("logging_config.ini")
#logger = logging.getLogger(__name__)

def analysis(xf, pf, y, sig, op, infl=False):
    JH = obs.dhdx(xf, op)
    R  = np.eye(y.size)*sig*sig

    if infl:
        pf *= 1.1
    # Kalman gain
    K = pf @ JH.T @ la.inv(JH @ pf @ JH.T + R)

    ob = y - obs.h_operator(xf, op)
    xa = xf + K @ ob

    pa = (np.eye(xf.size) - K @ JH) @ pf

    return xa, pa

def get_linear(xa, h, F, Mb, step):
    eps = 1e-5
    nx = xa.size
    E = np.eye(nx)*eps
    M = np.zeros((nx,nx))

    xf = step(xa, h, F)
    M[:,:] = (step(xa[:,None]+E, h, F) - xf[:,None])/eps
    return M @ Mb