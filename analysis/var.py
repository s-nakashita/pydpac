import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
from . import obs

logging.config.fileConfig("./logging_config.ini")

zetak = []
def callback(xk):
    global zetak
    zetak.append(xk)

def calc_j(x, *args):
    binv, JH, sig, ob = args
    jb = 0.5 * x.T @ binv @ x
    d = JH @ x - ob
    R_inv = np.eye(ob.size)/sig/sig
    jo = 0.5 * d.T @ R_inv @ d
    return jb + jo

def calc_grad_j(x, *args):
    binv, JH, sig, ob = args
    d = JH @ x - ob
    R_inv = np.eye(ob.size)/sig/sig
    return binv @ x + JH.T @ R_inv @ d

def analysis(xf, pf, y, sig, htype, gtol=1e-6,\
        disp=False, save_hist=False, model="z08", icycle=0):
    global zetak
    logger = logging.getLogger('anl')
    zetak = []
    op = htype["operator"]
    pt = htype["perturbation"]
    JH = obs.dhdx(xf, op)
    ob = y - obs.h_operator(xf, op)

    x0 = np.zeros_like(xf)
    binv = la.inv(pf)
    args_j = (binv, JH, sig, ob)
    logger.info(f"save_hist={save_hist} cycle={icycle}")
    if save_hist:
        res = spo.minimize(calc_j, x0, args=args_j, method='BFGS',\
            jac=calc_grad_j,options={'gtol':gtol, 'disp':disp}, callback=callback)
        jh = np.zeros(len(zetak))
        gh = np.zeros(len(zetak))
        for i in range(len(zetak)):
            jh[i] = calc_j(np.array(zetak[i]), *args_j)
            g = calc_grad_j(np.array(zetak[i]), *args_j)
            gh[i] = np.sqrt(g.transpose() @ g)
        np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), jh)
        np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), gh)
    else:
        res = spo.minimize(calc_j, x0, args=args_j, method='BFGS',\
            jac=calc_grad_j,options={'gtol':gtol, 'disp':disp})
    logger.info("success={} message={}".format(res.success, res.message))
#    print("success={} message={}".format(res.success, res.message))
    logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
            res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))
#    print("J={:7.3e} dJ={:7.3e} nit={}".format( \
#            res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))
    
    xa = xf + res.x

    return xa