import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
from . import obs
from . import costJ


#logging.config.fileConfig("./logging_config.ini")
#logger = logging.getLogger(__name__)


def precondition(zmat):
    u, s, vt = la.svd(zmat)
    v = vt.transpose()
    is2r = 1 / (1 + s**2)
    tmat = v @ np.diag(np.sqrt(is2r)) @ vt
    heinv = v @ np.diag(is2r) @ vt
#    logger.debug("tmat={}".format(tmat))
#    logger.debug("heinv={}".format(heinv))
#    logger.debug("s={}".format(s))
    print("s={}".format(s))
    return tmat, heinv

zetak = []
def callback(xk):
    global zetak
#    logger.debug("xk={}".format(xk))
    zetak.append(xk)


def calc_j(zeta, *args):
    xc, pf, y, tmat, gmat, heinv, rinv, htype = args
    nmem = zeta.size
    x = xc + gmat @ zeta
    ob = y - obs.h_operator(x, htype["operator"])
    j = 0.5 * (zeta.transpose() @ heinv @ zeta + ob.transpose() @ rinv @ ob)
    #j = 0.5 * ((nmem-1)*zeta.transpose() @ heinv @ zeta + nob.transpose() @ rinv @ ob)
#    logger.debug("zeta.shape={}".format(zeta.shape))
#    logger.debug("j={} zeta={}".format(j, zeta))
    return j
    

def calc_grad_j(zeta, *args):
    xc, pf, y, tmat, gmat, heinv, rinv, htype = args
    nmem = zeta.size
    x = xc + gmat @ zeta
    hx = obs.h_operator(x, htype["operator"])
    ob = y - hx
    if htype["perturbation"] == "grad":
        dh = obs.dhdx(x, htype["operator"]) @ pf
    else:
        dh = obs.h_operator(x[:, None] + pf, htype["operator"]) - hx[:, None]
    return tmat @ zeta - dh.transpose() @ rinv @ ob
    #return (nmem-1)*tmat @ zeta - dh.transpose() @ rinv @ ob


def analysis(xf, xc, y, rmat, rinv, htype, gtol=1e-6, 
        disp=False, save_hist=False, save_dh=False,
        infl=False, loc = False, model="z08", icycle=0):
    global zetak
    zetak = []
    op = htype["operator"]
    pt = htype["perturbation"]
    nmem = xf.shape[1]
    pf = xf - xc[:, None]
#    logger.debug("norm(pf)={}".format(la.norm(pf)))
    if pt == "grad":
#        logger.debug("dhdx.shape={}".format(obs.dhdx(xc, op).shape))
        dh = obs.dhdx(xc, op) @ pf
    else:
        dh = obs.h_operator(xf, op) - obs.h_operator(xc, op)[:, None]
    if save_dh:
        np.save("{}_dh_{}_{}_cycle{}.npy".format(model, op, pt, icycle), dh)
        ob = y - obs.h_operator(xc, op)
        np.save("{}_d_{}_{}_cycle{}.npy".format(model, op, pt, icycle), ob)
#    logger.info("save_dh={}".format(save_dh))
#    print("save_dh={}".format(save_dh))
    zmat = rmat @ dh
#    logger.debug("cond(zmat)={}".format(la.cond(zmat)))
    tmat, heinv = precondition(zmat)
#    logger.debug("pf.shape={}".format(pf.shape))
#    logger.debug("tmat.shape={}".format(tmat.shape))
#    logger.debug("heinv.shape={}".format(heinv.shape))
    gmat = pf @ tmat
    #gmat = np.sqrt(nmem-1) * pf @ tmat
#    logger.debug("gmat.shape={}".format(gmat.shape))
    x0 = np.zeros(xf.shape[1])
    args_j = (xc, pf, y, tmat, gmat, heinv, rinv, htype)
#    logger.info("save_hist={}".format(save_hist))
#    print("save_hist={} cycle{}".format(save_hist, icycle))
    if save_hist:
        g = calc_grad_j(x0, *args_j)
        #print("g={}".format(g))
        res = spo.minimize(calc_j, x0, args=args_j, method='BFGS', \
                jac=calc_grad_j, options={'gtol':gtol, 'disp':disp}, callback=callback)
        #res = spo.minimize(calc_j, x0, args=args_j, method='BFGS', \
        #        jac=None, options={'gtol':gtol, 'disp':disp}, callback=callback)
        jh = np.zeros(len(zetak))
        gh = np.zeros(len(zetak))
        for i in range(len(zetak)):
            jh[i] = calc_j(np.array(zetak[i]), *args_j)
            g = calc_grad_j(np.array(zetak[i]), *args_j)
            gh[i] = np.sqrt(g.transpose() @ g)
        np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), jh)
        np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle), gh)
        if model=="z08":
            xmax = max(np.abs(np.min(res.x)),np.max(res.x))
            print("resx max={}".format(xmax))
            if xmax < 1000:
                cost_j(1000, xf.shape[1], model, res.x, icycle, *args_j)
            else:
                xmax = int(xmax*0.01+1)*100
                print("resx max={}".format(xmax))
                cost_j(xmax, xf.shape[1], model, res.x, icycle, *args_j)
        elif model=="l96":
            cost_j(200, xf.shape[1], model, res.x, icycle, *args_j)
    else:
        res = spo.minimize(calc_j, x0, args=args_j, method='BFGS', \
                jac=calc_grad_j, options={'gtol':gtol, 'disp':disp})
        #res = spo.minimize(calc_j, x0, args=args_j, method='BFGS', \
        #        jac=None, options={'gtol':gtol, 'disp':disp})
#    logger.info("success={} message={}".format(res.success, res.message))
    print("success={} message={}".format(res.success, res.message))
#    logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
#            res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))
    print("J={:7.3e} dJ={:7.3e} nit={}".format( \
            res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))
    xa = xc + gmat @ res.x
    if save_dh:
        np.save("{}_dx_{}_{}_cycle{}.npy".format(model, op, pt, icycle), gmat@res.x)
    if pt == "grad":
        dh = obs.dhdx(xa, op) @ pf
    else:
        dh = obs.h_operator(xa[:, None] + pf, op) - obs.h_operator(xa, op)[:, None]
    zmat = rmat @ dh
#    logger.debug("cond(zmat)={}".format(la.cond(zmat)))
    tmat, heinv = precondition(zmat)
    d = y - obs.h_operator(xa, op)
    chi2 = chi2_test(zmat, heinv, rmat, d)
    pa = pf @ tmat 
    if save_dh:
        np.save("{}_pa_{}_{}_cycle{}.npy".format(model, op, pt, icycle), pa)
        ua = np.zeros((xa.size, nmem+1))
        ua[:, 0] = xa
        ua[:, 1:] = xa[:, None] + pa
        np.save("{}_ua_{}_{}_cycle{}.npy".format(model, op, pt, icycle), ua)
        #print("xa={}".format(xa))
    if infl:
        print("==inflation==")
        if pt == "mlef":
            pa *= 1.1
        else:
            pa *= 1.1
    return xa, pa, chi2

def cost_j(nx, nmem, model, xopt, icycle, *args):
    xc, pf, y, tmat, gmat, heinv, rinv, htype = args
    op = htype["operator"]
    pt = htype["perturbation"]
    delta = np.linspace(-nx,nx,4*nx)
    jvalb = np.zeros((len(delta)+1,nmem))
    jvalb[0,:] = xopt
    #jvalo = np.zeros_like(jvalb)
    #jvalo[0,:] = xopt
    #jvalb[1:,:], jvalo[1:,:] = costJ.cost_j(nx, nmem, *args)
    #np.save("{}_cJb_{}_{}_cycle{}.npy".format(model, op, pt, icycle), jvalb)
    #np.save("{}_cJo_{}_{}_cycle{}.npy".format(model, op, pt, icycle), jvalo)
    for k in range(nmem):
        x0 = np.zeros(nmem)
        for i in range(len(delta)):
            x0[k] = delta[i]
            j = calc_j(x0, *args)
            jvalb[i+1,k] = j
    np.save("{}_cJ_{}_{}_cycle{}.npy".format(model, op, pt, icycle), jvalb)

def chi2_test(zmat, heinv, rmat, d):
    p = d.size
    G_inv = np.eye(p) - zmat @ heinv @ zmat.T
    innv = rmat @ d[:,None]
    return innv.T @ G_inv @ innv / p