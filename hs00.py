import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from model.lorenz import L96
from analysis.obs import Obs
from hs00_func import HS_func

logging.config.fileConfig("logging_config.ini")

global nx, F, dt, dx

model = "hs00"
# model parameter
nx = 40     # number of points
F  = 8.0    # forcing
nt =   1    # number of step per forecast (=6 hour)
dt = 0.05 / nt  # time step (=1 hour)

# true model forward operator
tstep = L96(nx, dt, F)
# forecast model forward operator
step = L96(nx, dt, F*0.95)

x = np.linspace(-2.0, 2.0, nx)
dx = x[1] - x[0]
np.savetxt("x.txt", x)

nmem =  40 # ensemble size (not include control run)
t0off =   8 # initial offset between adjacent members
t0c =   360 # t0 for control
# t0 for ensemble members
if nmem%2 == 0: # even
    t0m = [t0c + t0off//2 + t0off * i for i in range(nmem//2)]
    t0f = t0m + [t0c + t0off//2 + t0off * i for i in range(-nmem//2, 0)]
else: # odd
    t0m = [t0c + t0off//2 + t0off * i for i in range(-(nmem-1)//2, (nmem-1)//2)]
    t0f = [t0c] + t0m
na =   100 + 360 # number of analysis

a_window = 1  # assimilation window length 
f_window = 40 # forecast length (10 days)

nobs = 20 # regular observation number (on land)

# observation error standard deviation
sigma = {"linear": 0.2, "test":1.0, "abs":1.0}
# inflation parameter (dictionary for each observation type)
infl_l = {"mlef":1.2,"etkf":1.2,"po":1.2,"srf":1.2,"letkf":1.2,"kf":1.2,"var":None,"var4d":None}
infl_q = {"mlef":1.2,"etkf":1.2,"po":1.2,"srf":1.3,"letkf":1.2,"kf":1.2,"var":None,"var4d":None}
infl_c = {"mlef":1.2,"etkf":1.5,"po":1.1,"srf":1.8,"letkf":1.3,"kf":1.3,"var":None,"var4d":None}
infl_qd = {"mlef":1.2,"etkf":1.2,"po":1.2,"srf":1.3,"letkf":1.2,"kf":1.2,"var":None,"var4d":None}
infl_cd = {"mlef":1.2,"etkf":1.5,"po":1.0,"srf":1.8,"letkf":1.3,"kf":1.3,"var":None,"var4d":None}
infl_t = {"mlef":1.2,"etkf":1.1,"po":1.0,"srf":1.1,"letkf":1.0,"kf":1.2,"var":None,"var4d":None}
dict_infl = {"linear": infl_l, "quadratic": infl_q, "cubic": infl_c, \
    "quadratic-nodiff": infl_qd, "cubic-nodiff": infl_cd, "test": infl_t, "abs": infl_l}
# localization parameter (dictionary for each observation type)
sig_l = {"mlef":8.0,"etkf":8.0,"po":2.0,"srf":8.0,"letkf":7.5,"kf":None,"var":None,"var4d":None}
sig_q = {"mlef":3.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":4.0,"kf":None,"var":None,"var4d":None}
sig_c = {"mlef":4.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":6.0,"kf":None,"var":None,"var4d":None}
sig_qd = {"mlef":6.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":4.0,"kf":None,"var":None,"var4d":None}
sig_cd = {"mlef":6.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":10.0,"kf":None,"var":None,"var4d":None}
sig_t = {"mlef":8.0,"etkf":8.0,"po":14.0,"srf":14.0,"letkf":15.0,"kf":None,"var":None,"var4d":None}
dict_sig = {"linear": sig_l, "quadratic": sig_q, "cubic": sig_c, \
    "quadratic-nodiff": sig_qd, "cubic-nodiff": sig_cd, "test":sig_t, "abs":sig_l}
# forecast type (ensemble or deterministic)
ftype = {"mlef":"ensemble","etkf":"ensemble",\
    "po":"ensemble","srf":"ensemble","letkf":"ensemble",\
    "kf":"deterministic","var":"deterministic","var4d":"deterministic",\
    "rep":"deterministic","rep-mb":"ensemble"}

## default parameter
htype = {"operator": "linear", "perturbation": "mlef"}
aostype = "NO"
vt = 1
linf = True
infl_parm = 1.1
lloc = True
lsig = 2.0
ltlm = True

## read from command options
# observation type
if len(sys.argv) > 1:
    htype["operator"] = sys.argv[1]
# assimilation scheme
if len(sys.argv) > 2:
    htype["perturbation"] = sys.argv[2]
# number of assimilation cycle
if len(sys.argv) > 3:
    na = int(sys.argv[3]) + 360
# Adaptive Observation Strategy (AOS) type
if len(sys.argv) > 4:
    aostype = sys.argv[4]
# verification time for Singular Vector
if len(sys.argv) > 5:
    vt = int(sys.argv[5])

namax = na + f_window # true state length (5 years + 90 days(spinup) + forecast length)

global op, pt, ft
op = htype["operator"]
pt = htype["perturbation"]
ft = ftype[pt]
    
"""
# switch of with/without inflation
if len(sys.argv) > 4:
    #infl_parm = float(sys.argv[4])
    #if infl_parm > 0.0:
    #    linf = True
    if sys.argv[4] == "T":
        linf = True
        dict_i = dict_infl[op]
        infl_parm = dict_i[pt]
# switch of with/without localization
if len(sys.argv) > 5:
    #lsig = float(sys.argv[5])
    #if lsig> 0.0:
    #    lloc = True
    if sys.argv[5] == "T":
        lloc = True
        dict_s = dict_sig[op]
        lsig = dict_s[pt]
# switch of using/not using tangent linear operator
if len(sys.argv) > 6:
    if sys.argv[6] == "F":
        ltlm = False
# number of ensemble member (or observation size)
if len(sys.argv) > 7:
    #nobs = int(sys.argv[7])
    nmem = int(sys.argv[7])
"""
# observation operator
obs = Obs(op, sigma[op])

# assimilation method
if pt == "mlef":
    if not lloc:
        from analysis.mlef import Mlef
    #    lloc = False
        analysis = Mlef(pt, nmem, obs, infl_parm, lsig, linf, lloc, ltlm, step.calc_dist, step.calc_dist1, model=model)
    else:
        from analysis.mlef_rloc import Mlef_rloc
        analysis = Mlef_rloc(pt, nmem, obs, infl_parm, lsig, linf, ltlm, step.calc_dist, step.calc_dist1, model=model)
elif pt == "etkf" or pt == "po" or pt == "letkf" or pt == "srf":
    from analysis.enkf import EnKF
    analysis = EnKF(pt, nmem, obs, infl_parm, lsig, linf, lloc, ltlm, step.calc_dist, step.calc_dist1, model=model)
elif pt == "kf":
    from analysis.kf import Kf
    analysis = Kf(pt, obs, infl_parm, linf, step, nt, model)
elif pt == "var":
    from analysis.var import Var
    sigb = np.sqrt(0.2)
    lb = -1.0
    analysis = Var(obs, 
    sigb=sigb, lb=lb, model=model)
elif pt == "4dvar":
    from analysis.var4d import Var4d
    #a_window = 5
    sigb = np.sqrt(0.2)
    lb = -1.0
    analysis = Var4d(obs, step, nt, a_window,
    sigb=sigb, lb=lb, model=model)
elif pt == "rep" or pt == "rep-mb":
    from analysis.rep import Rep
    analysis = Rep(pt, obs, model)

# functions load
params = {"tstep":tstep, "step":step, "obs":obs, "analysis":analysis, "nobs":nobs, \
    "t0c":t0c, "t0f":t0f, "nt":nt, "na":na,\
    "namax":namax, "a_window":a_window, "op":op, "pt":pt, "ft":ft,\
#    "linf":linf, "lloc":lloc, "ltlm":ltlm,\
#    "infl_parm":infl_parm, "lsig":lsig, \
    "aostype":aostype, "vt":vt}
func = HS_func(params)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("==initialize==")
    xt, yobs, u, xa, xf, pa, sqrtpa = func.initialize(opt=0)
    #logger.debug("x0 = {}".format(u))
    pf = analysis.calc_pf(u, pa, 0)
    #logger.debug("pf = {}".format(pf))
    pa = pf
    a_time = range(0, na, a_window)
    #logger.info("a_time={}".format([time for time in a_time[:100]]))
    if aostype == "MB":
        du = u[:, 1:] - u[:, 0].reshape(-1,1)
        scale0 = np.sqrt(np.sum(du**2, axis=0))
        logger.info("initial norm={}".format(scale0))
    emean = np.zeros((f_window+1, nx))
    nsample = 0
    for i in a_time:
        if i == 0:
            ua = np.zeros((1, u.shape[0]))
            if ft=="ensemble":
                if pt == "mlef" or pt == "rep-mb":
                    ua[0] = u[:, 0]
                    spa = u[:, 1:] - ua[0].reshape(-1,1)
                else:
                    ua[0] = np.mean(u, axis=1)
                    spa = (u - ua[0].reshape(-1,1))/np.sqrt(u.shape[1]-1)
            else:
                ua[0] = u
                spa = np.eye(u.size)
        else:
            ua = xa[max(0, i-vt):i]
        #yloc = yobs[i:i+a_window,:,0]
        #y = yobs[i:i+a_window,:,1]
        if aostype == "NO" or aostype == "RO":
            aos = None
        elif aostype == "AE":
            aos = np.argmax(np.sqrt((xf[i] - xt[i])**2))
        elif aostype == "AU":
            aos = np.argmax(np.diag(pa))
        elif aostype == "MB":
            du = u[:, 1:] - u[:, 0].reshape(-1,1)
            aos = np.argmax(np.sum(du**2, axis=1))
            scale = np.sqrt(np.sum(du**2, axis=0))
            logger.info("cycle{} norm={}".format(i, scale))
            scale = scale / scale0
            du = du / scale.reshape(1,-1)
            u[:, 1:] = u[:, 0].reshape(-1,1) + du
        elif aostype == "SVI" or aostype == "SVA":
            if aostype == "SVA":
                gnorm = spa
            else:
                gnorm = None
            v = func.lanczos(ua, gnorm=gnorm)
            aos = np.argmax(v)
        yloc, y = func.get_aos(yobs, i, aos)
        logger.debug("observation location {}".format(yloc))
        logger.debug("obs={}".format(y))
        ##if i in [1, 50, 100, 150, 200, 250]:
        #if i in range(1):
        #    logger.info("cycle{} analysis".format(i))
        #    if a_window > 1:
        #        u, pa, ds = analysis(u, pf, y, yloc, \
        #            save_hist=True, save_dh=True, icycle=i)
        #    else:
        #        u, pa, innv, chi2, ds = analysis(u, pf, y[0], yloc[0], \
        #            save_hist=True, save_dh=True, icycle=i)
        #else:
        logger.info("cycle{} analysis".format(i))
        if a_window > 1:
            u, pa, ds = analysis(u, pf, y, yloc, icycle=i)
        else:
            u, pa, spa, innv, chi2, ds = analysis(u, pf, y[0], yloc[0], icycle=i)
            
        if ft=="ensemble":
            if pt == "mlef" or pt == "rep-mb":
                xa[i] = u[:, 0]
            else:
                xa[i] = np.mean(u, axis=1)
        else:
            xa[i] = u
        sqrtpa[i] = pa

        if i >= 360: # after spinup
            nsample += 1
            uf = np.zeros_like(u)
            uf = u[:]
            for j in range(f_window+1):
                if ft == "deterministic":
                    emean[j,:] = emean[j,:] + np.sqrt((uf - xt[i+j])**2)
                else:
                    if pt == "mlef" or pt == "rep-mb":
                        emean[j,:] = emean[j,:] + np.sqrt((uf[:, 0] - xt[i+j])**2)
                    else:
                        emean[j,:] = emean[j,:] + np.sqrt((np.mean(uf, axis=1) - xt[i+j])**2)
                uf = func.forecast(uf)
        if i < na-1:
            if a_window > 1:
                uf = func.forecast(u)
                if (i+1+a_window <= na):
                    xa[i+1:i+1+a_window] = uf
                    xf[i+1:i+1+a_window] = uf
                    ii = 0
                    for k in range(i+1,i+1+a_window):
                        sqrtpa[k, :, :] = analysis.calc_pf(uf[ii], pa, k)
                        ii += 1
                else:
                    xa[i+1:na] = uf[:na-i-1]
                    xf[i+1:na] = uf[:na-i-1]
                    ii = 0
                    for k in range(i+1,na):
                        sqrtpa[k, :, :] = analysis.calc_pf(uf[ii], pa, k)
                        ii += 1
                u = uf[-1]
                pf = analysis.calc_pf(u, pa, i+1)
            else:
                u = func.forecast(u)
                pf = analysis.calc_pf(u, pa, i+1)

            if ft=="ensemble":
                if pt == "mlef" or pt == "rep-mb":
                    xf[i+1] = u[:, 0]
                else:
                    xf[i+1] = np.mean(u, axis=1)
            else:
                xf[i+1] = u
        #if a_window > 1:
        #    for k in range(i, min(i+a_window,na)):
        #        e[k] = np.sqrt(np.mean((xa[k, :] - xt[k, :])**2))
        #else:
        #    e[i] = np.sqrt(np.mean((xa[i, :] - xt[i, :])**2))
            
    np.save("{}_ut.npy".format(model), xt)
    np.save("{}_uf_{}_{}.npy".format(model, op, pt), xf)
    np.save("{}_ua_{}_{}.npy".format(model, op, pt), xa)
    np.save("{}_pa_{}_{}.npy".format(model, op, pt), sqrtpa)
    
    logger.info(f"{nsample} samples")
    emean = emean / nsample
    np.save("{}_e_{}_{}.npy".format(model, op, pt), emean)
    