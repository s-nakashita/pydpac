import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.lorenz import L96
from analysis.obs import Obs
from l96_func import L96_func

logging.config.fileConfig("logging_config.ini")

global nx, F, dt, dx

model = "l96"
# model parameter
nx = 40     # number of points
F  = 8.0    # forcing
nt =   6    # number of step per forecast (=6 hour)
dt = 0.05 / nt  # time step (=1 hour)

# forecast model forward operator
step = L96(nx, dt, F)

x = np.linspace(-2.0, 2.0, nx)
dx = x[1] - x[0]
np.savetxt("x.txt", x)

nmem =   20 # ensemble size
t0off =   8 # initial offset between adjacent members
t0c =   500 # t0 for control
            # t0 for ensemble members
t0m = [t0c + t0off//2 + t0off * i for i in range(-nmem//2, nmem//2)]
t0f = [t0c] + t0m
na =   100 # number of analysis
namax = 1460 # max number of analysis (1 year)

a_window = 1 # assimilation window length

#sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
#    "quadratic-nodiff": 1.0, "cubic-nodiff": 1.0, "test":1.0}
sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0}
#infl = {"linear": 1.05, "quadratic": 1.3, "cubic": 1.6, \
#    "quadratic-nodiff": 1.3, "cubic-nodiff": 1.6, "test":1.1}
infl_l = {"mlef":1.2,"grad":1.2,"etkf":1.1,"po":1.0,"srf":1.1,"letkf":1.0}
infl_q = {"mlef":1.2,"grad":1.2,"etkf":1.2,"po":1.2,"srf":1.3,"letkf":1.2}
infl_c = {"mlef":1.2,"grad":1.2,"etkf":1.5,"po":1.0,"srf":1.7,"letkf":1.3}
infl_qd = {"mlef":1.2,"grad":1.2,"etkf":1.2,"po":1.2,"srf":1.3,"letkf":1.2}
infl_cd = {"mlef":1.2,"grad":1.2,"etkf":1.5,"po":1.0,"srf":1.7,"letkf":1.3}
infl_t = {"mlef":1.2,"grad":1.2,"etkf":1.1,"po":1.0,"srf":1.1,"letkf":1.0}
dict_infl = {"linear": infl_l, "quadratic": infl_q, "cubic": infl_c, \
    "quadratic-nodiff": infl_qd, "cubic-nodiff": infl_cd, "test": infl_t}
sig_l = {"mlef":8.0,"grad":8.0,"etkf":8.0,"po":14.0,"srf":14.0,"letkf":15.0}
sig_q = {"mlef":6.0,"grad":6.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":4.0}
sig_c = {"mlef":6.0,"grad":6.0,"etkf":6.0,"po":6.0,"srf":6.0,"letkf":10.0}
sig_qd = {"mlef":6.0,"grad":6.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":4.0}
sig_cd = {"mlef":6.0,"grad":6.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":10.0}
sig_t = {"mlef":8.0,"grad":8.0,"etkf":8.0,"po":14.0,"srf":14.0,"letkf":15.0}
dict_sig = {"linear": sig_l, "quadratic": sig_q, "cubic": sig_c, \
    "quadratic-nodiff": sig_qd, "cubic-nodiff": sig_cd, "test":sig_t}
ftype = {"mlef":"ensemble","grad":"ensemble","etkf":"ensemble",\
    "po":"ensemble","srf":"ensemble","letkf":"ensemble",\
        "kf":"deterministic","var":"deterministic","var4d":"deterministic"}

htype = {"operator": "linear", "perturbation": "mlef"}
linf = False
infl_parm = -1.0
lloc = False
lsig = -1.0
ltlm = True

if len(sys.argv) > 1:
    htype["operator"] = sys.argv[1]
if len(sys.argv) > 2:
    htype["perturbation"] = sys.argv[2]

global op, pt, ft
op = htype["operator"]
pt = htype["perturbation"]
ft = ftype[pt]

if len(sys.argv) > 3:
    na = int(sys.argv[3])
if len(sys.argv) > 4:
    infl_parm = float(sys.argv[4])
    if infl_parm > 0.0:
        linf = True
    #if sys.argv[4] == "T":
    #    linf = True
    #    dict_i = dict_infl[op]
    #    infl_parm = dict_i[pt]
if len(sys.argv) > 5:
    #lsig = float(sys.argv[5])
    #if lsig> 0.0:
    #    lloc = True
    if sys.argv[5] == "T":
        lloc = True
        dict_s = dict_sig[op]
        lsig = dict_s[pt]
if len(sys.argv) > 6:
    if sys.argv[6] == "F":
        ltlm = False
if htype["perturbation"] == "var4d":
    if len(sys.argv) > 7:
        a_window = int(sys.argv[7])


# observation operator
obs = Obs(op, sigma[op])

# assimilation method
if pt == "mlef" or pt == "grad":
    from analysis.mlef import Mlef
    analysis = Mlef(pt, obs, infl_parm, lsig, linf, lloc, ltlm, model)
elif pt == "etkf" or pt == "po" or pt == "letkf" or pt == "srf":
    from analysis.enkf import EnKF
    analysis = EnKF(pt, obs, infl_parm, lsig, linf, lloc, ltlm, model)
elif pt == "kf":
    from analysis.kf import Kf
    analysis = Kf(pt, obs, infl_parm, linf, step)
elif pt == "var":
    from analysis.var import Var
    analysis = Var(pt, obs, model)
elif pt == "var4d":
    from analysis.var4d import Var4d
    analysis = Var4d(pt, obs, step, nt, a_window, model)
    
# functions load
params = {"step":step, "obs":obs, "analysis":analysis, \
    "nmem":nmem, "t0c":t0c, "t0f":t0f, "nt":nt, "na":na,\
    "namax":namax, "a_window":a_window, "op":op, "pt":pt, "ft":ft,\
    "linf":linf, "lloc":lloc, "ltlm":ltlm,\
    "infl_parm":infl_parm, "lsig":lsig}
func = L96_func(params)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("==initialize==")
    xt, yobs = func.get_true_and_obs()
    u, xa, xf, pf, sqrtpa = func.initialize(opt=0)
    
    a_time = range(0, na, a_window)
    logger.info("a_time={}".format([time for time in a_time]))
    e = np.zeros(na)
    #if ft == "ensemble":
    innov = np.zeros((na,yobs.shape[1]))
    chi = np.zeros(na)
    dof = np.zeros(na)
    for i in a_time:
        y = yobs[i:i+a_window]
        logger.debug("observation shape {}".format(y.shape))
        if i in [0, 50, 100, 150]:
            logger.info("cycle{} analysis".format(i))
            if a_window > 1:
                u, pa, ds = analysis(u, pf, y, \
                    save_hist=True, save_dh=True, icycle=i)
            #elif ft == "ensemble":
            else:
                u, pa, innv, chi2, ds = analysis(u, pf, y[0], \
                    save_hist=True, save_dh=True, icycle=i)
                chi[i] = chi2
                innov[i] = innv
            #else:
            #    u, pa, ds = analysis(u, pf, y[0], \
            #        save_hist=True, save_dh=True, icycle=i)
        else:
            if a_window > 1:
                u, pa, ds = analysis(u, pf, y, icycle=i)
            #elif ft == "ensemble":
            else:
                u, pa, innv, chi2, ds = analysis(u, pf, y[0], icycle=i)
                chi[i] = chi2
                innov[i] = innv
            #else:
            #    u, pa, ds = analysis(u, pf, y[0], icycle=i)

        if ft=="ensemble":
            xa[i] = u[:, 0]
        else:
            xa[i] = u
        sqrtpa[i] = pa
        dof[i] = ds
        if i < na-1:
            if a_window > 1:
                uf, p = func.forecast(u, pa)
                if (i+1+a_window <= na):
                    xa[i+1:i+1+a_window] = uf
                    xf[i+1:i+1+a_window] = uf
                    sqrtpa[i+1:i+1+a_window, :, :] = p[:, :]
                else:
                    xa[i+1:na] = uf[:na-i-1]
                    xf[i+1:na] = uf[:na-i-1]
                    sqrtpa[i+1:na, :, :] = p[:na-i-1, :, :]
                u = uf[-1]
                pf = p[-1]
            else:
                u, pf = func.forecast(u, pa, tlm=True)
            if ft=="ensemble":
                xf[i+1] = u[:, 0]
            else:
                xf[i+1] = u
        if a_window > 1:
            for k in range(i, min(i+a_window,na)):
                e[k] = np.sqrt(np.mean((xa[k, :] - xt[k, :])**2))
        else:
            e[i] = np.sqrt(np.mean((xa[i, :] - xt[i, :])**2))
            
    np.save("{}_ua_{}_{}.npy".format(model, op, pt), xa)
    np.save("{}_pa_{}_{}.npy".format(model, op, pt), sqrtpa)
    #if len(sys.argv) > 7:
    #    np.savetxt("{}_e_{}_{}_w{}.txt".format(model, op, pt, a_window), e)
    #    np.savetxt("{}_chi_{}_{}_w{}.txt".format(model, op, pt, a_window), chi)
    #else:
    np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)
    #if ft == "ensemble":
    np.save("{}_innv_{}_{}.npy".format(model, op, pt), innov)
    np.savetxt("{}_chi_{}_{}.txt".format(model, op, pt), chi)
    np.savetxt("{}_dof_{}_{}.txt".format(model, op, pt), dof)
