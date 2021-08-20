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
from l96_func import L96_func
from hs00_func import HS_func

logging.config.fileConfig("logging_config.ini")

global nx, F, dt, dx

model = "l96"
# model parameter
nx = 40     # number of points
F  = 8.0    # forcing
nt =   6    # number of step per forecast (=6 hour)
dt = 0.05 / 6  # time step (=1 hour)

# forecast model forward operator
step = L96(nx, dt, F)

x = np.linspace(-2.0, 2.0, nx)
dx = x[1] - x[0]
np.savetxt("x.txt", x)

nmem =   20 # ensemble size (not include control run)
t0off =   8 # initial offset between adjacent members
t0c =   500 # t0 for control
# t0 for ensemble members
if nmem%2 == 0: # even
    t0m = [t0c + t0off//2 + t0off * i for i in range(nmem//2)]
    t0f = t0m + [t0c + t0off//2 + t0off * i for i in range(-nmem//2, 0)]
else: # odd
    t0m = [t0c + t0off//2 + t0off * i for i in range(-(nmem-1)//2, (nmem-1)//2)]
    t0f = [t0c] + t0m
na =   100 # number of analysis
namax = 1460 # max number of analysis (1 year)

a_window = 1 # assimilation window length

nobs = 40 # observation number (nobs<=nx)

v_time = 4 * nt # optimization time

lb = 0.5 # correlation length

# observation error standard deviation
sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0, "abs":1.0}
# inflation parameter (dictionary for each observation type)
infl_l = {"mlef":1.2,"etkf":1.2,"po":1.2,"srf":1.2,"letkf":1.2,"kf":1.2,"var":None,
          "4dmlef":1.3,"4detkf":1.3,"4dpo":1.4,"4dsrf":1.2,"4dletkf":1.2,"4dvar":None}
infl_q = {"mlef":1.2,"etkf":1.2,"po":1.2,"srf":1.3,"letkf":1.2,"kf":1.2,"var":None,"4dvar":None}
infl_c = {"mlef":1.2,"etkf":1.5,"po":1.1,"srf":1.8,"letkf":1.3,"kf":1.3,"var":None,"4dvar":None}
infl_qd = {"mlef":1.2,"etkf":1.2,"po":1.2,"srf":1.3,"letkf":1.2,"kf":1.2,"var":None,"4dvar":None}
infl_cd = {"mlef":1.2,"etkf":1.5,"po":1.0,"srf":1.8,"letkf":1.3,"kf":1.3,"var":None,"4dvar":None}
infl_t = {"mlef":1.2,"etkf":1.1,"po":1.0,"srf":1.1,"letkf":1.0,"kf":1.2,"var":None,"4dvar":None}
dict_infl = {"linear": infl_l, "quadratic": infl_q, "cubic": infl_c, \
    "quadratic-nodiff": infl_qd, "cubic-nodiff": infl_cd, "test": infl_t, "abs": infl_l}
# localization parameter (dictionary for each observation type)
sig_l = {"mlef":8.0,"etkf":8.0,"po":2.0,"srf":8.0,"letkf":7.5,"kf":None,"var":None,
        "4dmlef":8.0,"4detkf":8.0,"4dpo":2.0,"4dsrf":8.0,"4dletkf":7.5,"4dvar":None}
sig_q = {"mlef":3.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":4.0,"kf":None,"var":None,"4dvar":None}
sig_c = {"mlef":4.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":6.0,"kf":None,"var":None,"4dvar":None}
sig_qd = {"mlef":6.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":4.0,"kf":None,"var":None,"4dvar":None}
sig_cd = {"mlef":6.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":10.0,"kf":None,"var":None,"4dvar":None}
sig_t = {"mlef":8.0,"etkf":8.0,"po":14.0,"srf":14.0,"letkf":15.0,"kf":None,"var":None,"4dvar":None}
dict_sig = {"linear": sig_l, "quadratic": sig_q, "cubic": sig_c, \
    "quadratic-nodiff": sig_qd, "cubic-nodiff": sig_cd, "test":sig_t, "abs":sig_l}
# forecast type (ensemble or deterministic)
ftype = {"mlef":"ensemble","etkf":"ensemble","po":"ensemble","srf":"ensemble","letkf":"ensemble",\
    "kf":"deterministic","var":"deterministic",\
    "4dmlef":"ensemble","4detkf":"ensemble","4dpo":"ensemble","4dsrf":"ensemble","4dletkf":"ensemble",\
    "4dvar":"deterministic"}

## default parameter
htype = {"operator": "linear", "perturbation": "mlef"}
linf = False
infl_parm = -1.0
lloc = False
lsig = -1.0
ltlm = True
#lplot = False
ntype = "inner"

## read from command options
# observation type
if len(sys.argv) > 1:
    htype["operator"] = sys.argv[1]
# assimilation scheme
if len(sys.argv) > 2:
    htype["perturbation"] = sys.argv[2]
# number of assimilation cycle
if len(sys.argv) > 3:
    na = int(sys.argv[3])

global op, pt, ft
op = htype["operator"]
pt = htype["perturbation"]
ft = ftype[pt]

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
    v_time = int(sys.argv[7])
    v_time = v_time * nt
#if len(sys.argv) > 8:
#    if sys.argv[8] == "T":
#        lplot = True
if len(sys.argv) > 8:
    ntype = sys.argv[8]

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
    analysis = Kf(pt, obs, infl_parm, linf, step, nt, model=model)
elif pt == "var":
    from analysis.var import Var
    analysis = Var(pt, obs, lb, model=model)
elif pt == "4dvar":
    from analysis.var4d import Var4d
    #a_window = 5
    analysis = Var4d(pt, obs, step, nt, a_window, model=model)
elif pt == "4detkf" or pt == "4dpo" or pt == "4dletkf" or pt == "4dsrf":
    from analysis.enks import EnKS
    #a_window = 5
    analysis = EnKS(pt, nmem+1, obs, infl_parm, lsig, linf, lloc, ltlm, step, nt, a_window, model=model)
elif pt == "4dmlef":
    from analysis.mles import Mles
    lloc = False
    analysis = Mles(pt, nmem, obs, infl_parm, lsig, linf, lloc, ltlm, step, nt, a_window, model=model)

# functions load
params = {"step":step, "obs":obs, "analysis":analysis, "nobs":nobs, \
    "t0c":t0c, "t0f":t0f, "nt":nt, "na":na,\
    "namax":namax, "a_window":a_window, "op":op, "pt":pt, "ft":ft,\
    "linf":linf, "lloc":lloc, "ltlm":ltlm,\
    "infl_parm":infl_parm, "lsig":lsig}
func = L96_func(params)
hparams = {"tstep":step, "step":step, "obs":obs, "analysis":analysis, "nobs":nobs, \
    "t0c":t0c, "t0f":t0f, "nt":nt, "na":na,\
    "namax":namax, "a_window":a_window, "op":op, "pt":pt, "ft":ft,\
#    "linf":linf, "lloc":lloc, "ltlm":ltlm,\
#    "infl_parm":infl_parm, "lsig":lsig, \
    "aostype":"NO","vt":v_time}
SV = HS_func(hparams)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("==initialize==")
    xt, yobs = func.get_true_and_obs()
    u, xa, xf, pa, sqrtpa = func.initialize(opt=0)
    pf = analysis.calc_pf(u, pa, 0)
    
    a_time = range(0, na, a_window)
    logger.info("a_time={}".format([time for time in a_time]))
    count = 1
    logger.info("ntype={}".format(ntype))
    e = np.zeros(na)
    for i in a_time:
        yloc = yobs[i:i+a_window,:,0]
        y = yobs[i:i+a_window,:,1]
        logger.debug("observation location {}".format(yloc))
        logger.debug("obs={}".format(y))
        #if i in [1, 50, 100, 150, 200, 250]:
        if pt[:2] == "4d":
            u, pa, ds = analysis(u, pf, y, yloc, icycle=i)
        else:
            u, pa, spa, innv, chi2, ds = analysis(u, pf, y[0], yloc[0], icycle=i)
        
        if ft=="ensemble":
            if pt == "mlef" or pt == "4dmlef":
                u0 = u[:, 0]
            else:
                u0 = np.mean(u, axis=1)
        else:
            u0 = u.copy()
        err = np.sqrt(np.mean((u0 - xt[i, :])**2))
        e[i] = err
        if i >= 100:
            logger.info("RMSE={}".format(err))
            if ntype == "aec":
                gnorm = spa
            elif ntype == "inner":
                gnorm = None
            logger.debug(gnorm)
            if count == 1:
                fig, ax = plt.subplots()
                ax.plot(u0, linestyle="dotted", color="tab:gray", label="reference")
            ua = [u0]
            uf = u.copy()
            for j in range(v_time-1):
                #uf = func.forecast(uf)
                uf = step(uf)
                if ft=="ensemble":
                    if pt == "mlef" or pt == "4dmlef":
                        ua.append(uf[:, 0])
                    else:
                        ua.append(np.mean(uf, axis=1))
                else:
                    ua.append(uf)

            v0 = SV.lanczos(np.array(ua), gnorm=gnorm)
            v = v0.copy()
            for j in range(v_time):
                v = step.step_t(ua[j], v)
            if count == 1:
                ax.plot(v0*10, linestyle="dashed", label="initial")
                ax.plot(v, label="final")
                ax.set_title("{}h".format(v_time))
                ax.legend()
                fig.savefig("initialSVA_{}_{}h.png".format(pt,v_time))
            #np.save("initialSVA_{}_{}h.npy".format(pt,6*v_time),v0)
            np.save("isv_{}_{}h_{}.npy".format(pt,v_time,count),v0)
            #np.save("finalSVA_{}_{}h.npy".format(pt,6*v_time),v)
            np.save("fsv_{}_{}h_{}.npy".format(pt,v_time,count),v)

            vr = np.random.rand(nx)
            scale = np.sqrt(np.mean(vr**2))
            scale0 = np.sqrt(np.mean(v0**2))
            vr = vr / scale * 0.5
            v0 = v0 / scale0 * 0.5
            un = u0.copy()
            ur = un + vr
            us = un + v0
            er = np.zeros(9*nt+1)
            es = np.zeros(9*nt+1)
            ert = np.zeros(9*nt+1)
            est = np.zeros(9*nt+1)
            er[0] = np.sqrt(np.mean((ur - un)**2))
            es[0] = np.sqrt(np.mean((us - un)**2))
            ert[0] = np.sqrt(np.mean(vr**2))
            est[0] = np.sqrt(np.mean(v0**2))
            #params["ft"] = "deterministic"
            #func = L96_func(params)
            j = 1
            for l in range(9):
                for k in range(nt):
                    vr = step.step_t(un, vr)
                    v0 = step.step_t(un, v0)
                    un = step(un)
                    ur = step(ur)
                    us = step(us)
                    er[j] = np.sqrt(np.mean((ur - un)**2))
                    es[j] = np.sqrt(np.mean((us - un)**2))
                    ert[j] = np.sqrt(np.mean(vr**2))
                    est[j] = np.sqrt(np.mean(v0**2))
                    j += 1
            #np.savetxt("erandom.txt",er)
            np.savetxt("er_{}_{}_{}.txt".format(op, pt, count),er)
            np.savetxt("ert_{}_{}_{}.txt".format(op, pt, count),ert)
            #np.savetxt("esv{}h.txt".format(6*v_time),es)
            np.savetxt("es_{}_{}_{}.txt".format(op, pt, count),es)
            np.savetxt("est_{}_{}_{}.txt".format(op, pt, count),est)
            count += 1
        
        if i < na-1:
            if a_window > 1:
                uf = func.forecast(u)
                u = uf[-1]
                pf = analysis.calc_pf(u, pa, i+1)
            else:
                u = func.forecast(u)
                pf = analysis.calc_pf(u, pa, i+1)
    np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)