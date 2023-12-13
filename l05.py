import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.lorenz import L96
from analysis.obs import Obs
from l05_func import L05_func

logging.config.fileConfig("logging_config.ini")

model = "l05II"
global nx, nk, F, dt
if len(sys.argv) > 1:
    model = sys.argv[1]
dt6h=0.05 # model time step equivalent to 6 hours
if model == "l05II":
    from model.lorenz2 import L05II as L05
    # model parameter
    nx = 240       # number of points
    nk = 8         # advection length scale
    F  = 10.0      # forcing
    dt = dt6h / 6  # time step (=1 hour)
    args = nx, nk, dt, F
elif model == "l05III":
    from model.lorenz3 import L05III as L05
    global ni, b, c
    # model parameter
    nx = 960       # number of points
    nk = 32        # advection length scale
    ni = 12        # spatial filter width
    F  = 15.0      # forcing
    b  = 10.0      # frequency of small-scale perturbation
    c  = 0.6       # coupling factor
    dt = dt6h / 6 / 6 # time step (=1/6 hour)
    args = nx, nk, ni, b, c, dt, F
# forecast model forward operator
step = L05(*args)

np.savetxt("ix.txt",np.arange(step.nx))

# observation error standard deviation
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
# inflation parameter (dictionary for each observation type)
infl_l = {"mlef":1.05,"mlefw":1.2,"etkf":1.2,"po":1.2,"srf":1.2,"letkf":1.05,"kf":1.2,"var":None,
          "4dmlef":1.4,"4detkf":1.3,"4dpo":1.2,"4dsrf":1.2,"4dletkf":1.2,"4dvar":None}
infl_q = {"mlef":1.2,"etkf":1.2,"po":1.2,"srf":1.3,"letkf":1.2,"kf":1.2,"var":None,
          "4dmlef":1.4,"4detkf":1.3,"4dpo":1.2,"4dsrf":1.2,"4dletkf":1.2,"4dvar":None}
infl_c = {"mlef":1.2,"etkf":1.5,"po":1.1,"srf":1.8,"letkf":1.3,"kf":1.3,"var":None,"4dvar":None}
infl_qd = {"mlef":1.2,"etkf":1.2,"po":1.2,"srf":1.3,"letkf":1.2,"kf":1.2,"var":None,"4dvar":None}
infl_cd = {"mlef":1.2,"etkf":1.5,"po":1.0,"srf":1.8,"letkf":1.3,"kf":1.3,"var":None,"4dvar":None}
infl_t = {"mlef":1.2,"etkf":1.1,"po":1.0,"srf":1.1,"letkf":1.0,"kf":1.2,"var":None,"4dvar":None}
infl_h = {"mlef":1.3,"etkf":1.1,"po":1.0,"srf":1.1,"letkf":1.0,"kf":1.2,"var":None,"4dvar":None}
dict_infl = {"linear": infl_l, "quadratic": infl_q, "cubic": infl_c, \
    "quadratic-nodiff": infl_qd, "cubic-nodiff": infl_cd, \
        "test": infl_t, "abs": infl_l, "hint": infl_h}
# localization parameter (dictionary for each observation type)
sig_l = {"mlef":3.0,"mlefw":2.0,"etkf":2.0,"po":2.0,"srf":2.0,"letkf":3.0,"kf":None,"var":None,
        "4dmlef":2.0,"4detkf":2.0,"4dpo":2.0,"4dsrf":2.0,"4dletkf":2.0,"4dvar":None}
sig_q = {"mlef":2.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":4.0,"kf":None,"var":None,
        "4dmlef":2.0,"4detkf":6.0,"4dpo":6.0,"4dsrf":8.0,"4dletkf":4.0,"4dvar":None}
sig_c = {"mlef":4.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":6.0,"kf":None,"var":None,"4dvar":None}
sig_qd = {"mlef":6.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":4.0,"kf":None,"var":None,"4dvar":None}
sig_cd = {"mlef":6.0,"etkf":6.0,"po":6.0,"srf":8.0,"letkf":10.0,"kf":None,"var":None,"4dvar":None}
sig_t = {"mlef":8.0,"etkf":8.0,"po":14.0,"srf":14.0,"letkf":15.0,"kf":None,"var":None,"4dvar":None}
dict_sig = {"linear": sig_l, "quadratic": sig_q, "cubic": sig_c, \
    "quadratic-nodiff": sig_qd, "cubic-nodiff": sig_cd, \
    "test":sig_t, "abs":sig_l, "hint": sig_l}
# forecast type (ensemble or deterministic)
ftype = {"mlef":"ensemble","mlefw":"ensemble","etkf":"ensemble","po":"ensemble","srf":"ensemble","letkf":"ensemble",\
    "kf":"deterministic","var":"deterministic",\
    "4dmlef":"ensemble","4detkf":"ensemble","4dpo":"ensemble","4dsrf":"ensemble","4dletkf":"ensemble",\
    "4dvar":"deterministic"}

## default parameter
params = dict()
### experiment settings
htype = {"operator": "linear", "perturbation": "mlef"}
params["t0off"]      =  4*int(dt6h/dt)      # initial offset between adjacent members
params["t0c"]        = 100*int(dt6h/dt)     # t0 for control
params["nobs"]       =  40      # observation number (nobs<=nx)
params["obsloctype"] = "random" # observation location type
params["op"]         = "linear" # observation operator type
params["na"]         =  100     # number of analysis cycle
params["nt"]         =  int(dt6h/dt)      # number of step per forecast (=6 hour)
params["namax"]      =  1460    # maximum number of analysis cycle (1 year)
### assimilation method settings
params["pt"]         = "mlef"   # assimilation method
params["nmem"]       =  40      # ensemble size (include control run)
params["a_window"]   =  0       # assimilation window length
params["sigb"]       =  0.6     # (For var & 4dvar) background error standard deviation
params["lb"]         = -1.0     # (For var & 4dvar) correlation length for background error covariance
params["linf"]       =  False   # inflation flag
params["infl_parm"]  = -1.0     # multiplicative inflation coefficient
params["lloc"]       =  False   # localization flag
params["lsig"]       = -1.0     # localization radius
params["iloc"]       =  None    # localization type
params["ss"]         =  False   # (For model space localization) statistical resampling flag
params["getkf"]      =  False   # (For model space localization) gain form resampling flag
params["ltlm"]       =  True    # flag for tangent linear observation operator
params["incremental"] = False   # (For mlef & 4dmlef) flag for incremental form

## update from configure file
sys.path.append('./')
from config import params as params_new
params.update(params_new)
global op, pt, ft
op = params["op"]
pt = params["pt"]
ft = ftype[pt]
global na, a_window
na = params["na"]
a_window = params["a_window"]
params["ft"] = ft
if params["linf"] and params["infl_parm"] == -1.0:
    params["infl_parm"] = dict_infl[params["op"]][params["pt"]]
if params["lloc"] and params["lsig"] == -1.0:
    params["lsig"] = dict_sig[params["op"]][params["pt"]]

# observation operator
obs = Obs(op, sigma[op])

# assimilation class
state_size = nx
if a_window < 1:
    if pt[:2] == "4d":
        a_window = 5
    else:
        a_window = 1
if pt == "mlef":
    from analysis.mlef import Mlef
    analysis = Mlef(state_size, params["nmem"], obs, \
            linf=params["linf"], infl_parm=params["infl_parm"], \
            iloc=params["iloc"], lsig=params["lsig"], ss=params["ss"], getkf=params["getkf"], \
            calc_dist=step.calc_dist, calc_dist1=step.calc_dist1,\
            ltlm=params["ltlm"], incremental=params["incremental"], model=model)
elif pt == "mlefw":
    from analysis.mlefw import Mlefw
    analysis = Mlefw(pt, state_size, nmem, obs, \
        linf=linf, infl_parm=infl_parm, \
        iloc=iloc, lsig=lsig, ss=False, gain=False, \
        calc_dist=step.calc_dist, calc_dist1=step.calc_dist1,\
        ltlm=ltlm, model=model)
elif pt == "etkf" or pt == "po" or pt == "letkf" or pt == "srf":
    from analysis.enkf import EnKF
    analysis = EnKF(pt, state_size, params["nmem"], obs, \
        linf=params["linf"], infl_parm=params["infl_parm"], \
        iloc=params["iloc"], lsig=params["lsig"], ss=params["ss"], getkf=params["getkf"], \
        ltlm=params["ltlm"], \
        calc_dist=step.calc_dist, calc_dist1=step.calc_dist1, model=model)
elif pt == "kf":
    from analysis.kf import Kf
    analysis = Kf(obs, 
    infl=params["infl_parm"], linf=params["linf"], 
    step=step, nt=params["nt"], model=model)
elif pt == "var":
    from analysis.var import Var
    analysis = Var(obs, 
    sigb=params["sigb"], lb=params["lb"], model=model)
elif pt == "4dvar":
    from analysis.var4d import Var4d
    #a_window = 5
    sigb = params["sigb"] * np.sqrt(float(a_window))
    analysis = Var4d(obs, step, params["nt"], a_window,
    sigb=sigb, lb=params["lb"], model=model)
elif pt == "4detkf" or pt == "4dpo" or pt == "4dletkf" or pt == "4dsrf":
    from analysis.enkf4d import EnKF4d
    #a_window = 5
    analysis = EnKF4d(pt, state_size, params["nmem"], obs, step, params["nt"], a_window, \
        linf=params["linf"], infl_parm=params["infl_parm"], 
        iloc=params["iloc"], lsig=params["lsig"], calc_dist=step.calc_dist, calc_dist1=step.calc_dist1, \
        ltlm=params["ltlm"], model=model)
elif pt == "4dmlef":
    #a_window = 5
    from analysis.mlef4d import Mlef4d
    analysis = Mlef4d(state_size, params["nmem"], obs, step, params["nt"], a_window, \
            linf=params["linf"], infl_parm=params["infl_parm"], \
            iloc=params["iloc"], lsig=params["lsig"], calc_dist=step.calc_dist, calc_dist1=step.calc_dist1, \
            ltlm=params["ltlm"], incremental=params["incremental"], model=model)

# functions load
func = L05_func(model,step,obs,params)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("==initialize==")
    xt, yobs = func.get_true_and_obs(obsloctype=params["obsloctype"])
    u, xa, xf, pa = func.initialize(opt=0)
    logger.debug(u.shape)
    func.plot_initial(u[:,0], u[:,1:], xt[0])
    pf = analysis.calc_pf(u, pa, 0)
    
    a_time = range(0, na, a_window)
    logger.info("a_time={}".format([time for time in a_time]))
    e = np.zeros(na)
    stda = np.zeros(na)
    xdmean = np.zeros(nx)
    xsmean = np.zeros(nx)
    innov = np.zeros((na,yobs.shape[1]*a_window))
    chi = np.zeros(na)
    dof = np.zeros(na)
    for i in a_time:
        yloc = yobs[i:min(i+a_window,na),:,0]
        y = yobs[i:min(i+a_window,na),:,1]
        logger.debug("observation location {}".format(yloc))
        logger.debug("obs={}".format(y))
        logger.info("cycle{} analysis : window length {}".format(i,y.shape[0]))
        #if i in [1, 50, 100, 150, 200, 250]:
        if i < 0:
            ##if a_window > 1:
            if pt[:2] == "4d":
                u, pa, spa, innv, chi2, ds = analysis(u, pf, y, yloc, \
                    save_hist=True, save_dh=True, icycle=i)
                for j in range(y.shape[0]):
                    chi[i+j] = chi2
                    innov[i+j,:innv.size] = innv
                    dof[i+j] = ds
            else:
                u, pa, spa, innv, chi2, ds = analysis(u, pf, y[0], yloc[0], \
                    save_hist=True, save_dh=True, icycle=i)
                chi[i] = chi2
                innov[i] = innv
                dof[i] = ds
        else:
            ##if a_window > 1:
            if pt[:2] == "4d":
                u, pa, spa, innv, chi2, ds = analysis(u, pf, y, yloc, icycle=i)
                for j in range(y.shape[0]):
                    chi[i+j] = chi2
                    innov[i+j,:innv.size] = innv
                    dof[i+j] = ds
            else:
                u, pa, spa, innv, chi2, ds = analysis(u, pf, y[0], yloc[0], icycle=i)#,\
                #    save_w=True)
                chi[i] = chi2
                innov[i] = innv
                dof[i] = ds
        ## additive inflation
        #if linf:
        #    logger.info("==additive inflation==")
        #    if pt == "mlef" or pt == "4dmlef":
        #        u[:, 1:] += np.random.randn(u.shape[0], u.shape[1]-1)
        #    else:
        #        u += np.random.randn(u.shape[0], u.shape[1])
        if ft=="ensemble":
            if pt == "mlef" or pt == "mlefw" or pt == "4dmlef":
                xa[i] = u[:, 0]
            else:
                xa[i] = np.mean(u, axis=1)
        else:
            xa[i] = u
        if i < na-1:
            if a_window > 1:
                uf = func.forecast(u)
                if (i+1+a_window <= na):
                    if ft=="ensemble":
                        xa[i+1:i+1+a_window] = np.mean(uf, axis=2)
                        xf[i+1:i+1+a_window] = np.mean(uf, axis=2)
                    else:
                        xa[i+1:i+1+a_window] = uf
                        xf[i+1:i+1+a_window] = uf
                    ii = 0
                    for k in range(i+1,i+1+a_window):
                        if pt=="4dvar":
                            stda[k] = np.sqrt(np.trace(pa)/nx)
                        else:
                            patmp = analysis.calc_pf(uf[ii], pa, k)
                            stda[k] = np.sqrt(np.trace(patmp)/nx)
                        ii += 1
                else:
                    if ft=="ensemble":
                        xa[i+1:na] = np.mean(uf[:na-i-1], axis=2)
                        xf[i+1:na] = np.mean(uf[:na-i-1], axis=2)
                    else:
                        xa[i+1:na] = uf[:na-i-1]
                        xf[i+1:na] = uf[:na-i-1]
                    ii = 0
                    for k in range(i+1,na):
                        if pt=="4dvar":
                            stda[k] = np.sqrt(np.trace(pa)/nx)
                        else:
                            patmp = analysis.calc_pf(uf[ii], pa, k)
                            stda[k] = np.sqrt(np.trace(patmp)/nx)
                        ii += 1
                u = uf[-1]
                pf = analysis.calc_pf(u, pa, i+1)
            else:
                u = func.forecast(u)
                pf = analysis.calc_pf(u, pa, i+1)

            if ft=="ensemble":
                if pt == "mlef" or pt == "mlefw" or pt == "4dmlef":
                    xf[i+1] = u[:, 0]
                else:
                    xf[i+1] = np.mean(u, axis=1)
            else:
                xf[i+1] = u
        if a_window > 1:
            for k in range(i, min(i+a_window,na)):
                e[k] = np.sqrt(np.mean((xa[k, :] - xt[k, :])**2))
                xdmean += np.abs(xa[k,:] - xt[k,:])
        else:
            e[i] = np.sqrt(np.mean((xa[i, :] - xt[i, :])**2))
            xdmean += np.abs(xa[i,:] - xt[i,:])
        stda[i] = np.sqrt(np.trace(pa)/nx)
        xsmean += np.sqrt(np.diag(pa))

    np.save("{}_xf_{}_{}.npy".format(model, op, pt), xf)
    np.save("{}_xa_{}_{}.npy".format(model, op, pt), xa)
    np.save("{}_innv_{}_{}.npy".format(model, op, pt), innov)
    
    np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)
    np.savetxt("{}_stda_{}_{}.txt".format(model, op, pt), stda)
    np.savetxt("{}_chi_{}_{}.txt".format(model, op, pt), chi)
    np.savetxt("{}_dof_{}_{}.txt".format(model, op, pt), dof)

    xdmean /= float(na)
    xsmean /= float(na)
    np.savetxt("{}_xdmean_{}_{}.txt".format(model, op, pt), xdmean)
    np.savetxt("{}_xsmean_{}_{}.txt".format(model, op, pt), xsmean)
