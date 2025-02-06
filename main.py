import os
import logging
from logging.config import fileConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.obs import Obs
from exp_func import Exp_func

logging.config.fileConfig("logging_config.ini")

global sigma
# observation error standard deviation
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
# inflation parameter (dictionary for each observation type)
infl_l = {"mlef":1.05,"envar":1.1,"etkf":1.2,"po":1.2,"srf":1.2,"eakf":1.2,"letkf":1.05,"kf":1.2,"var":None,
          "4dmlef":1.4,"4detkf":1.3,"4dpo":1.2,"4dsrf":1.2,"4dletkf":1.2,"4dvar":None}
infl_q = {"mlef":1.2,"etkf":1.2,"po":1.2,"srf":1.3,"eakf":1.2,"letkf":1.2,"kf":1.2,"var":None,
          "4dmlef":1.4,"4detkf":1.3,"4dpo":1.2,"4dsrf":1.2,"4dletkf":1.2,"4dvar":None}
infl_c = {"mlef":1.2,"etkf":1.5,"po":1.1,"srf":1.8,"eakf":1.2,"letkf":1.3,"kf":1.3,"var":None,"4dvar":None}
infl_qd = {"mlef":1.2,"etkf":1.2,"po":1.2,"srf":1.3,"eakf":1.2,"letkf":1.2,"kf":1.2,"var":None,"4dvar":None}
infl_cd = {"mlef":1.2,"etkf":1.5,"po":1.0,"srf":1.8,"eakf":1.2,"letkf":1.3,"kf":1.3,"var":None,"4dvar":None}
infl_t = {"mlef":1.2,"etkf":1.1,"po":1.0,"srf":1.1,"eakf":1.2,"letkf":1.0,"kf":1.2,"var":None,"4dvar":None}
infl_h = {"mlef":1.3,"etkf":1.1,"po":1.0,"srf":1.1,"eakf":1.2,"letkf":1.0,"kf":1.2,"var":None,"4dvar":None}
dict_infl = {"linear": infl_l, "quadratic": infl_q, "cubic": infl_c, \
    "quadratic-nodiff": infl_qd, "cubic-nodiff": infl_cd, \
        "test": infl_t, "abs": infl_l, "hint": infl_h}
# localization parameter (dictionary for each observation type)
sig_l = {"mlef":3.0,"envar":2.0,"etkf":2.0,"po":2.0,"srf":2.0,"eakf":2.0,"letkf":3.0,"kf":None,"var":None,
        "4dmlef":2.0,"4detkf":2.0,"4dpo":2.0,"4dsrf":2.0,"4dletkf":2.0,"4dvar":None}
sig_q = {"mlef":2.0,"etkf":6.0,"po":6.0,"srf":8.0,"eakf":1.2,"letkf":4.0,"kf":None,"var":None,
        "4dmlef":2.0,"4detkf":6.0,"4dpo":6.0,"4dsrf":8.0,"4dletkf":4.0,"4dvar":None}
sig_c = {"mlef":4.0,"etkf":6.0,"po":6.0,"srf":8.0,"eakf":1.2,"letkf":6.0,"kf":None,"var":None,"4dvar":None}
sig_qd = {"mlef":6.0,"etkf":6.0,"po":6.0,"srf":8.0,"eakf":1.2,"letkf":4.0,"kf":None,"var":None,"4dvar":None}
sig_cd = {"mlef":6.0,"etkf":6.0,"po":6.0,"srf":8.0,"eakf":1.2,"letkf":10.0,"kf":None,"var":None,"4dvar":None}
sig_t = {"mlef":8.0,"etkf":8.0,"po":14.0,"srf":14.0,"eakf":1.2,"letkf":15.0,"kf":None,"var":None,"4dvar":None}
dict_sig = {"linear": sig_l, "quadratic": sig_q, "cubic": sig_c, \
    "quadratic-nodiff": sig_qd, "cubic-nodiff": sig_cd, \
    "test":sig_t, "abs":sig_l, "hint": sig_l}
# forecast type (ensemble or deterministic)
ftype = {"mlef":"ensemble","envar":"ensemble","etkf":"ensemble","po":"ensemble","srf":"ensemble","eakf":"ensemble","letkf":"ensemble",\
    "kf":"deterministic","var":"deterministic",\
    "4dmlef":"ensemble","4detkf":"ensemble","4dpo":"ensemble","4dsrf":"ensemble","4dletkf":"ensemble",\
    "4dvar":"deterministic"}

## default parameter
params = dict()
### experiment settings
htype = {"operator": "linear", "perturbation": "mlef"}
params["t0off"]      =  24      # initial offset between adjacent members
params["t0c"]        =  500     # t0 for control
params["nobs"]       =  40      # observation number (nobs<=nx)
params["obsloctype"] = "regular" # observation location type
params["op"]         = "linear" # observation operator type
params["na"]         =  100     # number of analysis cycle
params["nspinup"]    =  params["na"] // 5 # spinup periods
params["nt"]         =  6       # number of step per forecast (=6 hour)
params["namax"]      =  1460    # maximum number of analysis cycle (1 year)
### assimilation method settings
params["pt"]         = "mlef"   # assimilation method
params["nmem"]       =  20      # ensemble size (include control run)
params["a_window"]   =  0       # assimilation window length
params["sigb"]       =  0.6     # (For var & 4dvar) background error standard deviation
params["functype"]   = "gc5"    # (For var & 4dvar) background error correlation function type
params["lb"]         = -1.0     # (For var & 4dvar) correlation length for background error covariance
params["linf"]       =  False   # inflation flag
params["iinf"]       =  None    # inflation type
params["infl_parm"]  = -1.0     # inflation factor
params["lloc"]       =  False   # localization flag
params["lsig"]       = -1.0     # localization radius
params["iloc"]       =  None    # localization type
params["ss"]         =  False   # (For model space localization) statistical resampling flag
params["getkf"]      =  False   # (For model space localization) gain form resampling flag
params["ltlm"]       =  True    # flag for tangent linear observation operator
params["incremental"] = False   # (For mlef & 4dmlef) flag for incremental form
params["rseed"] = None # random seed
params["roseed"] = None # random seed for obsope
params["extfcst"]    = False    # extended forecast
params["model_error"] = False # valid for l05II, True: perfect model experiment, False: inperfect model experiment
params_keylist = params.keys()

# (optional) save DA details
save_hist_cycles = []

def get_model(model):
    global dt6h
    dt6h = 0.05 # non-dimensional time step for lorenz models equivalent to 6 hours
    kwargs = {}
    if model=="l96":
        from model.lorenz import L96 as Model
        global nx, F, dt, dx
        # model parameter
        nx = 40     # number of points
        F  = 8.0    # forcing
        dt = dt6h / 6  # time step (=1 hour)
        args = nx, dt, F
        ix = np.arange(nx)
    elif model=="l05II":
        from model.lorenz2 import L05II as Model
        # model parameter
        nx = 240       # number of points
        nk = 8         # advection length scale
        F  = 10.0      # forcing
        dt = dt6h / 6  # time step (=1 hour)
        args = nx, nk, dt, F
        ix = np.arange(nx)
    elif model=="l05IIm":
        from model.lorenz2m import L05IIm as Model
        # model parameter
        nx = 240           # number of points
        nk = [64,32,16,8]  # advection length scales
        F  = 15.0          # forcing
        dt = dt6h / 36     # time step (=1/6 hour)
        args = nx, nk, dt, F
        ix = np.arange(nx)
    elif model == "l05III":
        from model.lorenz3 import L05III as Model
        # model parameter
        nx = 960       # number of points
        nk = 32        # advection length scale
        ni = 12        # spatial filter width
        F  = 15.0      # forcing
        b  = 10.0      # frequency of small-scale perturbation
        c  = 0.6       # coupling factor
        dt = dt6h / 6 / 6 # time step (=1/6 hour)
        args = nx, nk, ni, b, c, dt, F
        ix = np.arange(nx)
    elif model == "l05IIIm":
        from model.lorenz3m import L05IIIm as Model
        #global ni, b, c 
        # model parameter
        nx = 960             # number of points
        nk = [256,128,64,32] # advection length scales
        ni = 12              # spatial filter width
        F  = 15.0            # forcing
        b  = 10.0            # frequency of small-scale perturbation
        c  = 0.6             # coupling factor
        dt = dt6h / 36       # time step (=1/6 hour)
        args = nx, nk, ni, b, c, dt, F
        ix = np.arange(nx)
    elif model=="kdvb":
        # KdVB model
        from model.kdvb import KdVB as Model
        # model parameter
        nx = 101          # number of points
        dt = 0.01         # timestep
        dx = 0.5          # grid spacing
        nu = 0.07         # diffusion
        kwargs['fd'] = True # FFT or finite-difference
        args = nx, dt, dx, nu
        sigma.update({"linear":0.05,"quadratic":0.05,"cubic":0.05})
        ix = np.linspace(-25.0,25.0,nx)
    elif model=="burgers":
        # 1-dimensional advection-diffusion model
        from model.burgers import Bg as Model
        # model parameter
        nx = 81           # number of points
        nu = 0.05         # diffusion
        dt = 0.0125       # timestep
        dx = 4.0 / (nx-1) # grid interval
        args = nx, dx, dt, nu
        sigma.update({"linear":8.0e-2,"quadratic":1.0e-3,"cubic":1.0e-3,"quartic":1.0e-3,"quadratic-nodiff":1.0e-3,"cubic-nodiff":1.0e-3,"quartic-nodiff":1.0e-3})
        ix = np.linspace(-2.0,2.0,nx)
    # forecast model forward operator
    step = Model(*args,**kwargs)
    return step, ix

def get_daclass(params,step,obs_mod,model):
    global a_window
    for key in params_keylist:
        if not key in params:
            params[key] = None
    # import assimilation class
    pt = params["pt"]
    a_window = params["a_window"]
    if params["linf"] and params["infl_parm"] == -1.0: 
        params["infl_parm"] = dict_infl[params["op"]][params["pt"]]
    if params["lloc"]: params["lsig"] = dict_sig[params["op"]][params["pt"]]
    if params["lb"] > 0.0:
        params["lb"] = params["lb"] * np.pi / 180.0 # degree => radian
    state_size = step.nx
    if a_window < 1:
        if pt[:2] == "4d":
            a_window = 5
        else:
            a_window = 1
    if pt == "mlef":
        from analysis.mlef import Mlef
        analysis = Mlef(state_size, params["nmem"], obs_mod, \
            linf=params["linf"], iinf=params["iinf"], infl_parm=params["infl_parm"], \
            lloc=params["lloc"], iloc=params["iloc"], lsig=params["lsig"], ss=params["ss"], getkf=params["getkf"], \
            calc_dist=step.calc_dist, calc_dist1=step.calc_dist1,\
            ltlm=params["ltlm"], incremental=params["incremental"], model=model)
    elif pt == "envar":
        from analysis.envar import EnVAR
        analysis = EnVAR(state_size, params["nmem"], obs_mod, \
            linf=params["linf"], iinf=params["iinf"], infl_parm=params["infl_parm"],\
            lloc=params["lloc"], iloc=params["iloc"], lsig=params["lsig"], ss=params["ss"], getkf=params["getkf"], \
            calc_dist=step.calc_dist, calc_dist1=step.calc_dist1,\
            ltlm=params["ltlm"], incremental=params["incremental"], model=model)
    elif pt == "etkf" or pt == "po" or pt == "letkf" or pt == "srf" or pt=="eakf":
        from analysis.enkf import EnKF
        analysis = EnKF(pt, state_size, params["nmem"], obs_mod, \
            linf=params["linf"], iinf=params["iinf"], infl_parm=params["infl_parm"], \
            lloc=params["lloc"], iloc=params["iloc"], lsig=params["lsig"], ss=params["ss"], getkf=params["getkf"], \
            ltlm=params["ltlm"], \
            calc_dist=step.calc_dist, calc_dist1=step.calc_dist1, model=model)
    elif pt == "kf":
        from analysis.kf import Kf
        analysis = Kf(obs_mod, 
        infl=params["infl_parm"], linf=params["linf"], 
        step=step, nt=params["nt"], model=model)
    elif pt == "var":
        from analysis.var import Var
        bmat = None
        analysis = Var(obs_mod, nx, ix=ix, 
        sigb=params["sigb"], lb=params["lb"], functype=params["functype"], a=params["a"], bmat=bmat, \
        calc_dist1=step.calc_dist1, model=model)
    elif pt == "4dvar":
        from analysis.var4d import Var4d
        #a_window = 5
        sigb = params["sigb"] * np.sqrt(float(a_window))
        analysis = Var4d(obs_mod, step, params["nt"], a_window,
        sigb=sigb, lb=params["lb"], model=model)
    elif pt == "4detkf" or pt == "4dpo" or pt == "4dletkf" or pt == "4dsrf":
        from analysis.enkf4d import EnKF4d
        #a_window = 5
        analysis = EnKF4d(pt, state_size, params["nmem"], obs_mod, step, params["nt"], a_window, \
            linf=params["linf"], infl_parm=params["infl_parm"], 
            iloc=params["iloc"], lsig=params["lsig"], calc_dist=step.calc_dist, calc_dist1=step.calc_dist1, \
            ltlm=params["ltlm"], model=model)
    elif pt == "4dmlef":
        #a_window = 5
        from analysis.mlef4d import Mlef4d
        analysis = Mlef4d(state_size, params["nmem"], obs_mod, step, params["nt"], a_window, \
                linf=params["linf"], infl_parm=params["infl_parm"], \
                iloc=params["iloc"], lsig=params["lsig"], calc_dist=step.calc_dist, calc_dist1=step.calc_dist1, \
                ltlm=params["ltlm"], incremental=params["incremental"], model=model)
    return analysis

def main(model,params_in=None,save_results=True):
    
    ## get forecast model
    step, ix = get_model(model)
    nx = step.nx
    dx = ix[1] - ix[0]
    if save_results: np.savetxt("ix.txt", ix)

    if params["model_error"]:
        if model[:5] == 'l05II':
            if model=='l05II':
                model_t = 'l05III'
            elif model=='l05IIm':
                model_t = 'l05IIIm'
            step_t, ix_t = get_model(model_t)
            params["nx_true"] = step_t.nx
            intmod = params["nx_true"]//nx
            ix = np.arange(0,params["nx_true"],intmod)
    else:
        step_t = step
        ix_t = ix.copy()
        params["nx_true"] = nx
    if save_results: np.savetxt("ix_t.txt", ix_t)
    params["ix"] = ix
    params["ix_t"] = ix_t
    
    initopt=0
    if model=='kdvb':
        params["t0c"] = -6.0 # t0 for control
        params["t0e"] = -7.0 # t0 for initial ensemble
        params["et0"] =  2.0 # standard deviation of perturbations for ensemble t0
        params["nt"] = 200
    elif model=='burgers':
        params["t0off"] = 24
        params["t0true"] = 20
        params["t0c"] = 60
        params["nt"] = 20
        params["namax"] = 20
        initopt=1
    else:
        params["t0off"]      =  4*int(dt6h/step.dt)      # initial offset between adjacent members
        params["t0c"]        =  100*int(dt6h/step.dt)     # t0 for control
        params["nt"]         =  int(dt6h/step.dt)      # number of step per forecast (=6 hour)
    if model[:3]=="l05":
        if model[-1]=="m":
            params["lb"]     = 16.93
            params["a"]      = 0.22     # (For var & 4dvar) background error correlation function shape parameter
        else:
            params["lb"]     = 24.6
            params["a"]      = -0.2
    ## update parameters from input arguments
    if params_in is not None:
        params.update(params_in)
    if params["na"] > params["namax"]:
        params["na"] = params["namax"]

    global op, pt, ft
    op = params["op"]
    pt = params["pt"]
    ft = ftype[pt]
    params["ft"] = ft

    # observation operator
    obs = Obs(op, sigma[op], ix=ix_t)
    obs_mod = Obs(op, sigma[op], ix=ix)

    # DA
    analysis = get_daclass(params,step,obs_mod,model)
    
    # functions load
    func = Exp_func(model,step,obs,params,step_t=step_t,save_data=save_results)

    logger.info("==initialize==")
    xt, yobs = func.get_true_and_obs(obsloctype=params["obsloctype"])
    u, xa, xf, pa, xsa = func.initialize(opt=initopt)
    logger.debug(u.shape)
    #if logger.isEnabledFor(logging.DEBUG):
    func.plot_initial(u, xt[0])
    pf = analysis.calc_pf(u, cycle=0)
    
    na = params["na"]
    a_time = range(0, na, a_window)
    logger.info("a_time={}".format([time for time in a_time]))
    e = np.zeros(na)
    stda = np.zeros(na)
    ef = np.zeros(na)
    stdf = np.zeros(na)
    xdmean = np.zeros(nx)
    xsmean = np.zeros(nx)
    xdfmean = np.zeros(nx)
    xsfmean = np.zeros(nx)
    innov = np.zeros((na,yobs.shape[1]*a_window))
    chi = np.zeros(na)
    dof = np.zeros(na)
    
    stdf[0] = np.sqrt(np.trace(pf)/pf.shape[0])
    if params["nspinup"] <= 0:
        xsfmean += np.diag(pf)
    if params["extfcst"]:
        xf12 = np.zeros((na+1,nx))
        xf24 = np.zeros((na+3,nx))
        xf48 = np.zeros((na+7,nx))
        ## extended forecast
        utmp = u.copy()
        logger.debug("id(u)=%s"%id(u))
        logger.debug("id(utmp)=%s"%id(utmp))
        um, utmp = func.forecast(utmp)
        xf12[1] = um
        for j in range(2): # 12h->24h
            um, utmp = func.forecast(utmp)
        xf24[3] = um
        for j in range(4): # 24h->48h
            um, utmp = func.forecast(utmp)
        xf48[7] = um
    
    nanl = 0
    for i in a_time:
        yloc = yobs[i:min(i+a_window,na),:,0]
        y = yobs[i:min(i+a_window,na),:,1]
        logger.debug("observation location {}".format(yloc))
        logger.debug("obs={}".format(y))
        logger.info("cycle{} analysis : window length {}".format(i,y.shape[0]))
        if i in save_hist_cycles:
            if pt[:2] == "4d":
                u, pa = analysis(u, pf, y, yloc, \
                    save_hist=True, save_dh=True, icycle=i)
                for j in range(y.shape[0]):
                    chi[i+j] = analysis.chi2
                    innov[i+j,:analysis.innv.size] = analysis.innv
                    dof[i+j] = analysis.ds
            else:
                u, pa = analysis(u, pf, y[0], yloc[0], \
                    save_hist=True, save_dh=True, icycle=i)
                chi[i] = analysis.chi2
                innov[i] = analysis.innv
                dof[i] = analysis.ds
        else:
            if pt[:2] == "4d":
                u, pa = analysis(u, pf, y, yloc, icycle=i)
                for j in range(y.shape[0]):
                    chi[i+j] = analysis.chi2
                    innov[i+j,:innv.size] = analysis.innv
                    dof[i+j] = analysis.ds
            else:
                u, pa = analysis(u, pf, y[0], yloc[0], icycle=i)#,\
                #    save_w=True)
                chi[i] = analysis.chi2
                innov[i] = analysis.innv
                dof[i] = analysis.ds
        if ft=="ensemble":
            if pt == "mlef" or pt == "4dmlef":
                xa[i] = u[:, 0]
            else:
                xa[i] = np.mean(u, axis=1)
        else:
            xa[i] = u
        if i < na-1:
            if a_window > 1:
                um, uf = func.forecast(u)
                if (i+1+a_window <= na):
                    xa[i+1:i+1+a_window] = um
                    xf[i+1:i+1+a_window] = um
                    ii = 0
                    for k in range(i+1,i+1+a_window):
                        if pt=="4dvar":
                            stda[k] = np.sqrt(np.trace(pa)/nx)
                        else:
                            patmp = analysis.calc_pf(uf[ii], pa, k)
                            stda[k] = np.sqrt(np.trace(patmp)/nx)
                        ii += 1
                else:
                    xa[i+1:na] = um[:na-i-1]
                    xf[i+1:na] = um[:na-i-1]
                    ii = 0
                    for k in range(i+1,na):
                        if pt=="4dvar":
                            stda[k] = np.sqrt(np.trace(pa)/nx)
                        else:
                            patmp = analysis.calc_pf(uf[ii], pa, k)
                            stda[k] = np.sqrt(np.trace(patmp)/nx)
                        ii += 1
                u = uf[-1]
                pf = analysis.calc_pf(u, pa=pa, cycle=i+1)
                um = um[-1]
            else:
                um, u = func.forecast(u)
                pf = analysis.calc_pf(u, pa=pa, cycle=i+1)
            xf[i+1] = um
            stdf[i+1] = np.sqrt(np.trace(pf)/pf.shape[0])
            if i>=params["nspinup"]:
                xsfmean += np.diag(pf)

            if params["extfcst"]:
                ## extended forecast
                utmp = u.copy()
                um, utmp = func.forecast(utmp) #6h->12h
                xf12[i+2] = um
                for j in range(2): #12h->24h
                    um, utmp = func.forecast(utmp)
                xf24[i+4] = um
                for j in range(4): #24h->48h
                    um, utmp = func.forecast(utmp)
                xf48[i+8] = um
        if np.isnan(u).any():
            e[i:] = np.nan
            ef[i+1:] = np.nan
            stda[i:] = np.nan
            stdf[i+1:] = np.nan
            xa[i:,:] = np.nan
            xsa[i:,:] = np.nan
            xf[i+1:,:] = np.nan
            break
        if a_window > 1:
            for k in range(i, min(i+a_window,na)):
                e[k]  = np.sqrt(np.mean((xa[k, :] - xt[k, :])**2))
                ef[k] = np.sqrt(np.mean((xf[k, :] - xt[k, :])**2))
                if k>=params["nspinup"]:
                    xdmean  += (xa[k, :] - xt[k, :])**2
                    xdfmean += (xf[k, :] - xt[k, :])**2
        else:
            e[i]  = np.sqrt(np.mean((xa[i, :] - xt[i, :])**2))
            ef[i] = np.sqrt(np.mean((xf[i, :] - xt[i, :])**2))
            if i>=params["nspinup"]:
                xdmean  += (xa[i, :] - xt[i, :])**2
                xdfmean += (xf[i, :] - xt[i, :])**2
        stda[i] = np.sqrt(np.trace(pa)/nx)
        xsa[i] = np.sqrt(np.diag(pa))
        if i>=params["nspinup"]:
            xsmean += np.diag(pa)
            nanl += 1

    if not save_results:
        return
    np.save("{}_xf_{}_{}.npy".format(model, op, pt), xf)
    np.save("{}_xa_{}_{}.npy".format(model, op, pt), xa)
    np.save("{}_xsa_{}_{}.npy".format(model, op, pt), xsa)
    np.save("{}_innv_{}_{}.npy".format(model, op, pt), innov)
    
    if params["extfcst"]:
        np.save("{}_xf12_{}_{}.npy".format(model, op, pt), xf12)
        np.save("{}_xf24_{}_{}.npy".format(model, op, pt), xf24)
        np.save("{}_xf48_{}_{}.npy".format(model, op, pt), xf48)
    
    np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)
    np.savetxt("{}_stda_{}_{}.txt".format(model, op, pt), stda)
    np.savetxt("{}_ef_{}_{}.txt".format(model, op, pt), ef)
    np.savetxt("{}_stdf_{}_{}.txt".format(model, op, pt), stdf)
    np.savetxt("{}_chi_{}_{}.txt".format(model, op, pt), chi)
    np.savetxt("{}_dof_{}_{}.txt".format(model, op, pt), dof)

    if nanl > 0:
        xdmean = np.sqrt(xdmean/float(nanl))
        xsmean = np.sqrt(xsmean/float(nanl))
        xdfmean = np.sqrt(xdfmean/float(nanl))
        xsfmean = np.sqrt(xsfmean/float(nanl))
        np.savetxt("{}_xdmean_{}_{}.txt".format(model, op, pt), xdmean)
        np.savetxt("{}_xsmean_{}_{}.txt".format(model, op, pt), xsmean)
        np.savetxt("{}_xdfmean_{}_{}.txt".format(model, op, pt), xdfmean)
        np.savetxt("{}_xsfmean_{}_{}.txt".format(model, op, pt), xsfmean)

    if params["iinf"]==-2:
        logger.info(len(analysis.infladap.asave))
        # save adaptive inflation
        np.savetxt("{}_infl_{}_{}.txt".format(model, op, pt), np.array(analysis.infladap.asave))
    if params["iinf"]==-3:
        logger.info(len(analysis.inflfunc.rhosave))
        # save adaptive inflation
        np.savetxt("{}_infl_{}_{}.txt".format(model, op, pt), np.array(analysis.inflfunc.rhosave))
    if len(analysis.inflfunc.pdrsave)>0:
        logger.info(len(analysis.inflfunc.pdrsave))
        # save posterior diagnostic ratio
        np.savetxt("{}_pdr_{}_{}.txt".format(model, op, pt), np.array(analysis.inflfunc.pdrsave))

if __name__ == "__main__":
    import sys
    model = "l96"
    if len(sys.argv) > 1:
        model = sys.argv[1]
    logger = logging.getLogger(__name__)
    ## update parameters from configure file
    sys.path.append('./')
    from config import params as params_in
    main(model,params_in=params_in)