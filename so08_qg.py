import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
import matplotlib.pyplot as plt
from model.qgmain import QG
from analysis.obs import Obs
from so08_qg_func import SO08_qg_func
from math import tau

logging.config.fileConfig("logging_config.ini")

model = "qg"
# model parameter
nx = 129
ny = nx
dt = 1.25
x = np.linspace(0.0,1.0,nx)
y = np.linspace(0.0,1.0,ny)
itermax = 1, 1, 100
beta, f, eps, a, tau0 = 1.0, 1600.0, 1.0e-5, 2.0e-12, -tau
tol = 1.0e-4
params = beta, f, eps, a, tau0, itermax, tol
# forecast model forward operator
step = QG(nx,ny,dt,y,*params)

nmem = 25 #ensemble size (include control run)
t0off = 200 # initial offset between adacent members
t0c = 4000 # t0 for control
t0true = 3500 # t0 for truth
# t0 for ensemble members
if nmem%2 == 0: # even
    t0m = [t0c + t0off//2 + t0off * i for i in range(nmem//2)]
    t0f = t0m + [t0c + t0off//2 + t0off * i for i in range(-nmem//2, 0)]
else: # odd
    t0m = [t0c + t0off//2 + t0off * i for i in range(-(nmem-1)//2, (nmem-1)//2)]
    t0f = [t0c] + t0m
na =   10 # number of analysis
namax = 300 # max number of analysis 
nt =  4    # number of step per cycle

a_window = 1 # assimilation window length

nobs = 300 # observation number

# observation error standard deviation
sigma = {"linear": 4.0}
# inflation parameter (dictionary for each observation type)
infl_l = {"mlef":1.2,"etkf":1.2,"po":1.2,"srf":1.2,"letkf":1.2,"kf":1.2,"var":None,
          "4dmlef":1.3,"4detkf":1.3,"4dpo":1.2,"4dsrf":1.2,"4dletkf":1.2,"4dvar":None}
dict_infl = {"linear": infl_l}
# localization parameter (dictionary for each observation type)
sig_l = {"mlef":2.0,"etkf":2.0,"po":2.0,"srf":2.0,"letkf":2.0,"kf":None,"var":None,
        "4dmlef":2.0,"4detkf":2.0,"4dpo":2.0,"4dsrf":2.0,"4dletkf":2.0,"4dvar":None}
dict_sig = {"linear": sig_l}
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
iloc = None
ss = False
getkf = False

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
        ## only for mlef
        if len(sys.argv) > 8:
            iloc = int(sys.argv[8])
        else:
            # default is R-localization
            iloc = 0

# observation operator
obs = Obs(op, sigma[op],ndims=2,ni=nx,nj=ny)

# assimilation method
state_size = nx*ny
if pt == "mlef":
    if iloc == 0:
        from analysis.mlef_rloc import Mlef_rloc
        analysis = Mlef_rloc(pt, nmem, obs, \
            linf=linf, infl_parm=infl_parm, \
            lsig=lsig, calc_dist=step.calc_dist, calc_dist1=step.calc_dist1,\
            ltlm=ltlm, model=model)
    else:
        from analysis.mlef import Mlef
        analysis = Mlef(pt, state_size, nmem, obs, \
            linf=linf, infl_parm=infl_parm, \
            iloc=iloc, lsig=lsig, ss=False, gain=False, \
            calc_dist=step.calc_dist, calc_dist1=step.calc_dist1,\
            ltlm=ltlm, model=model)
elif pt == "etkf" or pt == "po" or pt == "letkf" or pt == "srf":
    from analysis.enkf import EnKF
    analysis = EnKF(pt, state_size, nmem, obs, \
        linf=linf, infl_parm=infl_parm, \
        iloc=iloc, lsig=lsig, ss=True, getkf=False, \
        ltlm=ltlm, \
        calc_dist=step.calc_dist, calc_dist1=step.calc_dist1, model=model)
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
    analysis = EnKS(pt, state_size, nmem, obs, step, nt, a_window, \
        linf=linf, infl_parm=infl_parm, 
        iloc=iloc, lsig=lsig, calc_dist=step.calc_dist, calc_dist1=step.calc_dist1, \
        ltlm=ltlm, model=model)
elif pt == "4dmlef":
    if iloc == 0:
        from analysis.mles_rloc import Mles_rloc
        analysis = Mles_rloc(pt, nmem, obs, step, nt, a_window, \
            linf=linf, infl_parm=infl_parm, \
            lsig=lsig, calc_dist=step.calc_dist, calc_dist1=step.calc_dist1, \
            ltlm=ltlm, model=model)
    else:
        from analysis.mles import Mles
        analysis = Mles(pt, state_size, nmem, obs, step, nt, a_window, \
            linf=linf, infl_parm=infl_parm, \
            iloc=iloc, lsig=lsig, calc_dist=step.calc_dist, calc_dist1=step.calc_dist1, \
            ltlm=ltlm, model=model)

# functions load
params = {"step":step, "obs":obs, "analysis":analysis, "nobs":nobs, \
    "t0c":t0c, "t0f":t0f, "t0true":t0true, "nt":nt, "na":na,\
    "namax":namax, "a_window":a_window, "op":op, "pt":pt, "ft":ft,\
    "linf":linf, "lloc":lloc, "ltlm":ltlm,\
    "infl_parm":infl_parm, "lsig":lsig}
func = SO08_qg_func(params)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("==initialize==")
    qt, psit, yobs = func.get_true_and_obs()
    u, qa,psia, qf,psif, pa = func.initialize()
    logger.debug(u.shape)
    func.plot_initial(u[state_size:,:], qt[nt,], psit[nt,], yobs[0,])
    pf = analysis.calc_pf(u, pa, 0)
    
    a_time = range(0, na, a_window)
    logger.info("a_time={}".format([time for time in a_time]))
    e = np.zeros(na)
    stda = np.zeros(na)
    innov = np.zeros((na,yobs.shape[1]))
    chi = np.zeros(na)
    dof = np.zeros(na)
    for i in a_time:
        yloc = yobs[i:i+a_window,:,:2]
        y = yobs[i:i+a_window,:,2]
        logger.debug("observation location {}".format(yloc))
        logger.debug("obs={}".format(y))
        exit()
        logger.info("cycle{} analysis".format(i))
        #if i in [1, 50, 100, 150, 200, 250]:
        if i < 0:
            #if a_window > 1:
            if pt[:2] == "4d":
                u, pa, ds = analysis(u, pf, y, yloc, \
                    save_hist=True, save_dh=True, icycle=i)
            else:
                u, pa, spa, innv, chi2, ds = analysis(u, pf, y[0], yloc[0], \
                    save_hist=True, save_dh=True, icycle=i)
                chi[i] = chi2
                innov[i] = innv
        else:
            #if a_window > 1:
            if pt[:2] == "4d":
                u, pa, ds = analysis(u, pf, y, yloc, icycle=i)
            else:
                u, pa, spa, innv, chi2, ds = analysis(u, pf, y[0], yloc[0], icycle=i)
                chi[i] = chi2
                innov[i] = innv
        ## additive inflation
        #if linf:
        #    logger.info("==additive inflation==")
        #    if pt == "mlef" or pt == "4dmlef":
        #        u[:, 1:] += np.random.randn(u.shape[0], u.shape[1]-1)
        #    else:
        #        u += np.random.randn(u.shape[0], u.shape[1])
        if ft=="ensemble":
            if pt == "mlef" or pt == "4dmlef":
                xa[i] = u[:, 0]
            else:
                xa[i] = np.mean(u, axis=1)
        else:
            xa[i] = u
        dof[i] = ds
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
                        patmp = analysis.calc_pf(uf[ii], pa, k)
                        stda[k] = np.sqrt(np.trace(patmp)/nx)
                        ii += 1
                u = uf[-1]
                pf = analysis.calc_pf(u, pa, i+1)
            else:
                u = func.forecast(u)
                pf = analysis.calc_pf(u, pa, i+1)

            if ft=="ensemble":
                if pt == "mlef" or pt == "4dmlef":
                    xf[i+1] = u[:, 0]
                else:
                    xf[i+1] = np.mean(u, axis=1)
            else:
                xf[i+1] = u
        if a_window > 1:
            for k in range(i, min(i+a_window,na)):
                e[k] = np.sqrt(np.mean((xa[k, :] - xt[k, :])**2))
        else:
            e[i] = np.sqrt(np.mean((xa[i, :] - xt[i, :])**2))
            stda[i] = np.sqrt(np.trace(pa)/nx)

    np.save("{}_xf_{}_{}.npy".format(model, op, pt), xf)
    np.save("{}_xa_{}_{}.npy".format(model, op, pt), xa)
    np.save("{}_innv_{}_{}.npy".format(model, op, pt), innov)
    
    np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)
    np.savetxt("{}_stda_{}_{}.txt".format(model, op, pt), stda)
    np.savetxt("{}_chi_{}_{}.txt".format(model, op, pt), chi)
    np.savetxt("{}_dof_{}_{}.txt".format(model, op, pt), dof)