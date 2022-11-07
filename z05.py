import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
from model.kdvb import KdVB
from analysis.obs import Obs
from z05_func import Z05_func

logging.config.fileConfig("logging_config.ini")

model="z05"
# model parameter
nx = 101 # number of grids
dt = 0.01 # time step
dx = 0.5 # grid spacing
nu = 0.07 # diffusion
xmin, xmax = -25, 25 # space limits

# forecast model
step = KdVB(nx, dt, dx, nu, fd=True)

# experiment setup
nt   = 200 # number of steps per assimilation
nmem = 10 # ensemble size (not include control run)
t0c  = -6.0 # t0 for control
t0e  = -7.0 # t0 for initial ensemble
et0  = 2.0 # standard deviation of perturbations for ensemble t0
t0f  = (np.random.randn(nmem)*et0 + t0e).tolist() # t0 for ensemble
na   = 100 # number of analysis cycle
a_window = 1 # (for 4D) assimilation window length
# observation setup
nobs = nx # observation number
obs_s = 0.05 # observation error standard deviation
obsnet = "all" # observation network (all or fixed or targeted)

# forecast type (ensemble or deterministic)
ftype = {
    "mlef":"ensemble","etkf":"ensemble","po":"ensemble","srf":"ensemble","letkf":"ensemble"\
    #,"kf":"deterministic","var":"deterministic"\
    ,"4dmlef":"ensemble","4detkf":"ensemble","4dpo":"ensemble","4dsrf":"ensemble","4dletkf":"ensemble"\
#    ,"4dvar":"deterministic"
}

## default parameter
htype = {"operator": "linear", "perturbation": "mlef"}
linf = False # no inflation
infl_parm = -1.0
lloc = False # no localization
lsig = -1.0
ltlm = False # without tangent linear observation operator
iloc = None # localization type
ss = False  # switch for modulation
getkf = False # same as above

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

# observation number
if len(sys.argv) > 4:
    nobs = int(sys.argv[4])
    if nobs < nx:
        obsnet = 'fixed'
# observation network
if len(sys.argv) > 5 and nobs != nx:
    obsnet = sys.argv[5]

# switch of using/not using tangent linear operator
if len(sys.argv) > 6:
    if sys.argv[6] == "F":
        ltlm = False
# number of ensemble member
if len(sys.argv) > 7:
    nmem = int(sys.argv[7])

# observation operator
obs = Obs(op, obs_s)

# assimilation method
if pt == "mlef":
    from analysis.mlef import Mlef
    analysis = Mlef(nx, nmem, obs, ltlm=ltlm, model=model)
elif pt == "etkf" or pt == "po" or pt == "letkf" or pt == "srf":
    from analysis.enkf import EnKF
    analysis = EnKF(pt, nx, nmem, obs,ltlm=ltlm, model=model)
elif pt == "4detkf" or pt == "4dpo" or pt == "4dletkf" or pt == "4dsrf":
    from analysis.enkf4d import EnKF4d
    #a_window = 5
    analysis = EnKF4d(pt, nx, nmem, obs, step, nt, a_window, \
        ltlm=ltlm, model=model)
elif pt == "4dmlef":
    from analysis.mlef4d import Mlef4d
    analysis = Mlef4d(nx, nmem, obs, step, nt, a_window, \
        ltlm=ltlm, model=model)

# load functions
params = {
    "step":step, "obs":obs, "analysis":analysis\
    ,"nobs":nobs, "obsnet":obsnet\
    ,"t0c":t0c, "t0f":t0f, "t0e":t0e\
    ,"nt":nt, "na":na\
    ,"a_window":a_window, "op":op, "pt":pt, "ft":ft\
    ,"ltlm":ltlm\
    #,"linf":linf, "lloc":lloc\
    #,"infl_parm":infl_parm, "lsig":lsig
    }
func = Z05_func(params)

# main code
if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("==initialize==")
    xt = func.gen_true()
    yobs = func.gen_obs(xt)
    u, xa, xf, pa = func.initialize()
    logger.debug(u.shape)
    func.plot_initial(u[:,0],u[:,1:],xt[0],yobs[0])
    pf = analysis.calc_pf(u, pa, 0)

    x_nda = np.zeros_like(xf)      # No DA
    u_nda = np.zeros_like(u[:, 0]) # No DA
    if ft=="ensemble":
        if pt == "mlef" or pt == "4dmlef":
            u_nda[:] = u[:, 0]
        else:
            u_nda[:] = np.mean(u, axis=1)
    else:
        u_nda[:] = u 
    x_nda[0] = u_nda

    a_time = range(0, na, a_window)
    logger.info("a_time={}".format([time for time in a_time]))
    e = np.zeros(na)
    e_nda = np.zeros(na) # No DA
    stda = np.zeros(na)
    innov = np.zeros((na,yobs.shape[1]))
    chi = np.zeros(na)
    dof = np.zeros(na)
    for i in a_time:
        yloc = yobs[i:i+a_window,:,0]
        y = yobs[i:i+a_window,:,1]
        logger.debug("observation location {}".format(yloc))
        logger.debug("obs={}".format(y))
        logger.info("cycle{} analysis".format(i))
        if i in range(0,10,3):
        #if i < 0:
            if pt[:2] == "4d":
                u, pa, ds = analysis(u, pf, y, yloc, \
                    save_hist=True, save_dh=True, icycle=i)
            else:
                u, pa, spa, innv, chi2, ds = analysis(u, pf, y[0], yloc[0], \
                    #maxiter=1,\
                    save_hist=True, save_dh=True, icycle=i)
                chi[i] = chi2
                innov[i] = innv
        else:
            if pt[:2] == "4d":
                u, pa, ds = analysis(u, pf, y, yloc, icycle=i)
            else:
                u, pa, spa, innv, chi2, ds = analysis(u, pf, y[0], yloc[0],\
                    #maxiter=1,\
                    icycle=i)
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
        #savepa[i] = pa
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
            u_nda = func.forecast(u_nda)
            x_nda[i+1] = u_nda
        if a_window > 1:
            for k in range(i, min(i+a_window,na)):
                e[k] = np.sqrt(np.mean((xa[k, :] - xt[k, :])**2))
        else:
            e[i] = np.sqrt(np.mean((xa[i, :] - xt[i, :])**2))
            stda[i] = np.sqrt(np.trace(pa)/nx)
        e_nda[i] = np.sqrt(np.mean((x_nda[i, :] - xt[i, :])**2))
            
    np.save("{}_xf_{}_{}.npy".format(model, op, pt), xf)
    np.save("{}_xa_{}_{}.npy".format(model, op, pt), xa)
    np.save("{}_xnda_{}_{}.npy".format(model, op, pt), x_nda)
    #np.save("{}_pa_{}_{}.npy".format(model, op, pt), savepa)
    np.save("{}_innv_{}_{}.npy".format(model, op, pt), innov)
    
    np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)
    np.savetxt("{}_enda_{}.txt".format(model, op), e_nda)
    np.savetxt("{}_stda_{}_{}.txt".format(model, op, pt), stda)
    np.savetxt("{}_chi_{}_{}.txt".format(model, op, pt), chi)
    np.savetxt("{}_dof_{}_{}.txt".format(model, op, pt), dof)