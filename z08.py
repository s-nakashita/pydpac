import sys
import logging
from logging.config import fileConfig
import os
import numpy as np
from model.burgers import Bg
from analysis.obs import Obs
from z08_func import Z08_func

logging.config.fileConfig("logging_config.ini")

global nx, nu, dt, dx

model = "z08"
# model parameter
nx = 81     # number of points
nu = 0.05   # diffusion
dt = 0.0125 # time step

x = np.linspace(-2.0, 2.0, nx)
dx = x[1] - x[0]
np.savetxt("x.txt", x)

# forecast model forward operator
step = Bg(nx, dx, dt, nu)

nmem =    4 # ensemble size (except control run)
t0off =  12 # initial offset between adjacent members
t0true = 20 # t0 for true
t0c =    60 # t0 for control
# t0 for control run and ensemble members
if nmem%2 == 0: # even
    t0m = [t0c + t0off//2 + t0off * i for i in range(-nmem//2, nmem//2)]
    t0f = [t0c] + t0m
else: # odd
    t0m = [t0c + t0off//2 + t0off * i for i in range((nmem+1)//2)]
    t0f = t0m + [t0c + t0off//2 + t0off * i for i in range(-(nmem+1)//2, 0)]
#t0c = t0true
nt =     20 # number of step per forecast
na =     20 # number of analysis
nobs =   81 # number of observation

# observation error
sigma = {"linear": 8.0e-2, "quadratic": 1.0e-3, "cubic": 1.0e-3, "quartic": 1.0e-2, \
    "quadratic-nodiff": 1.0e-3, "cubic-nodiff": 1.0e-3, "quartic-nodiff": 1.0e-2}
# forecast type (ensemble or deterministic)
ftype = {"mlef":"ensemble","etkf":"ensemble",\
    "po":"ensemble","srf":"ensemble","letkf":"ensemble",\
    "kf":"deterministic","var":"deterministic"}

## default parameter
htype = {"operator": "linear", "perturbation": "mlef"}
linf = False
infl_parm = 1.0
lloc = False
lsig = -1.0
ltlm = True

## read from commant options
# observation type
if len(sys.argv) > 1:
    htype["operator"] = sys.argv[1]
# assimilation type
if len(sys.argv) > 2:
    htype["perturbation"] = sys.argv[2]
# number of assimilation type
if len(sys.argv) > 3:
    na = int(sys.argv[3])
# switch of with/without inflation
if len(sys.argv) > 4:
    if sys.argv[4] == "T":
        linf = True
        infl_parm = 1.1
# switch of with/without localization
if len(sys.argv) > 5:
    if sys.argv[5] == "T":
        lloc = True
        lsig = 4.0
# switch of using/not using tangent linear operator
if len(sys.argv) > 6:
    if sys.argv[6] == "F":
        ltlm = False
obs_s = sigma[htype["operator"]]
# observation error
if len(sys.argv) > 7:
    obs_s = float(sys.argv[7])

global op, pt, ft
op = htype["operator"]
pt = htype["perturbation"]
ft = ftype[pt]

# observation operator
obs = Obs(op, obs_s)

# assimilation method
if pt == "mlef":
    from analysis.mlef import Mlef
    analysis = Mlef(pt, nmem, obs, infl_parm, lsig, linf, lloc, ltlm, model=model)
elif pt == "etkf" or pt == "po" or pt == "letkf" or pt == "srf":
    from analysis.enkf import EnKF
    analysis = EnKF(pt, nmem+1, obs, infl_parm, lsig, linf, lloc, ltlm, model=model)
elif pt == "kf":
    from analysis.kf import Kf
    analysis = Kf(pt, obs, infl_parm, linf, step, nt, model=model)
elif pt == "var":
    from analysis.var import Var
    analysis = Var(pt, obs, model=model)

# functions load
params = {"step":step, "obs":obs, "analysis":analysis, \
    "nobs":nobs, "t0off":t0off, "t0true":t0true, "t0f":t0f, \
    "nt":nt, "na":na, "op":op, "pt":pt, "ft":ft,\
    "linf":linf, "lloc":lloc, "ltlm":ltlm}
func = Z08_func(params)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    ut, u, pa = func.gen_true(x)
    pf = analysis.calc_pf(u, pa, 0)
    oberr = int(obs_s*1e4)
    obsfile="obs_{}_{}.npy".format(op, oberr)
    if not os.path.isfile(obsfile):
        print("create obs")
        yobs = func.gen_obs(ut)
        np.save(obsfile, yobs)
    else:
        print("read obs")
        yobs = func.get_obs(obsfile)
    if ft=="ensemble":
        func.plot_initial(u, ut[0], model)
    ua, uf, sqrtpa = func.init_hist(u)
        
    e = np.zeros(na+1)
    if ft == "ensemble":
        e[0] = np.sqrt(np.mean((uf[0, :, 0] - ut[0, :])**2))
    else:
        e[0] = np.sqrt(np.mean((uf[0, :] - ut[0, :])**2))
    chi = np.zeros(na)
    innov = np.zeros((na,yobs.shape[1]))
    dof = np.zeros(na)
    dpa = np.zeros(na)
    ndpa = np.zeros(na)
    for i in range(na):
        yloc = yobs[i,:,0]
        y = yobs[i,:,1]
        logger.debug("observation location {}".format(yloc))
        logger.debug("obs={}".format(y))
        logger.info("cycle{} analysis".format(i))
        if i in range(4):
            u, pa, spa, innv, chi2, ds = analysis(u, pf, y, yloc, \
                save_hist=True, save_dh=True, icycle=i)
        else:
            u, pa, spa, innv, chi2, ds = analysis(u, pf, y, yloc, icycle=i)
            
        ua[i] = u
        sqrtpa[i] = pa
        if pt == "mlef":
            dpa[i] = np.sum(np.diag(pa@pa.T))
            ndpa[i] = np.sum(pa@pa.T) - dpa[i]
        else:
            dpa[i] = np.sum(np.diag(pa))
            ndpa[i] = np.sum(pa) - dpa[i]
        chi[i] = chi2
        dof[i] = ds
        innov[i] = innv
        if i < na-1:
            u = func.forecast(u, pa)
            pf = analysis.calc_pf(u, pa, i+1)
            uf[i+1] = u
        if ft == "ensemble":
            if pt == "mlef":
                e[i+1] = np.sqrt(np.mean((ua[i, :, 0] - ut[i, :])**2))
            else:
                e[i+1] = np.sqrt(np.mean((np.mean(ua[i, :, :], axis=1) - ut[i, :])**2))
        else:
            e[i+1] = np.sqrt(np.mean((ua[i, :] - ut[i, :])**2))
    
    np.save("{}_ut.npy".format(model), ut)
    np.save("{}_uf_{}_{}.npy".format(model, op, pt), uf)
    np.save("{}_ua_{}_{}.npy".format(model, op, pt), ua)
    np.save("{}_pa_{}_{}.npy".format(model, op, pt), sqrtpa)
    np.savetxt("{}_dpa_{}_{}.txt".format(model, op, pt), dpa)
    np.savetxt("{}_ndpa_{}_{}.txt".format(model, op, pt), ndpa)
    
    np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)
    np.savetxt("{}_chi_{}_{}.txt".format(model, op, pt), chi)
    np.savetxt("{}_dof_{}_{}.txt".format(model, op, pt), dof)
    #if ft == "ensemble":
    np.save("{}_innv_{}_{}.npy".format(model, op, pt), innov)