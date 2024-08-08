import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.lorenz_nest import L05nest
from model.lorenz_nestm import L05nestm
from analysis.obs import Obs
from l05nest_func import L05nest_func

logging.config.fileConfig("logging_config.ini")
parent_dir = os.path.abspath(os.path.dirname(__file__))

global nx_true, nx_gm, nx_lam, step

model = "l05nest"
if len(sys.argv)>1:
    model = sys.argv[1]
# model parameter
## true
nx_true = 960
nk_true = 32
nks_true = [256,128,64,32]
## GM
gm_same_with_nature = False # DEBUG: Lorenz III used for GM
intgm = 4                    # grid interval
if gm_same_with_nature:
    intgm = 1
nx_gm = nx_true // intgm     # number of points
nk_gm = nk_true // intgm     # advection length scale
nks_gm = np.array(nks_true) // intgm     # advection length scales
dt_gm = 0.05 / 36            # time step (=1/6 hour, Kretchmer et al. 2015)
#dt_gm = 0.05 / 48            # time step (=1/8 hour, Yoon et al. 2012)
## LAM
nx_lam = 240                 # number of LAM points
ist_lam = 240                # first grid index
nsp = 10                     # width of sponge region
po = 1                       # order of relaxation function
intrlx = 1                   # interval of boundary relaxation (K15)
#intrlx = 48                   # interval of boundary relaxation (Y12)
lamstep = 1                  # time steps relative to 1 step of GM
nk_lam = 32                  # advection length
nks_lam = nks_true           # advection lengths
ni = 12                      # spatial filter width
b = 10.0                     # frequency of small-scale perturbation
c = 0.6                      # coupling factor
F = 15.0                     # forcing

# forecast model forward operator
if model == "l05nest":
    step = L05nest(nx_true, nx_gm, nx_lam, nk_gm, nk_lam, \
        ni, b, c, dt_gm, F, intgm, ist_lam, nsp, \
        lamstep=lamstep, intrlx=intrlx, po=po, gm_same_with_nature=gm_same_with_nature)
elif model == "l05nestm":
    step = L05nestm(nx_true, nx_gm, nx_lam, nks_gm, nks_lam, \
        ni, b, c, dt_gm, F, intgm, ist_lam, nsp, \
        lamstep=lamstep, intrlx=intrlx, po=po, gm_same_with_nature=gm_same_with_nature)

np.savetxt("ix_true.txt",step.ix_true)
np.savetxt("ix_gm.txt",step.ix_gm)
np.savetxt("ix_lam.txt",step.ix_lam)

# forecast type (ensemble or deterministic)
ftype = {"mlef":"ensemble","mlef_nest":"ensemble","mlef_nestc":"ensemble",\
    "envar":"ensemble","envar_nest":"ensemble","envar_nestc":"ensemble",\
    "etkf":"ensemble","po":"ensemble","srf":"ensemble","letkf":"ensemble",\
    "kf":"deterministic","var":"deterministic","var_nest":"deterministic",\
    "4dmlef":"ensemble","4detkf":"ensemble","4dpo":"ensemble","4dsrf":"ensemble","4dletkf":"ensemble",\
    "4dvar":"deterministic"}

## default parameter
params_gm = dict()
### experiment settings
htype = {"operator": "linear", "perturbation": "mlef"}
params_gm["t0off"]      =  step.nt6h_gm * 4          # initial offset between adjacent members
params_gm["t0c"]        =  step.nt6h_gm * 4 * 60     # t0 for control
params_gm["na"]         =  100     # number of analysis cycle
params_gm["nt"]         =  1       # number of step per forecast (=6 hour)
params_gm["save1h"]     =  False   # save forecast per 1 hour
params_gm["ntmax"]      =  8       # number of forecast steps (=48 hour)
params_gm["namax"]      =  1460    # maximum number of analysis cycle (1 year)
params_gm["rseed"]      = None # random seed
params_gm["op"]         = "linear" # observation operator type
params_gm["pt"]         = "mlef"   # assimilation method
params_gm["nmem"]       =  40      # ensemble size (include control run)
params_gm["a_window"]   =  0       # assimilation window length
params_gm["sigb"]       =  1.0     # (For var & 4dvar) background error standard deviation
params_gm["functype"]   = "gc5"  # (For var & 4dvar) background error correlation function
if model=="l05nest":
    params_gm["lb"]     = 24.6    # (For var & 4dvar) correlation length for background error covariance in degree
    params_gm["a"]      = -0.2  # (For var & 4dvar) background error correlation function shape parameter
else:
    params_gm["lb"]     = 16.93
    params_gm["a"]      = 0.22
params_gm["linf"]       =  False   # inflation flag
params_gm["infl_parm"]  = -1.0     # multiplicative inflation coefficient
params_gm["lloc"]       =  False   # localization flag
params_gm["lsig"]       = -1.0     # localization radius
params_gm["iloc"]       =  None    # localization type
params_gm["ss"]         =  False   # (For model space localization) statistical resampling flag
params_gm["getkf"]      =  False   # (For model space localization) gain form resampling flag
params_gm["ltlm"]       =  True    # flag for tangent linear observation operator
params_gm["incremental"] = False   # (For mlef & 4dmlef) flag for incremental form
params_gm["rseed"]      = None # random seed
params_gm["roseed"]     = None # random seed for synthetic observation
params_gm["extfcst"]    = False # extended forecast
params_gm["save_dh"]    = True  # save intermediate files
params_gm["save_hist"]  = True  # save hist files
params_gm["saveGM"]     = False # preparing precomputed GM forecasts for LBC of LAM
#
params_lam = params_gm.copy()
params_lam["lamstart"]  = 0 # first cycle of LAM analysis and forecast
params_lam["anlsp"]     = True # True: analyzed in the sponge region
params_lam["sigb"]      =  0.8     # (For var & 4dvar) background error standard deviation
params_lam["sigv"]      =  1.8     # (For var_nest) GM background error standard deviation in LAM space
params_lam["functype"]  = "gc5"  # (For var & 4dvar) background error correlation function
if model=="l05nest":
    params_lam["lb"]    = 26.5     # (For var & 4dvar) correlation length for background error covariance in degree
    params_lam["a"]     = -0.2  # (For var & 4dvar) background error correlation function shape parameter
    params_lam["lv"]    = 23.5     # (For var_nest) GM correlation length for background error covariance in LAM space in degree
    params_lam["a_v"]   = -0.1  # (For var_nest) background error correlation function shape parameter
else:
    params_lam["lb"]    = 28.77
    params_lam["a"]     = -0.11
    params_lam["lv"]    = 12.03
    params_lam["a_v"]   = 0.12
params_lam["ntrunc"]    = 12    # (For var_nest & envar_nest) truncation number for GM error covariance
params_lam["infl_parm_lrg"] = -1.0  # (For envar_nest) inflation parameter for GM error covariance
params_lam["crosscov"]  = False     # (For var_nest & envar_nest) whether correlation between GM and LAM is considered or not
params_lam["ortho"]     = True  # (For envar_nestc) ridge regression
params_lam["coef_a"]    = 0.0
params_lam["ridge"]     = False # (For envar_nestc) ridge regression
params_lam["ridge_dx"]  = False 
params_lam["reg"]       = False
params_lam["hyper_mu"]  = 0.1
params_lam["preGM"]     = False # using precomputed GM forecasts as LBC
params_lam["preGMdir"]  = './'
params_lam["preGMda"]   = "mlef"

## update from configure file
sys.path.append('./')
try:
    from config_gm import params as params_gm_new
    params_gm.update(params_gm_new)
except ImportError:
    pass
try:
    from config_lam import params as params_lam_new
    params_lam.update(params_lam_new)
except ImportError:
    pass
global op, pt, ft
op  = params_lam["op"]
pt  = params_lam["pt"]
ft  = ftype[pt]
global na
na = params_lam["na"]
nspinup = 40 #na // 5
params_gm["ft"] = ft
params_lam["ft"] = ft
params_lam["nt"] = params_lam["nt"] * step.lamstep

# observation operator
obs = Obs(op, 1.0) # dummy

# functions load
func = L05nest_func(step,obs,params_gm,params_lam,model=model)

if __name__ == "__main__":
    from scipy.interpolate import interp1d
    opt=0
    if len(sys.argv)>2:
        opt=int(sys.argv[2])
    logger = logging.getLogger(__name__)
    logger.info("==initialize==")
    
    nanl = 0
    if params_lam["preGM"]:
        if ft=='ensemble':
            uf_gm = np.load(func.preGMdir/f"{model}_gm_ufeext_{op}_{func.preGMda}.npy")
        else:
            uf_gm = np.load(func.preGMdir/f"{model}_gm_ufext_{op}_{func.preGMda}.npy")
        logger.info(f"uf_gm.shape={uf_gm.shape}")
    else:
        uf_gm = []
        #ua_gm = np.load(f"xagm_{op}_{pt}.npy")
    uf_lam = []
    #ua_lam = np.load(f"xalam_{op}_{pt}.npy")
    for i in range(nspinup,na):
        if params_lam["preGM"]:
            #ua_gm = np.load(func.preGMdir/f"data/{func.preGMda}/{model}_gm_ua_{op}_{func.preGMda}_cycle{i}.npy")
            uf1_gm = uf_gm[i-nspinup,]
        else:
            ua_gm = np.load(f"data/{pt}/{model}_gm_ua_{op}_{pt}_cycle{i}.npy")
            if ft=="ensemble":
                if params_gm["save1h"]:
                    uf1_gm = np.zeros((params_gm["ntmax"]*6+1,ua_gm.shape[0],ua_gm.shape[1]))
                else:
                    uf1_gm = np.zeros((params_gm["ntmax"]+1,ua_gm.shape[0],ua_gm.shape[1]))
            else:
                if params_gm["save1h"]:
                    uf1_gm = np.zeros((params_gm["ntmax"]*6+1,nx_gm))
                else:
                    uf1_gm = np.zeros((params_gm["ntmax"]+1,nx_gm))
            uf1_gm[0,] = ua_gm #[i,]
        gm2lam = interp1d(step.ix_gm,uf1_gm[0,],axis=0)
        ua_lam = gm2lam(step.ix_lam)
        if i>=params_lam["lamstart"]:
            uanl = np.load(f"data/{pt}/{model}_lam_ua_{op}_{pt}_cycle{i}.npy")
            ua_lam[1:-1,] = uanl[:]
        if ft=='ensemble':
            if params_gm["save1h"]:
                uf1_lam = np.zeros((params_gm["ntmax"]*6+1,ua_lam.shape[0],ua_lam.shape[1]))
            else:
                uf1_lam = np.zeros((params_gm["ntmax"]+1,ua_lam.shape[0],ua_lam.shape[1]))
        else:
            if params_gm["save1h"]:
                uf1_lam = np.zeros((params_gm["ntmax"]*6+1,nx_lam))
            else:
                uf1_lam = np.zeros((params_gm["ntmax"]+1,nx_lam))
        uf1_lam[0,] = ua_lam #[i,]
        logger.info("cycle{} extended forecast length {}".format(i,uf1_lam.shape[0]))
        
        for j in range(params_gm["ntmax"]):
            if params_gm["save1h"]:
                if params_lam["preGM"]:
                    _, uf1_lam[6*j,],uf1_lam_1h = func.forecast(uf1_gm[j,],uf1_lam[j,],u_gm_pre=uf1_gm[j+1,],save1h=params_gm["save1h"])
                else:
                    uf1_gm[6*j,], uf1_lam[6*j,], uf1_gm_1h, uf_lam_1h, = func.forecast(uf1_gm[j,],uf1_lam[j,],save1h=params_gm["save1h"])
                    uf1_gm[6*j+1:6*j+6,] = uf1_gm_1h[:5,]
                uf1_lam[6*j+1:6*j+6,] = uf1_lam_1h[:5,]
            else:
                if params_lam["preGM"]:
                    _, uf1_lam[j+1,] = func.forecast(uf1_gm[j,],uf1_lam[j,],u_gm_pre=uf1_gm[j+1,])
                else:
                    uf1_gm[j+1,], uf1_lam[j+1,] = func.forecast(uf1_gm[j,],uf1_lam[j,])
        if not params_lam["preGM"]:
            uf_gm.append(uf1_gm)
        uf_lam.append(uf1_lam)
    if not params_lam["preGM"]:
        uf_gm = np.array(uf_gm)
        logger.info(f"uf_gm.shape={uf_gm.shape}")
        if ft=='ensemble':
            if params_gm["save1h"]:
                np.save("{}_gm_ufeext_p1h_{}_{}.npy".format(model, op, pt), uf_gm)
            else:
                np.save("{}_gm_ufeext_{}_{}.npy".format(model, op, pt), uf_gm)
        else:
            if params_gm["save1h"]:
                np.save("{}_gm_ufext_p1h_{}_{}.npy".format(model, op, pt), uf_gm)
            else:
                np.save("{}_gm_ufext_{}_{}.npy".format(model, op, pt), uf_gm)
    uf_lam = np.array(uf_lam)
    logger.info(f"uf_lam.shape={uf_lam.shape}")
    if params_lam["preGM"]:
        if ft=='ensemble':
            if params_gm["save1h"]:
                np.save("{}_lam_ufeext_p1h_preGM_{}_{}.npy".format(model, op, pt), uf_lam)
            else:
                np.save("{}_lam_ufeext_preGM_{}_{}.npy".format(model, op, pt), uf_lam)
        else:
            if params_gm["save1h"]:
                np.save("{}_lam_ufext_p1h_preGM_{}_{}.npy".format(model, op, pt), uf_lam)
            else:
                np.save("{}_lam_ufext_preGM_{}_{}.npy".format(model, op, pt), uf_lam)
    else:
        if ft=='ensemble':
            if params_gm["save1h"]:
                np.save("{}_lam_ufeext_p1h_{}_{}.npy".format(model, op, pt), uf_lam)
            else:
                np.save("{}_lam_ufeext_{}_{}.npy".format(model, op, pt), uf_lam)
        else:
            if params_gm["save1h"]:
                np.save("{}_lam_ufext_p1h_{}_{}.npy".format(model, op, pt), uf_lam)
            else:
                np.save("{}_lam_ufext_{}_{}.npy".format(model, op, pt), uf_lam)
