import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from model_jax.lorenz import L05j
from analysis.obs import Obs
from l05j_func import L05j_func

logging.config.fileConfig("logging_config.ini")

global nx, F, dt, dx

ltype = 1
if len(sys.argv)>1:
    ltype = int(sys.argv[1])
if ltype==1:
    model = "l05I"
    # model parameter
    nx    = 40     # number of points
    Ftrue = 8.0    # true forcing
    Fmodel= 8.0    # model forcing
    dt    = 0.05 / 6  # time step (=1 hour)
    # nature model forward operator
    naturestep = L05j(nx, dt, (Ftrue,), ltype=ltype)
    # forecast model forward operator
    step = L05j(nx, dt, (Fmodel,), ltype=ltype)
elif ltype==2:
    model = "l05II"
    # model parameter
    nx    = 240
    nk    = 8
    Ftrue = 10.0
    Fmodel= 10.0
    dt    = 0.05 / 6
    # nature model forward operator
    naturestep = L05j(nx, dt, (nk,Ftrue), ltype=ltype)
    # forecast model forward operator
    step = L05j(nx, dt, (nk,Fmodel), ltype=ltype)
elif ltype==3:
    model = "l05III"
    # model parameter
    nx    = 960
    nk    = 32
    ni    = 12
    b     = 10.0
    c     = 0.6
    Ftrue = 15.0
    Fmodel= 15.0
    dt    = 0.05 / 36
    filmat = set_filmat(nx,ni)
    # nature model forward operator
    naturestep = L05j(nx, dt, (nk,filmat,b,c,Ftrue), ltype=ltype)
    # forecast model forward operator
    step = L05j(nx, dt, (nk,filmat,b,c,Fmodel), ltype=ltype)

x = np.arange(nx)
dx = x[1] - x[0]
np.savetxt("ix.txt", x)

# observation error standard deviation
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}

# forecast type (ensemble or deterministic)
ftype = {"mlef":"ensemble","mlefw":"ensemble","etkf":"ensemble","po":"ensemble","srf":"ensemble","letkf":"ensemble",\
    "kf":"deterministic","var":"deterministic",\
    "4dmlef":"ensemble","4detkf":"ensemble","4dpo":"ensemble","4dsrf":"ensemble","4dletkf":"ensemble",\
    "4dvar":"deterministic"}

## default parameter
params = dict()
### experiment settings
htype = {"operator": "linear", "perturbation": "mlef"}
params["t0off"]      =  24      # initial offset between adjacent members
params["t0c"]        =  500     # t0 for control
params["na"]         =  100     # number of analysis cycle
params["nt"]         =   6      # number of step per forecast (=6 hours)
params["ntmax"]      =  20      # number of forecast steps (=120 hours)
params["namax"]      =  1460    # maximum number of analysis cycle (1 year)
params["extfcst"]    = False
### assimilation method settings
params["nobs"]       =  40      # observation number (nobs<=nx)
params["op"]         = "linear" # observation operator type
params["pt"]         = "mlef"   # assimilation method
params["nmem"]       =  20      # ensemble size (include control run)
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
params["seed"]       =  0       # random seed

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
params["ft"] = ft

# observation operator (dummy)
obs = Obs(op, sigma[op])

# functions load
func = L05j_func(step,step,obs,params)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("==initialize==")
    
    ua = np.load(f"{model}_xf00_{op}_{pt}.npy")
    logger.info(f"ua.shape={ua.shape}")
    uf = []
    for i in range(na):
        #if ft=='ensemble':
        #    uf1 = np.zeros((params["ntmax"]+1,ua.shape[1],ua.shape[2]))
        #else:
        #    uf1 = np.zeros((params["ntmax"]+1,ua.shape[1]))
        logger.info("cycle{} extended forecast length {}".format(i,params["ntmax"]+1))
        uf1 = [jnp.asarray(ua[i,])]
        for j in range(params["ntmax"]):
            uf1.append(func.forecast(uf1[j]))
        uf.append(uf1)
    uf = jnp.asarray(uf)
    logger.info(f"uf.shape={uf.shape}")
    np.save("{}_ufext_{}_{}.npy".format(model, op, pt), jax.device_get(uf))
