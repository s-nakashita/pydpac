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

# observation error standard deviation
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
# inflation parameter (dictionary for each observation type)
infl_gm_l = {"mlef":1.02,"envar":1.1,"etkf":1.02,"po":1.2,"srf":1.2,"letkf":1.02,"kf":1.2,"var":None,
          "4dmlef":1.4,"4detkf":1.3,"4dpo":1.2,"4dsrf":1.2,"4dletkf":1.2,"4dvar":None}
dict_infl_gm = {"linear": infl_gm_l}
infl_lam_l = {"mlef":1.02,"envar":1.05,"envar_nest":1.25,"etkf":1.02,"po":1.2,"srf":1.2,"letkf":1.02,"kf":1.2,"var":None,"var_nest":None,
          "4dmlef":1.4,"4detkf":1.3,"4dpo":1.2,"4dsrf":1.2,"4dletkf":1.2,"4dvar":None}
dict_infl_lam = {"linear": infl_lam_l}
infl_lrg_l = {"envar_nest":1.25,"var_nest":None}
dict_infl_lrg = {"linear":infl_lrg_l}
# localization parameter (dictionary for each observation type)
sig_gm_l = {"mlef":11.0,"envar":11.0,"etkf":2.0,"po":2.0,"srf":2.0,"letkf":11.0,"kf":None,"var":None,
        "4dmlef":2.0,"4detkf":2.0,"4dpo":2.0,"4dsrf":2.0,"4dletkf":2.0,"4dvar":None}
dict_sig_gm = {"linear": sig_gm_l}
sig_lam_l = {"mlef":11.0,"envar":11.0,"envar_nest":11.0,"etkf":2.0,"po":2.0,"srf":2.0,"letkf":11.0,"kf":None,"var":None,"var_nest":None,
        "4dmlef":2.0,"4detkf":2.0,"4dpo":2.0,"4dsrf":2.0,"4dletkf":2.0,"4dvar":None}
dict_sig_lam = {"linear": sig_lam_l}
# forecast type (ensemble or deterministic)
ftype = {"mlef":"ensemble","envar":"ensemble","envar_nest":"ensemble",\
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
params_gm["nobs"]       =  15       # observation number (nobs<=nx_true)
params_gm["obsloctype"] = "regular" # observation location type
params_gm["op"]         = "linear" # observation operator type
params_gm["sigo"]       = sigma[params_gm["op"]] # observation error standard deviation
params_gm["na"]         =  100     # number of analysis cycle
params_gm["nt"]         =  1       # number of step per forecast (=6 hour)
params_gm["namax"]      =  1460    # maximum number of analysis cycle (1 year)
### assimilation method settings
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
global op, pt, sigo, ft
op  = params_lam["op"]
pt  = params_lam["pt"]
sigo= params_lam["sigo"]
ft  = ftype[pt]
global na, a_window
na = params_lam["na"]
nspinup = na // 5
a_window = params_lam["a_window"]
params_gm["ft"] = ft
params_lam["ft"] = ft
if params_gm["linf"] and params_gm["infl_parm"]==-1.0:
    params_gm["infl_parm"] = dict_infl_gm[params_gm["op"]][params_gm["pt"]]
if params_gm["lloc"] and params_gm["lsig"]==-1.0:
    params_gm["lsig"] = dict_sig_gm[params_gm["op"]][params_gm["pt"]]
if params_lam["linf"] and params_lam["infl_parm"]==-1.0:
    params_lam["infl_parm"] = dict_infl_lam[params_lam["op"]][params_lam["pt"]]
if params_lam["lloc"] and params_lam["lsig"]==-1.0:
    params_lam["lsig"] = dict_sig_lam[params_lam["op"]][params_lam["pt"]]
if params_lam["pt"]=="envar_nest" and params_lam["linf"] and params_lam["infl_parm_lrg"]==-1.0:
    params_lam["infl_parm_lrg"] = dict_infl_lrg[params_lam["op"]][params_lam["pt"]]
params_lam["nt"] = params_lam["nt"] * step.lamstep
params_gm["lb"] = params_gm["lb"] * np.pi / 180.0 # degree => radian
params_lam["lb"] = params_lam["lb"] * np.pi / 180.0 # degree => radian
params_lam["lv"] = params_lam["lv"] * np.pi / 180.0 # degree => radian

# observation operator
obs = Obs(op, sigo, seed=params_gm["roseed"]) # for make observations
obs_gm = Obs(op, sigo, ix=step.ix_gm) # for analysis_gm
if params_lam["anlsp"]:
    obs_lam = Obs(op, sigo, ix=step.ix_lam[1:-1], icyclic=False) # analysis_lam
else:
    obs_lam = Obs(op, sigo, ix=step.ix_lam[nsp:-nsp], icyclic=False) # analysis_lam (exclude sponge regions)

# assimilation class
if a_window < 1:
    if pt[:2] == "4d":
        a_window = 5
    else:
        a_window = 1
if pt == "mlef":
    from analysis.mlef import Mlef
    analysis_gm = Mlef(nx_gm, params_gm["nmem"], obs_gm, \
            linf=params_gm["linf"], infl_parm=params_gm["infl_parm"], \
            iloc=params_gm["iloc"], lsig=params_gm["lsig"], \
            ss=params_gm["ss"], getkf=params_gm["getkf"], \
            calc_dist=step.calc_dist_gm, calc_dist1=step.calc_dist1_gm,\
            ltlm=params_gm["ltlm"], incremental=params_gm["incremental"], model=model+"_gm")
#    if params_gm["iloc"] is not None and params_gm["iloc"]>0:
#        rhofile=f"{model}_rho_{op}_{pt}.npy"
#        rhofile_new=f"{model}_rhogm_{op}_{pt}.npy"
#        os.rename(rhofile,rhofile_new)
    analysis_lam = Mlef(nx_lam, params_lam["nmem"], obs_lam, \
            linf=params_lam["linf"], infl_parm=params_lam["infl_parm"], \
            iloc=params_lam["iloc"], lsig=params_lam["lsig"], \
            ss=params_lam["ss"], getkf=params_lam["getkf"], \
            calc_dist=step.calc_dist_lam, calc_dist1=step.calc_dist1_lam,\
            ltlm=params_lam["ltlm"], incremental=params_lam["incremental"], model=model+"_lam")
#    if params_lam["iloc"] is not None and params_lam["iloc"]>0:
#        rhofile=f"{model}_rho_{op}_{pt}.npy"
#        rhofile_new=f"{model}_rholam_{op}_{pt}.npy"
#        os.rename(rhofile,rhofile_new)
elif pt == "envar":
    from analysis.envar import EnVAR
    analysis_gm = EnVAR(nx_gm, params_gm["nmem"], obs_gm, \
            linf=params_gm["linf"], infl_parm=params_gm["infl_parm"], \
            iloc=params_gm["iloc"], lsig=params_gm["lsig"], \
            ss=params_gm["ss"], getkf=params_gm["getkf"], \
            calc_dist=step.calc_dist_gm, calc_dist1=step.calc_dist1_gm,\
            ltlm=params_gm["ltlm"], incremental=params_gm["incremental"], model=model+"_gm")
    if params_lam["anlsp"]:
        analysis_lam = EnVAR(nx_lam-2, params_lam["nmem"], obs_lam, \
            linf=params_lam["linf"], infl_parm=params_lam["infl_parm"], \
            iloc=params_lam["iloc"], lsig=params_lam["lsig"], \
            ss=params_lam["ss"], getkf=params_lam["getkf"], \
            calc_dist=step.calc_dist_lam, calc_dist1=step.calc_dist1_lam,\
            ltlm=params_lam["ltlm"], incremental=params_lam["incremental"], model=model+"_lam")
    else:
        analysis_lam = EnVAR(nx_lam-2*nsp, params_lam["nmem"], obs_lam, \
            linf=params_lam["linf"], infl_parm=params_lam["infl_parm"], \
            iloc=params_lam["iloc"], lsig=params_lam["lsig"], \
            ss=params_lam["ss"], getkf=params_lam["getkf"], \
            calc_dist=step.calc_dist_lam, calc_dist1=step.calc_dist1_lam,\
            ltlm=params_lam["ltlm"], incremental=params_lam["incremental"], model=model+"_lam")
elif pt == "envar_nest":
    from analysis.envar import EnVAR
    from analysis.envar_nest import EnVAR_nest
    analysis_gm = EnVAR(nx_gm, params_gm["nmem"], obs_gm, pt="envar_nest", \
            linf=params_gm["linf"], infl_parm=params_gm["infl_parm"], \
            iloc=params_gm["iloc"], lsig=params_gm["lsig"], \
            ss=params_gm["ss"], getkf=params_gm["getkf"], \
            calc_dist=step.calc_dist_gm, calc_dist1=step.calc_dist1_gm,\
            ltlm=params_gm["ltlm"], incremental=params_gm["incremental"], model=model+"_gm")
    if params_lam["anlsp"]:
        analysis_lam = EnVAR_nest(nx_lam-2, params_lam["nmem"], obs_lam, \
            step.ix_gm, step.ix_lam[1:-1], ntrunc=params_lam["ntrunc"],\
            linf=params_lam["linf"], infl_parm=params_lam["infl_parm"], infl_parm_lrg=params_lam["infl_parm_lrg"], \
            iloc=params_lam["iloc"], lsig=params_lam["lsig"], \
            ss=params_lam["ss"], getkf=params_lam["getkf"], \
            calc_dist=step.calc_dist_lam, calc_dist1=step.calc_dist1_lam,\
            ltlm=params_lam["ltlm"], incremental=params_lam["incremental"], model=model+"_lam")
    else:
        analysis_lam = EnVAR_nest(nx_lam-2*nsp, params_lam["nmem"], obs_lam, \
            step.ix_gm, step.ix_lam[nsp:-nsp], ntrunc=params_lam["ntrunc"],\
            linf=params_lam["linf"], infl_parm=params_lam["infl_parm"], infl_parm_lrg=params_lam["infl_parm_lrg"], \
            iloc=params_lam["iloc"], lsig=params_lam["lsig"], \
            ss=params_lam["ss"], getkf=params_lam["getkf"], \
            calc_dist=step.calc_dist_lam, calc_dist1=step.calc_dist1_lam,\
            ltlm=params_lam["ltlm"], incremental=params_lam["incremental"], model=model+"_lam")
elif pt == "etkf" or pt == "po" or pt == "letkf" or pt == "srf":
    from analysis.enkf import EnKF
    analysis_gm = EnKF(pt, nx_gm, params_gm["nmem"], obs_gm, \
        linf=params_gm["linf"], infl_parm=params_gm["infl_parm"], \
        iloc=params_gm["iloc"], lsig=params_gm["lsig"], \
        ss=params_gm["ss"], getkf=params_gm["getkf"], \
        ltlm=params_gm["ltlm"], \
        calc_dist=step.calc_dist_gm, calc_dist1=step.calc_dist1_gm, model=model+"_gm")
    analysis_lam = EnKF(pt, nx_lam, params_lam["nmem"], obs_lam, \
        linf=params_lam["linf"], infl_parm=params_lam["infl_parm"], \
        iloc=params_lam["iloc"], lsig=params_lam["lsig"], \
        ss=params_lam["ss"], getkf=params_lam["getkf"], \
        ltlm=params_lam["ltlm"], \
        calc_dist=step.calc_dist_lam, calc_dist1=step.calc_dist1_lam, model=model+"_lam")
elif pt == "kf":
    from analysis.kf import Kf
    analysis_gm = Kf(obs_gm, 
    infl=params_gm["infl_parm"], linf=params_gm["linf"], 
    step=step.gm, nt=params_gm["nt"], model=model+"_gm")
    analysis_lam = Kf(obs_lam, 
    infl=params_lam["infl_parm"], linf=params_lam["linf"], 
    step=step.lam, nt=params_lam["nt"], model=model+"_lam")
elif pt == "var":
    from analysis.var import Var
    #bmatdir = f"model/lorenz/ng{nx_gm}nl{nx_lam}kg{nk_gm}kl{nk_lam}nsp{nsp}p{po}F{int(F)}b{b:.1f}c{c:.1f}"
    #f = os.path.join(parent_dir,bmatdir,"B_gmfull.npy")
#    nmcobs=240
#    bmatdir = f"data/l05nest/nmc_obs{nmcobs}"
#    f = os.path.join(parent_dir,bmatdir,"l05nest_B48m24_gm.npy")
#    try:
#        bmat_gm = np.load(f)
#    except FileNotFoundError or OSError:
#        bmat_gm = None
    bmat_gm = None
    analysis_gm = Var(obs_gm, nx_gm, ix=step.ix_gm,  
    sigb=params_gm["sigb"], lb=params_gm["lb"], functype=params_gm["functype"], a=params_gm["a"], bmat=bmat_gm, cyclic=True, \
    calc_dist1=step.calc_dist1_gm, model=model+"_gm")
    #f = os.path.join(parent_dir,bmatdir,"B_lam.npy")
#    f = os.path.join(parent_dir,bmatdir,"l05nest_B48m24_lam.npy")
#    try:
#        bmat_lam = np.load(f)
#        if not params_lam["anlsp"]:
#            bmat_lam = bmat_lam[nsp:-nsp,nsp:-nsp]
#    except FileNotFoundError or OSError:
#        bmat_lam = None
    bmat_lam = None
    if params_lam["anlsp"]:
        analysis_lam = Var(obs_lam, nx_lam-2, ix=step.ix_lam[1:-1], ioffset=1,
        sigb=params_lam["sigb"], lb=params_lam["lb"], functype=params_lam["functype"], a=params_lam["a"], bmat=bmat_lam, cyclic=False, \
        calc_dist1=step.calc_dist1_lam, model=model+"_lam")
    else:
        analysis_lam = Var(obs_lam, nx_lam-2*nsp, ix=step.ix_lam[nsp:-nsp], ioffset=nsp, 
        sigb=params_lam["sigb"], lb=params_lam["lb"], functype=params_lam["functype"], a=params_lam["a"], bmat=bmat_lam, cyclic=False, \
        calc_dist1=step.calc_dist1_lam, model=model+"_lam")
elif pt == "var_nest":
    from analysis.var_nest import Var_nest
    from analysis.var import Var
    #bmatdir = f"model/lorenz/ng{nx_gm}nl{nx_lam}kg{nk_gm}kl{nk_lam}nsp{nsp}p{po}F{int(F)}b{b:.1f}c{c:.1f}"
    #f = os.path.join(parent_dir,bmatdir,"B_gmfull.npy")
#    nmcobs=240
#    bmatdir = f"data/l05nest/nmc_obs{nmcobs}"
#    f = os.path.join(parent_dir,bmatdir,"l05nest_B48m24_gm.npy")
#    try:
#        bmat_gm = np.load(f)
#    except FileNotFoundError or OSError:
#        bmat_gm = None
    bmat_gm = None
    analysis_gm = Var(obs_gm, nx_gm, pt="var_nest", ix=step.ix_gm, 
    sigb=params_gm["sigb"], lb=params_gm["lb"], functype=params_gm["functype"], a=params_gm["a"], bmat=bmat_gm, cyclic=True, \
    calc_dist1=step.calc_dist1_gm, model=model+"_gm")
    #f = os.path.join(parent_dir,bmatdir,"B_lam.npy")
#    f = os.path.join(parent_dir,bmatdir,"l05nest_B48m24_lam.npy")
#    try:
#        bmat_lam = np.load(f)
#        if not params_lam["anlsp"]:
#            bmat_lam = bmat_lam[nsp:-nsp,nsp:-nsp]
#    except FileNotFoundError or OSError:
#        bmat_lam = None
    bmat_lam = None
    #f = os.path.join(parent_dir,bmatdir,"B_gm.npy")
#    f = os.path.join(parent_dir,bmatdir,"l05nest_V48m24.npy")
#    try:
#        vmat = np.load(f)
#        if not params_lam["anlsp"]:
#            i0 = np.argmin(np.abs(step.ix_gm-step.ix_lam[0]))
#            if step.ix_gm[i0]<step.ix_lam[0]: i0+=1
#            i1 = np.argmin(np.abs(step.ix_gm-step.ix_lam[-1]))
#            if step.ix_gm[i1]>step.ix_lam[-1]: i1-=1
#            ii0 = np.argmin(np.abs(step.ix_gm-step.ix_lam[nsp]))
#            if step.ix_gm[ii0]<step.ix_lam[nsp]: ii0+=1
#            ii1 = np.argmin(np.abs(step.ix_gm-step.ix_lam[-nsp]))
#            if step.ix_gm[ii1]>step.ix_lam[-nsp]: ii1-=1
#            vmat = vmat[ii0-i0+1:ii1-i0+2,ii0-i0+1:ii1-i0+2]
#    except FileNotFoundError or OSError:
#        vmat = None
    vmat = None
    if params_lam["crosscov"]:
#        #f = os.path.join(parent_dir,bmatdir,"E_lg.npy")
#        f = os.path.join(parent_dir,bmatdir,"l05nest_B48m24_gm2lam.npy")
#        try:
#            ebkmat = np.load(f)
#            ekbmat = ebkmat.T
#        except FileNotFoundError or OSError:
#            ebkmat = None
        ebkmat = None
#        #f = os.path.join(parent_dir,bmatdir,"E_gl.npy")
#        #try:
#        #    ekbmat = np.load(f)
#        #except FileNotFoundError or OSError:
#            ekbmat = None
        ekbmat = None
    else:
        ebkmat=None
        ekbmat=None
    if params_lam["anlsp"]:
        analysis_lam = Var_nest(obs_lam, step.ix_gm, step.ix_lam[1:-1], ioffset=1,
        sigb=params_lam["sigb"], lb=params_lam["lb"], functype=params_lam["functype"], a=params_lam["a"], bmat=bmat_lam, cyclic=False, 
        sigv=params_lam["sigv"], lv=params_lam["lv"], a_v=params_lam["a_v"], ntrunc=params_lam["ntrunc"], vmat=vmat, 
        crosscov=params_lam["crosscov"], ebkmat=ebkmat, ekbmat=ekbmat,
        calc_dist1=step.calc_dist1_lam, calc_dist1_gm=step.calc_dist1_gm,
        model=model+"_lam")
    else:
        analysis_lam = Var_nest(obs_lam, step.ix_gm, step.ix_lam[nsp:-nsp], ioffset=nsp,
        sigb=params_lam["sigb"], lb=params_lam["lb"], functype=params_lam["functype"], a=params_lam["a"], bmat=bmat_lam, cyclic=False, 
        sigv=params_lam["sigv"], lv=params_lam["lv"], a_v=params_lam["a_v"], ntrunc=params_lam["ntrunc"], vmat=vmat, 
        crosscov=params_lam["crosscov"], ebkmat=ebkmat, ekbmat=ekbmat,
        calc_dist1=step.calc_dist1_lam, calc_dist1_gm=step.calc_dist1_gm,
        model=model+"_lam")
else:
    print("not implemented yet")
    exit()
"""
elif pt == "4dvar":
    from analysis.var4d import Var4d
    #a_window = 5
    sigb_gm = params_gm["sigb"] * np.sqrt(float(a_window))
    analysis_gm = Var4d(obs_gm, step.gm, params_gm["nt"], a_window,
    sigb=sigb_gm, lb=params_gm["lb"], model=model+"_gm")
    sigb_lam = params_lam["sigb"] * np.sqrt(float(a_window))
    analysis_lam = Var4d(obs_lam, step.lam, params_lam["nt"], a_window,
    sigb=sigb_lam, lb=params_lam["lb"], model=model+"_lam")
elif pt == "4detkf" or pt == "4dpo" or pt == "4dletkf" or pt == "4dsrf":
    from analysis.enkf4d import EnKF4d
    #a_window = 5
    analysis_gm = EnKF4d(pt, nx_gm, params_gm["nmem"], obs_gm, step.gm, params_gm["nt"], a_window, \
        linf=params_gm["linf"], infl_parm=params_gm["infl_parm"], 
        iloc=params_gm["iloc"], lsig=params_gm["lsig"], calc_dist=step.calc_dist_gm, calc_dist1=step.calc_dist1_gm, \
        ltlm=params_gm["ltlm"], model=model+"_gm")
    analysis_lam = EnKF4d(pt, nx_lam, params_lam["nmem"], obs_lam, step.lam, params_lam["nt"], a_window, \
        linf=params_lam["linf"], infl_parm=params_lam["infl_parm"], 
        iloc=params_lam["iloc"], lsig=params_lam["lsig"], calc_dist=step.calc_dist_lam, calc_dist1=step.calc_dist1_lam, \
        ltlm=params_lam["ltlm"], model=model+"_lam")
elif pt == "4dmlef":
    #a_window = 5
    from analysis.mlef4d import Mlef4d
    analysis_gm = Mlef4d(nx_gm, params_gm["nmem"], obs_gm, step.gm, params_gm["nt"], a_window, \
            linf=params_gm["linf"], infl_parm=params_gm["infl_parm"], \
            iloc=params_gm["iloc"], lsig=params_gm["lsig"], calc_dist=step.calc_dist_gm, calc_dist1=step.calc_dist1_gm, \
            ltlm=params_gm["ltlm"], incremental=params_gm["incremental"], model=model+"_gm")
    analysis_lam = Mlef4d(nx_lam, params_lam["nmem"], obs_lam, step.lam, params_lam["nt"], a_window, \
            linf=params_lam["linf"], infl_parm=params_lam["infl_parm"], \
            iloc=params_lam["iloc"], lsig=params_lam["lsig"], calc_dist=step.calc_dist_lam, calc_dist1=step.calc_dist1_lam, \
            ltlm=params_lam["ltlm"], incremental=params_lam["incremental"], model=model+"_lam")
"""

# functions load
func = L05nest_func(step,obs,params_gm,params_lam)

if __name__ == "__main__":
    from scipy.interpolate import interp1d
    opt=0
    if len(sys.argv)>2:
        opt=int(sys.argv[2])
    logger = logging.getLogger(__name__)
    logger.info("==initialize==")
    xt, yobs, iobs_lam = func.get_true_and_obs(obsloctype=params_gm["obsloctype"])
    u_gm, xa_gm, xf_gm, u_lam, xa_lam, xf_lam = func.initialize(opt=opt)
    logger.debug(u_gm.shape)
    logger.debug(u_lam.shape)
    if u_gm.ndim == 2:
        func.plot_initial(u_gm[:,0], u_lam[:,0], xt[0], uens_gm=u_gm[:,1:], uens_lam=u_lam[:,1:], method=pt)
    else:
        func.plot_initial(u_gm, u_lam, xt[0], method=pt)
    #pa_gm  = np.zeros((nx_gm, nx_gm))
    #pa_lam  = np.zeros((nx_lam, nx_lam))
    pf_gm = analysis_gm.calc_pf(u_gm, cycle=0)
    pf_lam = analysis_lam.calc_pf(u_lam, cycle=0)
    pa_gm = pf_gm.copy()
    pa_lam = pf_lam.copy()
    xsa_gm = np.zeros((xa_gm.shape[0],pa_gm.shape[0]))
    xsa_lam = np.zeros((xa_lam.shape[0],pa_lam.shape[0]))

    a_time = range(0, na, a_window)
    logger.info("a_time={}".format([time for time in a_time]))
    e_gm = np.zeros(na)
    stda_gm = np.zeros(na)
    xdmean_gm = np.zeros(nx_gm)
    xsmean_gm = np.zeros(pa_gm.shape[0])
    ef_gm = np.zeros(na)
    stdf_gm = np.zeros(na)
    xdfmean_gm = np.zeros(nx_gm)
    xsfmean_gm = np.zeros(pf_gm.shape[0])
    e_lam = np.zeros(na)
    stda_lam = np.zeros(na)
    xdmean_lam = np.zeros(nx_lam)
    xsmean_lam = np.zeros(pa_lam.shape[0])
    ef_lam = np.zeros(na)
    stdf_lam = np.zeros(na)
    xdfmean_lam = np.zeros(nx_lam)
    xsfmean_lam = np.zeros(pf_lam.shape[0])
    innov_gm = np.zeros((na,yobs.shape[1]*a_window))
    chi_gm = np.zeros(na)
    dof_gm = np.zeros(na)
    innov_lam = np.zeros((na,yobs.shape[1]*a_window))
    chi_lam = np.zeros(na)
    dof_lam = np.zeros(na)
    
    stdf_gm[0] = np.sqrt(np.trace(pf_gm)/pf_gm.shape[0])
    stdf_lam[0] = np.sqrt(np.trace(pf_lam)/pf_lam.shape[0])
    if nspinup <= 0:
        xsfmean_gm += np.diag(pf_gm)
        xsfmean_lam += np.diag(pf_lam)
    if params_gm["extfcst"]:
        ## extended forecast
        xf12_gm = np.zeros((na+1,nx_gm))
        xf24_gm = np.zeros((na+3,nx_gm))
        xf48_gm = np.zeros((na+7,nx_gm))
        xf12_lam = np.zeros((na+1,nx_lam))
        xf24_lam = np.zeros((na+3,nx_lam))
        xf48_lam = np.zeros((na+7,nx_lam))
        utmp_gm = u_gm.copy()
        utmp_lam = u_lam.copy()
        utmp_gm, utmp_lam = func.forecast(utmp_gm,utmp_lam)
        if ft=="ensemble":
            if pt == "mlef" or pt == "4dmlef":
                xf12_gm[1] = utmp_gm[:, 0]
                xf12_lam[1] = utmp_lam[:, 0]
            else:
                xf12_gm[1] = np.mean(utmp_gm, axis=1)
                xf12_lam[1] = np.mean(utmp_lam, axis=1)
        else:
            xf12_gm[1] = utmp_gm
            xf12_lam[1] = utmp_lam
        for j in range(2):
            utmp_gm, utmp_lam = func.forecast(utmp_gm,utmp_lam)
        if ft=="ensemble":
            if pt == "mlef" or pt == "4dmlef":
                xf24_gm[3] = utmp_gm[:, 0]
                xf24_lam[3] = utmp_lam[:, 0]
            else:
                xf24_gm[3] = np.mean(utmp_gm, axis=1)
                xf24_lam[3] = np.mean(utmp_lam, axis=1)
        else:
            xf24_gm[3] = utmp_gm
            xf24_lam[3] = utmp_lam
        for j in range(4):
            utmp_gm, utmp_lam = func.forecast(utmp_gm,utmp_lam)
        if ft=="ensemble":
            if pt == "mlef" or pt == "4dmlef":
                xf48_gm[7] = utmp_gm[:, 0]
                xf48_lam[7] = utmp_lam[:, 0]
            else:
                xf48_gm[7] = np.mean(utmp_gm, axis=1)
                xf48_lam[7] = np.mean(utmp_lam, axis=1)
        else:
            xf48_gm[7] = utmp_gm
            xf48_lam[7] = utmp_lam
    nanl = 0
    for i in a_time:
        yloc = yobs[i:min(i+a_window,na),:,0]
        y = yobs[i:min(i+a_window,na),:,1]
        yloc_lam = yloc[iobs_lam[i:min(i+a_window,na)]==1.0]
        y_lam = y[iobs_lam[i:min(i+a_window,na),:]==1.0]
        logger.info("observation location {}".format(yloc))
        logger.info("obs={}".format(y))
        if i >= params_lam["lamstart"]:
            logger.info("iobs_lam={}".format(iobs_lam[i,]))
            logger.info("observation location in LAM {} {}".format  (yloc_lam,yloc_lam.shape))
            logger.info("obs in LAM={} {}".format(y_lam,y_lam.shape))
        logger.info("cycle{} analysis : window length {}".format(i,y.shape[0]))
        save_dh = params_gm["save_dh"]
        save_hist = params_gm["save_hist"]
        ##if a_window > 1:
        if pt[:2] == "4d":
            args_gm = (u_gm,pf_gm,y,yloc)
#            if params_lam["anlsp"]:
#                if pt == "var_nest":
#                    args_lam = (u_lam,pf_lam,y_lam,yloc_lam,u_gm,step.ix_lam)
#                else:
#                    args_lam = (u_lam,pf_lam,y_lam,yloc_lam)
#            else:
#                if pt == "var_nest":
#                    args_lam = (u_lam[nsp:-nsp],pf_lam[nsp:-nsp,nsp:-nsp],y_lam,yloc_lam,u_gm,step.ix_lam[nsp:-nsp])
#                else:
#                    args_lam = (u_lam[nsp:-nsp],pf_lam[nsp:-nsp,nsp:-nsp],y_lam,yloc_lam)
        else:
            args_gm = (u_gm,pf_gm,y[0],yloc[0])
        ua_gm, pa_gm, _, innv, chi2, ds = analysis_gm(*args_gm, \
                save_hist=save_hist, save_dh=save_dh, icycle=i)
        #pafile=f"{model}_pa_{op}_{pt}_cycle{i}.npy"
        #pafile_new=f"{model}_pagm_{op}_{pt}_cycle{i}.npy"
        #os.rename(pafile,pafile_new)
        chi_gm[i:min(i+a_window,na)] = chi2
        dof_gm[i:min(i+a_window,na)] = ds
        innov_gm[i:min(i+a_window,na),:innv.size] = innv
        if i >= params_lam["lamstart"]:
            if params_lam["anlsp"]:
                #if pt == "var_nest":
                #    args_lam = (u_lam,pf_lam,y_lam,yloc_lam,u_gm) #,step.ix_lam)
                #else:
                args_lam = (u_lam[1:-1],pf_lam,y_lam,yloc_lam)
                if pt == "var_nest" or pt == "envar_nest":
                    args_lam = (u_lam[1:-1],pf_lam,y_lam,yloc_lam,u_gm)
                u_tmp, pa_lam, _, innv, chi2, ds = analysis_lam(*args_lam, \
                    save_hist=save_hist, save_dh=save_dh, icycle=i)
                u_lam[1:-1] = u_tmp[:,...]
            else:
                #if pt == "var_nest":
                #    args_lam = (u_lam[nsp:-nsp],pf_lam[nsp:-nsp,nsp:-nsp],y_lam,yloc_lam,u_gm) #,step.ix_lam[nsp:-nsp])
                #else:
                args_lam = (u_lam[nsp:-nsp],pf_lam,y_lam,yloc_lam)
                if pt == "var_nest" or pt == "envar_nest":
                    args_lam = (u_lam[nsp:-nsp],pf_lam,y_lam,yloc_lam,u_gm)
                u_tmp, pa_lam, _, innv, chi2, ds = analysis_lam(*args_lam, \
                    save_hist=save_hist, save_dh=save_dh, icycle=i)
                u_lam[nsp:-nsp] = u_tmp[:,...]
                #pa_lam[nsp:-nsp,nsp:-nsp] = pa_tmp[:,:]
            if ft=="ensemble":
                pa_lam = analysis_lam.calc_pf(u_lam)
            #pafile=f"{model}_pa_{op}_{pt}_cycle{i}.npy"
            #pafile_new=f"{model}_palam_{op}_{pt}_cycle{i}.npy"
            #os.rename(pafile,pafile_new)
            chi_lam[i:min(i+a_window,na)] = chi2
            dof_lam[i:min(i+a_window,na)] = ds
            for ii in range(i,min(i+a_window,na)):
                innov_lam[ii,iobs_lam[ii]==1.0] = innv[:]
        else:
            gm2lam = interp1d(step.ix_gm,ua_gm,axis=0)
            u_lam = gm2lam(step.ix_lam)
            pa_lam = analysis_lam.calc_pf(u_lam, pa=pa_lam, cycle=i)
        u_gm = ua_gm.copy()
        ## additive inflation
        #if linf:
        #    logger.info("==additive inflation==")
        #    if pt == "mlef" or pt == "4dmlef":
        #        u[:, 1:] += np.random.randn(u.shape[0], u.shape[1]-1)
        #    else:
        #        u += np.random.randn(u.shape[0], u.shape[1])
        if ft=="ensemble":
            if pt == "mlef" or pt == "4dmlef":
                xa_gm[i] = u_gm[:, 0]
                xa_lam[i] = u_lam[:, 0]
            else:
                xa_gm[i] = np.mean(u_gm, axis=1)
                xa_lam[i] = np.mean(u_lam, axis=1)
        else:
            xa_gm[i] = u_gm
            xa_lam[i] = u_lam
        if i < na-1:
            if a_window > 1:
                uf_gm, uf_lam = func.forecast(u_gm,u_lam)
                if (i+1+a_window <= na):
                    if ft=="ensemble":
                        xa_gm[i+1:i+1+a_window] = np.mean(uf_gm, axis=2)
                        xf_gm[i+1:i+1+a_window] = np.mean(uf_gm, axis=2)
                        xa_lam[i+1:i+1+a_window] = np.mean(uf_lam, axis=2)
                        xf_lam[i+1:i+1+a_window] = np.mean(uf_lam, axis=2)
                    else:
                        xa_gm[i+1:i+1+a_window] = uf_gm
                        xf_gm[i+1:i+1+a_window] = uf_gm
                        xa_lam[i+1:i+1+a_window] = uf_lam
                        xf_lam[i+1:i+1+a_window] = uf_lam
                    ii = 0
                    for k in range(i+1,i+1+a_window):
                        if pt=="4dvar":
                            stda_gm[k] = np.sqrt(np.trace(pa_gm)/nx_gm)
                            stda_lam[k] = np.sqrt(np.trace(pa_lam)/nx_lam)
                        else:
                            pa_gmtmp = analysis_gm.calc_pf(uf_gm[ii], pa=pa_gm, cycle=k)
                            stda_gm[k] = np.sqrt(np.trace(pa_gmtmp)/nx_gm)
                            pa_lamtmp = analysis_lam.calc_pf(uf_lam[ii], pa=pa_lam, cycle=k)
                            stda_lam[k] = np.sqrt(np.trace(pa_lamtmp)/nx_lam)
                        ii += 1
                else:
                    if ft=="ensemble":
                        xa_gm[i+1:na] = np.mean(uf_gm[:na-i-1], axis=2)
                        xf_gm[i+1:na] = np.mean(uf_gm[:na-i-1], axis=2)
                        xa_lam[i+1:na] = np.mean(uf_lam[:na-i-1], axis=2)
                        xf_lam[i+1:na] = np.mean(uf_lam[:na-i-1], axis=2)
                    else:
                        xa_gm[i+1:na] = uf_gm[:na-i-1]
                        xf_gm[i+1:na] = uf_gm[:na-i-1]
                        xa_lam[i+1:na] = uf_lam[:na-i-1]
                        xf_lam[i+1:na] = uf_lam[:na-i-1]
                    ii = 0
                    for k in range(i+1,na):
                        if pt=="4dvar":
                            stda_gm[k] = np.sqrt(np.trace(pa_gm)/nx_gm)
                            stda_lam[k] = np.sqrt(np.trace(pa_lam)/pa_lam.shape[0])
                        else:
                            pa_gmtmp = analysis_gm.calc_pf(uf_gm[ii], pa=pa_gm, cycle=k)
                            stda_gm[k] = np.sqrt(np.trace(pa_gmtmp)/nx_gm)
                            pa_lamtmp = analysis_lam.calc_pf(uf_lam[ii], pa=pa_lam, cycle=k)
                            stda_lam[k] = np.sqrt(np.trace(pa_lamtmp)/pa_lam.shape[0])
                        ii += 1
                u_gm = uf_gm[-1]
                u_lam = uf_lam[-1]
            else:
                u_gm, u_lam = func.forecast(u_gm,u_lam)
            
            if ft=="ensemble":
                if pt == "mlef" or pt == "4dmlef":
                    xf_gm[i+1] = u_gm[:, 0]
                    xf_lam[i+1] = u_lam[:, 0]
                else:
                    xf_gm[i+1] = np.mean(u_gm, axis=1)
                    xf_lam[i+1] = np.mean(u_lam, axis=1)
            else:
                xf_gm[i+1] = u_gm
                xf_lam[i+1] = u_lam
            pf_gm = analysis_gm.calc_pf(u_gm, pa=pa_gm, cycle=i+1)
            pf_lam = analysis_lam.calc_pf(u_lam, pa=pa_lam, cycle=i+1)
            stdf_gm[i+1] = np.sqrt(np.trace(pf_gm)/pf_gm.shape[0])
            stdf_lam[i+1] = np.sqrt(np.trace(pf_lam)/pf_lam.shape[0])
            if i>=nspinup:
                xsfmean_gm += np.diag(pf_gm)
                xsfmean_lam += np.diag(pf_lam)
        
            if params_gm["extfcst"]:
                ## extended forecast
                utmp_gm = u_gm.copy()
                utmp_lam = u_lam.copy()
                utmp_gm, utmp_lam = func.forecast(utmp_gm,utmp_lam) #6h->12h
                if ft=="ensemble":
                    if pt == "mlef" or pt == "4dmlef":
                        xf12_gm[i+2] = utmp_gm[:, 0]
                        xf12_lam[i+2] = utmp_lam[:, 0]
                    else:
                        xf12_gm[i+2] = np.mean(utmp_gm, axis=1)
                        xf12_lam[i+2] = np.mean(utmp_lam, axis=1)
                else:
                    xf12_gm[i+2] = utmp_gm
                    xf12_lam[i+2] = utmp_lam
                utmp_gm, utmp_lam = func.forecast(utmp_gm,utmp_lam) #12h->18h
                utmp_gm, utmp_lam = func.forecast(utmp_gm,utmp_lam) #18h->24h
                if ft=="ensemble":
                    if pt == "mlef" or pt == "4dmlef":
                        xf24_gm[i+4] = utmp_gm[:, 0]
                        xf24_lam[i+4] = utmp_lam[:, 0]
                    else:
                        xf24_gm[i+4] = np.mean(utmp_gm, axis=1)
                        xf24_lam[i+4] = np.mean(utmp_lam, axis=1)
                else:
                    xf24_gm[i+4] = utmp_gm
                    xf24_lam[i+4] = utmp_lam
                utmp_gm, utmp_lam = func.forecast(utmp_gm,utmp_lam) #24h->30h
                utmp_gm, utmp_lam = func.forecast(utmp_gm,utmp_lam) #30h->36h
                utmp_gm, utmp_lam = func.forecast(utmp_gm,utmp_lam) #36h->42h
                utmp_gm, utmp_lam = func.forecast(utmp_gm,utmp_lam) #42h->48h
                if ft=="ensemble":
                    if pt == "mlef" or pt == "4dmlef":
                        xf48_gm[i+8] = utmp_gm[:, 0]
                        xf48_lam[i+8] = utmp_lam[:, 0]
                    else:
                        xf48_gm[i+8] = np.mean(utmp_gm, axis=1)
                        xf48_lam[i+8] = np.mean(utmp_lam, axis=1)
                else:
                    xf48_gm[i+8] = utmp_gm
                    xf48_lam[i+8] = utmp_lam
                if ft=="ensemble" and i >= 100:
                    #np.save("{}_pfgm_{}_{}_cycle{}.npy".format(model, op, pt, i), pf_gm)
                    #np.save("{}_pflam_{}_{}_cycle{}.npy".format(model, op, pt, i), pf_lam)
                    #tmp_lam2gm = interp1d(step.ix_lam,u_lam,axis=0)
                    #i0=np.argmin(np.abs(step.ix_gm-step.ix_lam[0]))
                    #if step.ix_gm[i0]<step.ix_lam[0]: i0+=1
                    #i1=np.argmin(np.abs(step.ix_gm-step.ix_lam[-1]))
                    #if step.ix_gm[i1]>step.ix_lam[-1]: i1-=1
                    #utmp_lam2gm = tmp_lam2gm(step.ix_gm[i0:i1+1])
                    #if pt == "mlef" or pt == "4dmlef":
                    #    pf_gmlam = (u_gm[i0:i1+1,1:]-u_gm[i0:i1+1,0].reshape(-1,1))@(utmp_lam2gm[:,1:]-utmp_lam2gm[:,0].reshape(-1,1)).T
                    #else:
                    #    pf_gmlam = (u_gm[i0:i1+1,:]-u_gm[i0:i1+1,:].mean(axis=1).reshape(-1,1))@(utmp_lam2gm-utmp_lam2gm.mean(axis=1).reshape(-1,1)).T/(u_lam.shape[1]-1)
                    #np.save("{}_pfgmlam_{}_{}_cycle{}.npy".format(model, op, pt, i), pf_gmlam)
                    pf48_gm = analysis_gm.calc_pf(utmp_gm, pa=pa_gm, cycle=i+1)
                    pf48_lam = analysis_lam.calc_pf(utmp_lam, pa=pa_lam, cycle=i+1)
                    np.save("{}_pf48gm_{}_{}_cycle{}.npy".format(model, op, pt, i), pf48_gm)
                    np.save("{}_pf48lam_{}_{}_cycle{}.npy".format(model, op, pt, i), pf48_lam)

        if np.isnan(u_gm).any() or np.isnan(u_lam).any():
            e_gm[i:] = np.nan
            e_lam[i:] = np.nan
            ef_gm[i+1:] = np.nan
            ef_lam[i+1:] = np.nan
            stda_gm[i:] = np.nan
            stda_lam[i:] = np.nan
            stdf_gm[i+1:] = np.nan
            stdf_lam[i+1:] = np.nan
            xa_gm[i:,:] = np.nan
            xa_lam[i:,:] = np.nan
            xf_gm[i+1:,:] = np.nan
            xf_lam[i+1:,:] = np.nan
            xsa_gm[i:,:] = np.nan
            xsa_lam[i:,:] = np.nan
            break
        if a_window > 1:
            for k in range(i, min(i+a_window,na)):
                true2gm = interp1d(step.ix_true,xt[k])
                e_gm[k] = np.sqrt(np.mean((xa_gm[k, :] - true2gm(step.ix_gm))**2))
                e_lam[k] = np.sqrt(np.mean((xa_lam[k, :] - true2gm(step.ix_lam))**2))
                ef_gm[k] = np.sqrt(np.mean((xf_gm[k, :] - true2gm(step.ix_gm))**2))
                ef_lam[k] = np.sqrt(np.mean((xf_lam[k, :] - true2gm(step.ix_lam))**2))
                if k>=nspinup:
                    xdmean_gm += (xa_gm[k,:]-true2gm(step.ix_gm))**2
                    xdmean_lam += (xa_lam[k,:]-true2gm(step.ix_lam))**2
                    xdfmean_gm += (xf_gm[k,:]-true2gm(step.ix_gm))**2
                    xdfmean_lam += (xf_lam[k,:]-true2gm(step.ix_lam))**2
                    nanl += 1
        else:
            true2gm = interp1d(step.ix_true,xt[i])
            e_gm[i] = np.sqrt(np.mean((xa_gm[i, :] - true2gm(step.ix_gm))**2))
            e_lam[i] = np.sqrt(np.mean((xa_lam[i, :] - true2gm(step.ix_lam))**2))
            ef_gm[i] = np.sqrt(np.mean((xf_gm[i, :] - true2gm(step.ix_gm))**2))
            ef_lam[i] = np.sqrt(np.mean((xf_lam[i, :] - true2gm(step.ix_lam))**2))
            if i>=nspinup:
                xdmean_gm += (xa_gm[i,:]-true2gm(step.ix_gm))**2
                xdmean_lam += (xa_lam[i,:]-true2gm(step.ix_lam))**2
                xdfmean_gm += (xf_gm[i,:]-true2gm(step.ix_gm))**2
                xdfmean_lam += (xf_lam[i,:]-true2gm(step.ix_lam))**2
                nanl += 1
        stda_gm[i] = np.sqrt(np.trace(pa_gm)/pa_gm.shape[0])
        stda_lam[i] = np.sqrt(np.trace(pa_lam)/pa_lam.shape[0])
        xsa_gm[i] = np.sqrt(np.diag(pa_gm))
        xsa_lam[i] = np.sqrt(np.diag(pa_lam))
        if i>=nspinup:
            xsmean_gm += np.diag(pa_gm)
            xsmean_lam += np.diag(pa_lam)

    np.save("{}_xfgm_{}_{}.npy".format(model, op, pt), xf_gm)
    np.save("{}_xagm_{}_{}.npy".format(model, op, pt), xa_gm)
    np.save("{}_xsagm_{}_{}.npy".format(model, op, pt), xsa_gm)
    np.save("{}_xflam_{}_{}.npy".format(model, op, pt), xf_lam)
    np.save("{}_xalam_{}_{}.npy".format(model, op, pt), xa_lam)
    np.save("{}_xsalam_{}_{}.npy".format(model, op, pt), xsa_lam)
    np.save("{}_innvgm_{}_{}.npy".format(model, op, pt), innov_gm)
    np.save("{}_innvlam_{}_{}.npy".format(model, op, pt), innov_lam)
    
    if params_gm["extfcst"]:
        np.save("{}_xf12gm_{}_{}.npy".format(model, op, pt), xf12_gm)
        np.save("{}_xf24gm_{}_{}.npy".format(model, op, pt), xf24_gm)
        np.save("{}_xf48gm_{}_{}.npy".format(model, op, pt), xf48_gm)
        np.save("{}_xf12lam_{}_{}.npy".format(model, op, pt), xf12_lam)
        np.save("{}_xf24lam_{}_{}.npy".format(model, op, pt), xf24_lam)
        np.save("{}_xf48lam_{}_{}.npy".format(model, op, pt), xf48_lam)

    np.savetxt("{}_e_gm_{}_{}.txt".format(model, op, pt), e_gm)
    np.savetxt("{}_stda_gm_{}_{}.txt".format(model, op, pt), stda_gm)
    np.savetxt("{}_ef_gm_{}_{}.txt".format(model, op, pt), ef_gm)
    np.savetxt("{}_stdf_gm_{}_{}.txt".format(model, op, pt), stdf_gm)
    np.savetxt("{}_chi_gm_{}_{}.txt".format(model, op, pt), chi_gm)
    np.savetxt("{}_dof_gm_{}_{}.txt".format(model, op, pt), dof_gm)
    np.savetxt("{}_e_lam_{}_{}.txt".format(model, op, pt), e_lam)
    np.savetxt("{}_stda_lam_{}_{}.txt".format(model, op, pt), stda_lam)
    np.savetxt("{}_ef_lam_{}_{}.txt".format(model, op, pt), ef_lam)
    np.savetxt("{}_stdf_lam_{}_{}.txt".format(model, op, pt), stdf_lam)
    np.savetxt("{}_chi_lam_{}_{}.txt".format(model, op, pt), chi_lam)
    np.savetxt("{}_dof_lam_{}_{}.txt".format(model, op, pt), dof_lam)

    xdmean_gm = np.sqrt(xdmean_gm/float(nanl))
    xdmean_lam = np.sqrt(xdmean_lam/float(nanl))
    xsmean_gm = np.sqrt(xsmean_gm/float(nanl))
    xsmean_lam = np.sqrt(xsmean_lam/float(nanl))
    np.savetxt("{}_xdmean_gm_{}_{}.txt".format(model, op, pt), xdmean_gm)
    np.savetxt("{}_xsmean_gm_{}_{}.txt".format(model, op, pt), xsmean_gm)
    np.savetxt("{}_xdmean_lam_{}_{}.txt".format(model, op, pt), xdmean_lam)
    np.savetxt("{}_xsmean_lam_{}_{}.txt".format(model, op, pt), xsmean_lam)
    xdfmean_gm = np.sqrt(xdfmean_gm/float(nanl))
    xdfmean_lam = np.sqrt(xdfmean_lam/float(nanl))
    xsfmean_gm = np.sqrt(xsfmean_gm/float(nanl))
    xsfmean_lam = np.sqrt(xsfmean_lam/float(nanl))
    np.savetxt("{}_xdfmean_gm_{}_{}.txt".format(model, op, pt), xdfmean_gm)
    np.savetxt("{}_xsfmean_gm_{}_{}.txt".format(model, op, pt), xsfmean_gm)
    np.savetxt("{}_xdfmean_lam_{}_{}.txt".format(model, op, pt), xdfmean_lam)
    np.savetxt("{}_xsfmean_lam_{}_{}.txt".format(model, op, pt), xsfmean_lam)
    
