import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.lorenz import step
#from analysis.obs import add_noise, h_operator
from analysis.obs import Obs
#from analysis import kf
from analysis.kf import Kf
#from analysis import var
from analysis.var import Var
#from analysis import var4d
from analysis.var4d import Var4d
#from analysis import mlef
from analysis.mlef import Mlef
#from analysis import enkf
from analysis.enkf import EnKF

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger(__name__)

global nx, F, dt, dx

nx = 40     # number of points
F  = 8.0    # forcing
dt = 0.05 / 6  # time step (=1 hour)
logger.info("nx={} F={} dt={:7.3e}".format(nx, F, dt))
#print("nx={} F={} dt={:7.3e}".format(nx, F, dt))

x = np.linspace(-2.0, 2.0, nx)
dx = x[1] - x[0]
np.savetxt("x.txt", x)

nmem =   20 # ensemble size
t0off =   8 # initial offset between adjacent members
t0c =    500 # t0 for control
            # t0 for ensemble members
t0m = [t0c + t0off//2 + t0off * i for i in range(-nmem//2, nmem//2)]
t0f = [t0c] + t0m
nt =     6 # number of step per forecast (=6 hour)
na =   100 # number of analysis
namax = 1460 # max number of analysis (1 year)

a_window = 1 # assimilation window length

sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0}
htype = {"operator": "linear", "perturbation": "mlef"}
ftype = {"mlef":"ensemble","grad":"ensemble","etkf":"ensemble",\
    "po":"ensemble","srf":"ensemble","letkf":"ensemble",\
        "kf":"deterministic","var":"deterministic","var4d":"deterministic"}
linf = False
lloc = False
ltlm = True
if len(sys.argv) > 1:
    htype["operator"] = sys.argv[1]
if len(sys.argv) > 2:
    htype["perturbation"] = sys.argv[2]
if len(sys.argv) > 3:
    na = int(sys.argv[3])
if len(sys.argv) > 4:
    if sys.argv[4] == "T":
        linf = True
if len(sys.argv) > 5:
    if sys.argv[5] == "T":
        lloc = True
if len(sys.argv) > 6:
    if sys.argv[6] == "F":
        ltlm = False
if htype["perturbation"] == "var4d":
    if len(sys.argv) > 7:
        a_window = int(sys.argv[7])
logger.info("nmem={} t0f={}".format(nmem, t0f))
#print("nmem={} t0f={}".format(nmem, t0f))
logger.info("nt={} na={}".format(nt, na))
#print("nt={} na={}".format(nt, na))
logger.info("htype={} sigma={} ftype={}".format\
    (htype, sigma[htype["operator"]], ftype[htype["perturbation"]]))
#print("htype={} sigma={}".format(htype, sigma[htype["operator"]]))
logger.info("inflation={} localization={} TLM={}".format(linf,lloc,ltlm))
#print("inflation={} localization={} TLM={}".format(linf,lloc,ltlm))
logger.info("Assimilation window size = {}".format(a_window))
#print("Assimilation window size = {}".format(a_window))

obs = Obs(htype["operator"], sigma[htype["operator"]])
mlef = Mlef(htype["perturbation"], obs, 1.1)
enkf = EnKF(htype["perturbation"], obs, 1.1, 4.0)
kf = Kf(htype["perturbation"], obs, 1.1)
var = Var(htype["perturbation"], obs)
var4d = Var4d(htype["perturbation"], obs)

def get_true_and_obs(na, nx):
    f = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
        "data/data.csv")
    truth = pd.read_csv(f)
    xt = truth.values.reshape(na,nx)

    #obs = pd.read_csv("observation_data.csv")
    #y = obs.values.reshape(na,nx)
    y = obs.h_operator(obs.add_noise(xt))

    return xt, y

def init_ctl(nx,t0c,dt,F):
    X0c = np.ones(nx)*F
    X0c[nx//2 - 1] += 0.001*F
    for j in range(t0c):
        X0c = step(X0c, dt, F)
    return X0c

def init_ens(nx,nmem,t0c,t0f,dt,F,opt):
    X0c = init_ctl(nx, t0c, dt, F)
    tmp = np.zeros_like(X0c)
    maxiter = np.max(np.array(t0f))+1
    if(opt==0): # random
        logger.info("spin up max = {}".format(t0c))
        #print("spin up max = {}".format(t0c))
        np.random.seed(514)
        X0 = np.random.normal(0.0,1.0,size=(nx,nmem)) + X0c[:, None]
        for j in range(t0c):
            X0 = step(X0, dt, F)
    else: # lagged forecast
        logger.info("spin up max = {}".format(maxiter))
        #print("spin up max = {}".format(maxiter))
        X0 = np.zeros((nx,nmem))
        tmp = np.ones(nx)*F
        tmp[nx//2 - 1] += 0.001*F
        for j in range(maxiter):
            tmp = step(tmp, dt, F)
            if j in t0f[1:]:
                X0[:,t0f.index(j)-1] = tmp
    pf = (X0 - X0c[:, None]) @ (X0 - X0c[:, None]).T / (nmem-1)
    return X0c, X0, pf

def initialize(nx, nmem, t0c, t0f, dt, F, ft, pt, opt=0):
    if ft == "deterministic":
        u = init_ctl(nx, t0c, dt, F)
        xa = np.zeros((na, nx))
        xf = np.zeros_like(xa)
        xf[0, :] = u
        if pt == "kf":
            pf = np.eye(nx)*25.0
        else:
            pf = np.eye(nx)*0.2
    else:
        u = np.zeros((nx, nmem+1))
        u[:, 0], u[:, 1:], pf = init_ens(nx, nmem, t0c, t0f, dt, F, opt)
        xa = np.zeros((na, nx, nmem+1))
        xf = np.zeros_like(xa)
        xf[0, :, :] = u
    if pt == "mlef" or pt == "grad":
        sqrtpa = np.zeros((na, nx, nmem))
    else:
        sqrtpa = np.zeros((na, nx, nx))
    return u, xa, xf, pf, sqrtpa

def forecast(u, pa, dt, F, kmax, htype, a_window=1, tlm=True):
    from model.lorenz import step_t, step_adj
    if len(u.shape) > 1:
        uf = np.zeros((a_window, u.shape[0], u.shape[1]))
    else:
        uf = np.zeros((a_window, u.size))
    pf = np.zeros((a_window, pa.shape[0], pa.shape[0]))
    for l in range(a_window):
        for k in range(kmax):
            u = step(u, dt, F)
        uf[l] = u
        
        if htype["perturbation"] == "etkf" or htype["perturbation"] == "po" \
            or htype["perturbation"] == "letkf" or htype["perturbation"] == "srf":
            nmem = u.shape[1] - 1
            u[:, 0] = np.mean(u[:, 1:], axis=1)
            dxf = u[:, 1:] - u[:, 0].reshape(-1,1)
            p = dxf @ dxf.T / (nmem-1)
        elif htype["perturbation"] == "mlef" or htype["perturbation"] == "grad":
            spf = u[:, 1:] - u[:, 0].reshape(-1,1)
            p = spf @ spf.T
        elif htype["perturbation"] == "kf":
            M = np.eye(u.shape[0])
            MT = np.eye(u.shape[0])
            if tlm:
                E = np.eye(u.shape[0])
                uk = u
                for k in range(kmax):
                    Mk = step_t(uk[:,None], E, dt, F)
                    M = Mk @ M
                    MkT = step_adj(uk[:,None], E, dt, F)
                    MT = MT @ MkT
                    uk = step(uk, dt, F)
            else:
                for k in range(kmax):
                    M = kf.get_linear(u, dt, F, M, step)
                MT = M.T
            p = M @ pa @ MT
        elif htype["perturbation"] == "var" or htype["perturbation"] == "var4d":
            p = pa
    pf[l] = p
    if a_window > 1:
        return uf, pf
    else:
        return u, p


def analysis(u, pf, y, hist=False, dh=False, \
    infl=False, loc=False, tlm=True,\
    model="l96", icycle=0):
    logger.info("hist={}".format(hist))
    #print("hist={}".format(hist))
    if htype["perturbation"] == "mlef" or htype["perturbation"] == "grad":
        ua, pa, chi2= mlef.analysis(u, y[0], \
            save_hist=hist, save_dh=dh, \
            infl = infl, loc = loc, model=model, icycle=icycle)
        u[:, 0] = ua
        u[:, 1:] = ua[:, None] + pa
    elif htype["perturbation"] == "etkf" or htype["perturbation"] == "po" \
        or htype["perturbation"] == "letkf" or htype["perturbation"] == "srf":
        u_ = np.mean(u[:, 1:],axis=1)
        u[:, 0] = u_
        ua, ua_, pa, chi2 = enkf.analysis(u, y[0], \
            save_hist=hist, save_dh=dh, \
            infl = infl, loc = loc, tlm=tlm, model=model, icycle=icycle)
        u[:, 0] = ua_
        u[:, 1:] = ua
    elif htype["perturbation"] == "kf":
        u, pa = kf.analysis(u, pf, y[0], \
            infl = infl) #, loc = loc, tlm=tlm, \
            #save_dh=dh, model=model, icycle=icycle)
        chi2 = 0.0
    elif htype["perturbation"] == "var":
        u = var.analysis(u, pf, y[0], \
            save_hist=hist, model=model, icycle=icycle)
        pa = pf
        chi2 = 0.0
    elif htype["perturbation"] == "var4d":
        params = (dt, F, nt, a_window)
        u = var4d.analysis(u, pf, y, params,\
            save_hist=hist, model=model, icycle=icycle)
        pa = pf
        chi2 = 0.0
    return u, pa, chi2

def plot_initial(uc, u, ut, lag, model):
    fig, ax = plt.subplots()
    x = np.arange(ut.size) + 1
    ax.plot(x, ut, label="true")
    ax.plot(x, uc, label="control")
    for i in range(u.shape[1]):
        ax.plot(x, u[:,i], linestyle="--", label="mem{}".format(i+1))
    ax.set(xlabel="points", ylabel="X", title="initial lag={}".format(lag))
    ax.set_xticks(x[::5])
    ax.set_xticks(x, minor=True)
    ax.legend()
    fig.savefig("{}_initial_lag{}.png".format(model, lag))

if __name__ == "__main__":
    op = htype["operator"]
    pt = htype["perturbation"]
    ft = ftype[pt]
    model = "l96"
    #rmat, rinv = set_r(nx, sigma[op])
    xt, obs = get_true_and_obs(namax, nx)
    np.save("{}_ut.npy".format(model), xt[:na,:])
    u, xa, xf, pf, sqrtpa = initialize(nx, nmem, t0c, t0f, dt, F, ft, pt, opt=0)
    """
    x0c, x0, p0 = init_ens(nx,nmem,t0c,t0f,dt,F,opt=0)
    if htype["perturbation"] == "kf":
        u = np.zeros(nx)
        u = x0c
        pf = np.eye(nx)*25.0
        xa = np.zeros((na, nx))
        xf = np.zeros_like(xa)
        xf[0, :] = u
    elif htype["perturbation"] == "var" or htype["perturbation"] == "var4d":
        u = np.zeros(nx)
        u = x0c
        pf = np.eye(nx)*0.2
        xa = np.zeros((na, nx))
        xf = np.zeros_like(xa)
        xf[0, :] = u
    else:
        u = np.zeros((nx, nmem+1))
        u[:, 0] = x0c
        u[:, 1:] = x0
        pf = p0
        xa = np.zeros((na, nx, nmem+1))
        xf = np.zeros_like(xa)
        xf[0, :, :] = u
    if pt == "mlef" or pt == "grad":
        sqrtpa = np.zeros((na, nx, nmem))
    else:
        sqrtpa = np.zeros((na, nx, nx))
    """
    a_time = range(0, na, a_window)
    #print("a_time={}".format([time for time in a_time]))
    logger.info("a_time={}".format([time for time in a_time]))
    e = np.zeros(na)
    chi = np.zeros(na)
    #for i in range(na):
    for i in a_time:
        y = obs[i:i+a_window]
        logger.debug("observation shape {}".format(y.shape))
        if i in range(0,4):
            logger.info("cycle{} analysis".format(i))
            #print("cycle{} analysis".format(i))
            u, pa, chi2 = analysis(u, pf, y, \
                hist=True, dh=True, \
                infl=linf, loc=lloc, tlm=ltlm,\
                model=model, icycle=i)
        else:
            u, pa, chi2 = analysis(u, pf, y, \
                infl=linf, loc=lloc, tlm=ltlm, \
                model=model, icycle=i)
        if htype["perturbation"] == "kf" or htype["perturbation"] == "var"\
            or htype["perturbation"] == "var4d":
            xa[i, :] = u
        else:
            xa[i, :, :] = u
        sqrtpa[i, :, :] = pa
        chi[i] = chi2
        if i < na-1:
            if a_window > 1:
                uf, p = forecast(u, pa, dt, F, nt, htype, a_window=a_window)
                if htype["perturbation"] == "var4d":
                    if (i+1+a_window <= na):
                        xa[i+1:i+1+a_window, :] = uf[:,:]
                        xf[i+1:i+1+a_window, :] = uf[:,:]
                    else:
                        xa[i+1:na, :] = uf[:na-i-1,:]
                        xf[i+1:na, :] = uf[:na-i-1,:]
                else:
                    xa[i+1:i+1+a_window, :, :] = uf[:, :, :]
                    xf[i+1:i+1+a_window, :, :] = uf[:, :, :]
                if (i+1+a_window <= na):
                    sqrtpa[i+1:i+1+a_window, :, :] = p[:, :]
                else:
                    sqrtpa[i+1:na, :, :] = p[:na-i-1, :, :]
                u = uf[-1,:]
                pf = p[-1,:]
            else:
                u, pf = forecast(u, pa, dt, F, nt, htype,\
                    a_window=a_window, tlm=ltlm)
                if htype["perturbation"] == "kf" or htype["perturbation"] == "var" \
                    or htype["perturbation"] == "var4d":
                    xf[i+1, :] = u
                else:
                    xf[i+1, :, :] = u
        if a_window > 1:
            if htype["perturbation"] == "var4d":
                for k in range(i, min(i+a_window,na)):
                    e[k] = np.sqrt(np.mean((xa[k, :] - xt[k, :])**2))
            else:
                for k in range(i, min(i+a_window,na)):
                    e[k] = np.sqrt(np.mean((xa[k, :, 0] - xt[k, :])**2))
        else:
            if htype["perturbation"] == "kf" or htype["perturbation"] == "var" \
                or htype["perturbation"] == "var4d":
                e[i] = np.sqrt(np.mean((xa[i, :] - xt[i, :])**2))
            else:
                e[i] = np.sqrt(np.mean((xa[i, :, 0] - xt[i, :])**2))
    np.save("{}_ua_{}_{}.npy".format(model, op, pt), xa)
    np.save("{}_pa_{}_{}.npy".format(model, op, pt), sqrtpa)
    #if len(sys.argv) > 7:
    #    np.savetxt("{}_e_{}_{}_w{}.txt".format(model, op, pt, a_window), e)
    #    np.savetxt("{}_chi_{}_{}_w{}.txt".format(model, op, pt, a_window), chi)
    #else:
    np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)
    np.savetxt("{}_chi_{}_{}.txt".format(model, op, pt), chi)
