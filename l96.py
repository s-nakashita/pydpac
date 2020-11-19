import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.lorenz import L96
from analysis.obs import Obs

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger(__name__)

global nx, F, dt, dx

model = "l96"

nx = 40     # number of points
F  = 8.0    # forcing
dt = 0.05 / 6  # time step (=1 hour)
logger.info("nx={} F={} dt={:7.3e}".format(nx, F, dt))

step = L96(dt, F)

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
logger.info("nt={} na={}".format(nt, na))
logger.info("htype={} sigma={} ftype={}".format\
    (htype, sigma[htype["operator"]], ftype[htype["perturbation"]]))
logger.info("inflation={} localization={} TLM={}".format(linf,lloc,ltlm))
logger.info("Assimilation window size = {}".format(a_window))

global op, pt, ft
op = htype["operator"]
pt = htype["perturbation"]
ft = ftype[pt]

obs = Obs(op, sigma[op])

if pt == "mlef" or pt == "grad":
    from analysis.mlef import Mlef
    analysis = Mlef(pt, obs, 1.1, model)
elif pt == "etkf" or pt == "po" or pt == "letkf" or pt == "srf":
    from analysis.enkf import EnKF
    analysis = EnKF(pt, obs, 1.1, 4.0, model)
elif pt == "kf":
    from analysis.kf import Kf
    analysis = Kf(pt, obs, 1.1, step)
elif pt == "var":
    from analysis.var import Var
    analysis = Var(pt, obs, model)
elif pt == "var4d":
    from analysis.var4d import Var4d
    analysis = Var4d(pt, obs, model, step, nt, a_window)
    
def get_true_and_obs(na, nx):
    f = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
        "data/data.csv")
    truth = pd.read_csv(f)
    xt = truth.values.reshape(na,nx)

    y = obs.h_operator(obs.add_noise(xt))

    return xt, y

def init_ctl(nx,t0c):
    X0c = np.ones(nx)*F
    X0c[nx//2 - 1] += 0.001*F
    for j in range(t0c):
        X0c = step(X0c)
    return X0c

def init_ens(nx,nmem,t0c,t0f,opt):
    X0c = init_ctl(nx, t0c)
    tmp = np.zeros_like(X0c)
    maxiter = np.max(np.array(t0f))+1
    if(opt==0): # random
        logger.info("spin up max = {}".format(t0c))
        np.random.seed(514)
        X0 = np.random.normal(0.0,1.0,size=(nx,nmem)) + X0c[:, None]
        for j in range(t0c):
            X0 = step(X0)
    else: # lagged forecast
        logger.info("spin up max = {}".format(maxiter))
        X0 = np.zeros((nx,nmem))
        tmp = np.ones(nx)*F
        tmp[nx//2 - 1] += 0.001*F
        for j in range(maxiter):
            tmp = step(tmp)
            if j in t0f[1:]:
                X0[:,t0f.index(j)-1] = tmp
    pf = (X0 - X0c[:, None]) @ (X0 - X0c[:, None]).T / (nmem-1)
    return X0c, X0, pf

def initialize(nx, nmem, t0c, t0f, opt=0):
    if ft == "deterministic":
        u = init_ctl(nx, t0c)
        xa = np.zeros((na, nx))
        xf = np.zeros_like(xa)
        xf[0, :] = u
        if pt == "kf":
            pf = np.eye(nx)*25.0
        else:
            pf = np.eye(nx)*0.2
    else:
        u = np.zeros((nx, nmem+1))
        u[:, 0], u[:, 1:], pf = init_ens(nx, nmem, t0c, t0f, opt)
        xa = np.zeros((na, nx, nmem+1))
        xf = np.zeros_like(xa)
        xf[0, :, :] = u
    if pt == "mlef" or pt == "grad":
        sqrtpa = np.zeros((na, nx, nmem))
    else:
        sqrtpa = np.zeros((na, nx, nx))
    return u, xa, xf, pf, sqrtpa

def forecast(u, pa, kmax, a_window=1, tlm=True):
    if ft == "ensemble":
        uf = np.zeros((a_window, u.shape[0], u.shape[1]))
    else:
        uf = np.zeros((a_window, u.size))
    pf = np.zeros((a_window, pa.shape[0], pa.shape[0]))
    for l in range(a_window):
        for k in range(kmax):
            u = step(u)
        uf[l] = u
        
        if pt == "etkf" or pt == "po" or pt == "letkf" or pt == "srf":
            nmem = u.shape[1] - 1
            u[:, 0] = np.mean(u[:, 1:], axis=1)
            dxf = u[:, 1:] - u[:, 0].reshape(-1,1)
            p = dxf @ dxf.T / (nmem-1)
        elif pt == "mlef" or pt == "grad":
            spf = u[:, 1:] - u[:, 0].reshape(-1,1)
            p = spf @ spf.T
        elif pt == "kf":
            M = np.eye(u.shape[0])
            MT = np.eye(u.shape[0])
            if tlm:
                E = np.eye(u.shape[0])
                uk = u
                for k in range(kmax):
                    Mk = step.step_t(uk[:,None], E)
                    M = Mk @ M
                    MkT = step.step_adj(uk[:,None], E)
                    MT = MT @ MkT
                    uk = step(uk)
            else:
                for k in range(kmax):
                    M = analysis.get_linear(u, M)
                MT = M.T
            p = M @ pa @ MT
        elif pt == "var" or pt == "var4d":
            p = pa
    pf[l] = p
    if a_window > 1:
        return uf, pf
    else:
        return u, p

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
    logger.info("==initialize==")
    xt, obs = get_true_and_obs(namax, nx)
    np.save("{}_ut.npy".format(model), xt[:na,:])
    u, xa, xf, pf, sqrtpa = initialize(nx, nmem, t0c, t0f, opt=0)
    
    a_time = range(0, na, a_window)
    logger.info("a_time={}".format([time for time in a_time]))
    e = np.zeros(na)
    chi = np.zeros(na)
    for i in a_time:
        y = obs[i:i+a_window]
        logger.debug("observation shape {}".format(y.shape))
        if i in range(0,4):
            logger.info("cycle{} analysis".format(i))
            if a_window > 1:
                u, pa, chi2 = analysis(u, pf, y, \
                    save_hist=True, save_dh=True, \
                    infl=linf, loc=lloc, tlm=ltlm,\
                    icycle=i)
            else:
                u, pa, chi2 = analysis(u, pf, y[0], \
                    save_hist=True, save_dh=True, \
                    infl=linf, loc=lloc, tlm=ltlm,\
                    icycle=i)
        else:
            if a_window > 1:
                u, pa, chi2 = analysis(u, pf, y, \
                    infl=linf, loc=lloc, tlm=ltlm,\
                    icycle=i)
            else:
                u, pa, chi2 = analysis(u, pf, y[0], \
                    infl=linf, loc=lloc, tlm=ltlm,\
                    icycle=i)

        if ft == "deterministic":
            xa[i, :] = u
        else:
            xa[i, :, :] = u
        sqrtpa[i, :, :] = pa
        chi[i] = chi2
        if i < na-1:
            if a_window > 1:
                uf, p = forecast(u, pa, nt, a_window=a_window)
                if ft == "deterministic":
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
                u, pf = forecast(u, pa, nt, \
                    a_window=a_window, tlm=ltlm)
                if ft == "deterministic":
                    xf[i+1, :] = u
                else:
                    xf[i+1, :, :] = u
        if a_window > 1:
            if ft == "deterministic":
                for k in range(i, min(i+a_window,na)):
                    e[k] = np.sqrt(np.mean((xa[k, :] - xt[k, :])**2))
            else:
                for k in range(i, min(i+a_window,na)):
                    e[k] = np.sqrt(np.mean((xa[k, :, 0] - xt[k, :])**2))
        else:
            if ft=="deterministic":
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
