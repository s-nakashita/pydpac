import sys
import logging
from logging.config import fileConfig
import os
import numpy as np
from model.burgers import Bg
from analysis.obs import Obs
import matplotlib.pyplot as plt

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger(__name__)

global nx, nu, dt, dx

model = "z08"

nx = 81     # number of points
nu = 0.05   # diffusion
dt = 0.0125 # time step
logger.info("nx={} nu={} dt={:7.3e}".format(nx, nu, dt))

x = np.linspace(-2.0, 2.0, nx)
dx = x[1] - x[0]
np.savetxt("x.txt", x)

# forecast model forward operator
step = Bg(dx, dt, nu)

nmem =    4 # ensemble size
t0off =  12 # initial offset between adjacent members
t0true = 20 # t0 for true
t0c =    60 # t0 for control
#t0c = t0true
            # t0 for ensemble members
nt =     20 # number of step per forecast
na =     20 # number of analysis

sigma = {"linear": 8.0e-2, "quadratic": 8.0e-2, "cubic": 7.0e-4, "quartic": 7.0e-4, \
    "quadratic-nodiff": 8.0e-2, "cubic-nodiff": 7.0e-4, "quartic-nodiff": 7.0e-4}
#sigma = {"linear": 8.0e-2, "quadratic": 1.0e-3, "cubic": 1.0e-3, "quartic": 1.0e-2, \
#    "quadratic-nodiff": 1.0e-3, "cubic-nodiff": 1.0e-3, "quartic-nodiff": 1.0e-2}
htype = {"operator": "linear", "perturbation": "mlef"}
ftype = {"mlef":"ensemble","grad":"ensemble","etkf":"ensemble",\
    "po":"ensemble","srf":"ensemble","letkf":"ensemble",\
    "kf":"deterministic","var":"deterministic"}

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
obs_s = sigma[htype["operator"]]
if len(sys.argv) > 7:
    #t0off = int(sys.argv[6])
    obs_s = float(sys.argv[7])
t0m = [t0c + t0off//2 + t0off * i for i in range(-nmem//2, nmem//2)]
t0f = [t0c] + t0m
logger.info("nmem={} t0true={} t0f={}".format(nmem, t0true, t0f))
logger.info("nt={} na={}".format(nt, na))
logger.info("htype={} sigma={} ftype={}".format\
    (htype, obs_s, ftype[htype["perturbation"]]))
logger.info("inflation={} localization={} TLM={}".format(linf,lloc,ltlm))

global op, pt, ft
op = htype["operator"]
pt = htype["perturbation"]
ft = ftype[pt]

# observation operator
obs = Obs(op, obs_s)

# assimilation method
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

# generate truth and initialize
def gen_true(x, t0true, t0f, nt, na):
    nx = x.size
    nmem = len(t0f)
    t0c = t0f[0]
    u = np.zeros_like(x)
    u[0] = 1
    ut = np.zeros((na, nx))
    if ft == "deterministic":
        u0 = np.zeros(nx)
    else:
        u0 = np.zeros((nx, nmem))
    for k in range(t0true):
        u = step(u)
        if ft == "ensemble":
            if k+1 in t0f:
                u0[:, t0f.index(k+1)] = u
    ut[0, :] = u
    for i in range(na-1):
        for k in range(nt):
            u = step(u)
            #j = (i + 1) * nt + k
            j = t0true + i * nt + k
            if ft == "deterministic":
                if j == t0c:
                    u0 = u
            if ft == "ensemble":
                if j in t0f:
                    u0[:, t0f.index(j)] = u
        ut[i+1, :] = u
    if pt == "etkf" or pt == "po" \
        or pt == "letkf" or pt == "srf":
        dxf = u0[:, 1:] - u0[:, 0].reshape(-1,1) / np.sqrt(nmem-1)
        pf = dxf @ dxf.T
    elif pt == "mlef" or pt == "grad":
        dxf = u0[:, 1:] - u0[:, 0].reshape(-1,1)
        pf = dxf @ dxf.T
    else:
        pf = np.eye(nx) * 0.02
    return ut, u0, pf

def init_hist(u0, nx, nmem):
    if ft == "ensemble":
        ua = np.zeros((na, nx, nmem+1))
    else:
        ua = np.zeros((na, nx))
    uf = np.zeros_like(ua)
    uf[0] = u0
    if pt == "mlef" or pt == "grad":
        sqrtpa = np.zeros((na, nx, nmem))
    else:
        sqrtpa = np.zeros((na, nx, nx))
    return ua, uf, sqrtpa

def gen_obs(u):
    y = obs.h_operator(obs.add_noise(u))
    return y

def get_obs(f):
    y = np.load(f)
    return y


def forecast(u, pa, kmax, tlm=False):
    for k in range(kmax):
        u = step(u)
    if pt == "etkf" or pt == "po" \
        or pt == "letkf" or pt == "srf":
        nmem = u.shape[1] - 1 
        u[:, 0] = np.mean(u[:, 1:], axis=1)
        dxf = u[:, 1:] - u[:, 0].reshape(-1,1) / np.sqrt(nmem-1)
        pf = dxf @ dxf.T
    elif pt == "mlef" or pt == "grad":
        dxf = u[:, 1:] - u[:, 0].reshape(-1,1)
        pf = dxf @ dxf.T
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
        pf = M @ pa @ MT
    else:
        pf = pa
    return u, pf


def plot_initial(u, ut, lag, model):
    fig, ax = plt.subplots()
    x = np.arange(ut.size) + 1
    ax.plot(x, ut, label="true")
    for i in range(u.shape[1]):
        if i==0:
            ax.plot(x, u[:,i], label="control")
        else:
            ax.plot(x, u[:,i], linestyle="--", color="tab:green", label="mem{}".format(i))
    ax.set(xlabel="points", ylabel="u", title="initial lag={}".format(lag))
    ax.set_xticks(x[::10])
    ax.set_xticks(x[::5], minor=True)
    ax.legend()
    fig.savefig("{}_initial_lag{}.pdf".format(model, lag))

if __name__ == "__main__":
    ut, u, pf = gen_true(x, t0true, t0f, nt, na)
    oberr = int(obs_s*1e4)
    obsfile="obs_{}_{}.npy".format(op, oberr)
    if not os.path.isfile(obsfile):
        print("create obs")
        obs = gen_obs(ut)
        np.save(obsfile, obs)
    else:
        print("read obs")
        obs = get_obs(obsfile)
    #plot_initial(u, ut[0], t0off, model)
    ua, uf, sqrtpa = init_hist(u, nx, nmem)
        
    e = np.zeros(na+1)
    if ft == "ensemble":
        e[0] = np.sqrt(np.mean((uf[0, :, 0] - ut[0, :])**2))
    else:
        e[0] = np.sqrt(np.mean((uf[0, :] - ut[0, :])**2))
    chi = np.zeros(na)
    dpa = np.zeros(na)
    ndpa = np.zeros(na)
    for i in range(na):
        y = obs[i]
        if i in range(4):
            logger.info("cycle{} analysis".format(i))
            u, pa, chi2 = analysis(u, pf, y, \
                save_hist=True, save_dh=True,\
                infl=linf, loc=lloc, tlm=ltlm,\
                icycle=i)
        else:
            u, pa, chi2 = analysis(u, pf, y, \
                infl=linf, loc=lloc, tlm=ltlm, \
                icycle=i)
        ua[i] = u
        sqrtpa[i] = pa
        if pt == "mlef" or pt == "grad":
            dpa[i] = np.sum(np.diag(pa@pa.T))
            ndpa[i] = np.sum(pa@pa.T) - dpa[i]
        else:
            dpa[i] = np.sum(np.diag(pa))
            ndpa[i] = np.sum(pa) - dpa[i]
        chi[i] = chi2
        if i < na-1:
            u, pf = forecast(u, pa, nt)
            uf[i+1] = u
        if ft == "ensemble":
            e[i+1] = np.sqrt(np.mean((ua[i, :, 0] - ut[i, :])**2))
        else:
            e[i+1] = np.sqrt(np.mean((ua[i, :] - ut[i, :])**2))
    #np.save("{}_ut.npy".format(model), ut)
    #np.save("{}_uf_{}_{}.npy".format(model, op, pt), uf)
    #np.save("{}_ua_{}_{}.npy".format(model, op, pt), ua)
    ##np.save("{}_pa_{}_{}.npy".format(model, op, pt), sqrtpa)
    #np.savetxt("{}_dpa_{}_{}.txt".format(model, op, pt), dpa)
    #np.savetxt("{}_ndpa_{}_{}.txt".format(model, op, pt), ndpa)
    #if len(sys.argv) > 6:
    #    oberr = str(int(obs_s*1e4)).zfill(4)
    #    np.savetxt("{}_e_{}_{}_oberr{}.txt".format(model, op, pt, oberr), e)
    #    np.savetxt("{}_chi_{}_{}_oberr{}.txt".format(model, op, pt, oberr), chi)
    #else:
    np.savetxt("{}_e_{}_{}.txt".format(model, op, pt), e)
    np.savetxt("{}_chi_{}_{}.txt".format(model, op, pt), chi)
