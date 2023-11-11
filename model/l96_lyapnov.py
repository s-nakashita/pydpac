import numpy as np
from numpy import random
import matplotlib.pyplot as plt 
import scipy.optimize as opt 
from lorenz import L96
from lorenz2 import L05II
from lorenz3 import L05III
plt.rcParams['font.size'] = 16

def fit_func(x, a, b):
    return a*x + b

model = "l05III"

F = 15.0
if model=="l96":
    nx = 40
    dt = 0.05 / 6
    nt = 500 * 6
    step = L96(nx, dt, F)
elif model=="l05II":
    nx = 240
    nk = 8
    dt = 0.05 / 6
    nt = 500 * 6
    step = L05II(nx, nk, dt, F)
elif model=="l05III":
    nx = 960
    nk = 32
    ni = 12
    b = 10.0
    c = 2.5
    dt = 0.05 / 6 / b
    nt = 500 * 6 * int(b)
    step = L05III(nx,nk,ni,b,c,dt,F)
print(f"model={model}, F={F}")
    
x0 = random.normal(0, scale=1.0, size=nx)
for j in range(500): # spin up
    x0 = step(x0)

emean = np.zeros(nt+1)
if model=='l05III':
    emean2 = np.zeros((2,nt+1))
x1 = np.zeros_like(x0)
x2 = np.zeros_like(x0)
for j in range(50):
    x1[:] = x0
    x2[:] = x0 + random.normal(0, scale=1e-4, size=nx)

    time = []
    e = np.zeros(nt+1)
    if model=='l05III':
        e2 = np.zeros((2,nt+1))
    time.append(0.0)
    e[0] = np.sqrt(np.mean((x2 - x1)**2))
    for k in range(nt):
        x2 = step(x2)
        x1 = step(x1)
        e[k+1] = np.sqrt(np.mean((x2 - x1)**2))
        time.append(dt*(k+1))
        if model=='l05III':
            x1s, x1l = step.decomp(x1)
            x2s, x2l = step.decomp(x2)
            e2[0,k+1] = np.sqrt(np.mean((x2l - x1l)**2))
            e2[1,k+1] = np.sqrt(np.mean((x2s - x1s)**2))
    emean += e
    if model == 'l05III':
        emean2 += e2
emean = emean / 50

fig, ax = plt.subplots(nrows=2,figsize=[12,12],constrained_layout=True)
ax[0].plot(time, emean)
ax[0].set_yscale("log")
ax[0].grid("both")
ax[0].set_xlabel("time")
ax[0].set_ylabel("RMSE")
ax[0].set_title("error growth")
print("final RMSE = {:.4f}".format(emean[-1]))

y = np.log(emean[:600])
t = np.array(time[:600])
popt, pcov = opt.curve_fit(fit_func, t, y)

ly = fit_func(t, popt[0], popt[1])

ax[1].scatter(t, y/np.log(10))
ax[1].plot(t, ly/np.log(10), color="tab:orange")
ax[1].set_xlabel("time")
ax[1].set_ylabel("RMSE(log10 scale)")
ax[1].set_title("Leading Lyapnov exponent = {:.2e}".format(popt[0]))

fig.savefig(f"{model}_lyapnov_F={int(F)}.png",dpi=300)
print("doubling time = {:.4f}".format(np.log(2)/popt[0]))

if model == 'l05III':
    emean2 += e2 / 50
    fig, ax = plt.subplots(nrows=2,figsize=[12,12],constrained_layout=True)
    ax[0].plot(time, emean2[0,])
    ax[0].set_yscale("log")
    ax[0].grid("both")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("RMSE")
    ax[0].set_title("large-scale error growth")

    ax[1].plot(time, emean2[1,])
    ax[1].set_yscale("log")
    ax[1].grid("both")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("RMSE")
    ax[1].set_title("small-scale error growth")

    fig.savefig(f"{model}_lyapnov_scale_F={int(F)}.png",dpi=300)