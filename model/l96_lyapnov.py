import numpy as np
from numpy import random
import matplotlib.pyplot as plt 
import scipy.optimize as opt 
from lorenz import L96

def fit_func(x, a, b):
    return a*x + b
nx = 120
dt = 0.05 / 6 / 3
F = 8.0
print(f"nx={nx} F={F}")

l96 = L96(nx, dt, F)
    
x0 = random.normal(0, scale=1.0, size=nx)
for j in range(500): # spin up
    x0 = l96(x0)

nt = 500 * 6 * (nx // 40)
emean = np.zeros(nt+1)
x1 = np.zeros_like(x0)
x2 = np.zeros_like(x0)
for j in range(50):
    x1[:] = x0
    x2[:] = x0 + random.normal(0, scale=1e-4, size=nx)

    time = []
    e = np.zeros(nt+1)
    time.append(0.0)
    e[0] = np.sqrt(np.mean((x2 - x1)**2))
    for k in range(nt):
        x2 = l96(x2)
        x1 = l96(x1)
        e[k+1] = np.sqrt(np.mean((x2 - x1)**2))
        time.append(dt*(k+1))
    emean += e
emean = emean / 50

fig, ax = plt.subplots(2)
ax[0].plot(time, emean)
ax[0].set_yscale("log")
ax[0].grid("both")
ax[0].set_xlabel("time")
ax[0].set_ylabel("RMSE")
ax[0].set_title("error growth")
print("final RMSE = {}".format(emean[-1]))

y = np.log(emean[:100])
t = np.array(time[:100])
popt, pcov = opt.curve_fit(fit_func, t, y)

ly = fit_func(t, popt[0], popt[1])

ax[1].scatter(t, y/np.log(10))
ax[1].plot(t, ly/np.log(10), color="tab:orange")
ax[1].set_xlabel("time")
ax[1].set_ylabel("RMSE(log10 scale)")
ax[1].set_title("Leading Lyapnov exponent = {}".format(popt[0]))

fig.tight_layout()
fig.savefig(f"l96_lyapnov_n{nx}F={int(F)}.png")
print("doubling time = {}".format(np.log(2)/popt[0]))