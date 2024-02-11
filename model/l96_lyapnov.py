import numpy as np
from numpy import random
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FixedFormatter
plt.rcParams['font.size'] = 16
import scipy.optimize as opt 
from lorenz import L96
from lorenz2 import L05II
from lorenz3 import L05III
import sys
sys.path.append('../plot')
from nmc_tools import psd, wnum2wlen, wlen2wnum
from pathlib import Path

def fit_func(x, a, b):
    return a*x + b

model = "l05II"
if len(sys.argv)>1:
    imodel = int(sys.argv[1])
    if imodel==1:
        model="l96"
    elif imodel==2:
        model="l05II"
    elif imodel==3:
        model="l05III"
figdir = Path(f'lorenz/{model}')
if not figdir.exists():
    figdir.mkdir(parents=True)

F = 15.0
if len(sys.argv)>2:
    F = float(sys.argv[2])

if model=="l96":
    nx = 40
    dt = 0.05 / 6
    nt = 500 * 6
    isave = 1
    step = L96(nx, dt, F)
    figname = f'nx{nx}F{int(F)}'
    figtitle = f'N={nx} F={F:.1f}'
elif model=="l05II":
    nx = 240
    nk = 8
    if len(sys.argv)>3:
        nk = int(sys.argv[3])
    dt = 0.05 / 6
    nt = 500 * 6
    isave = 1
    step = L05II(nx, nk, dt, F)
    figname = f'nx{nx}nk{nk}F{int(F)}'
    figtitle = f'N={nx} K={nk} F={F:.1f}'
elif model=="l05III":
    nx = 960
    nk = 32
    if len(sys.argv)>3:
        nk = int(sys.argv[3])
    ni = 12
    b = 10.0
    if len(sys.argv)>4:
        b = float(sys.argv[4])
    c = 0.6
    if len(sys.argv)>5:
        c = float(sys.argv[5])
    dt = 0.05 / 6 / b
    nt = 500 * 6 * int(b)
    isave = int(b)
    step = L05III(nx,nk,ni,b,c,dt,F)
    figname = f'nx{nx}nk{nk}ni{ni}b{b:.1f}c{c:.1f}F{int(F)}'
    figtitle = f'N={nx} K={nk} I={ni}\nb={b:.1f} c={c:.1f} F={F:.1f}'
print(f"model={model}, F={F}")
nsave = nt//isave+1

ix = np.arange(nx) 
ix_rad = ix * 2.0 * np.pi / nx
x0 = random.normal(0, scale=1.0, size=nx)
for j in range(500): # spin up
    x0 = step(x0)

emean = np.zeros(nsave)
if model=='l05III':
    emean2 = np.zeros((2,nsave))
sp = []
xc = []
time = []
time.append(0.0)
x1 = x0.copy()
xc.append(x1)
wnum, sp1 = psd(x1,ix_rad)
sp.append(sp1)
for k in range(nt):
    x1 = step(x1)
    if k%isave==0:
        time.append(dt*(k+1))
        xc.append(x1)
        wnum, sp1 = psd(x1,ix_rad)
        sp.append(sp1)

day = np.array(time) / 0.05 / 4
fig = plt.figure(figsize=[10,6],constrained_layout=True)
cmap = plt.get_cmap('coolwarm')
gs = gridspec.GridSpec(1,3,figure=fig)
# hovmöller
ax0 = fig.add_subplot(gs[0,0])
mp = ax0.pcolormesh(ix,day,xc,shading='auto',\
        cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
fig.colorbar(mp,ax=ax0,pad=0.01,shrink=0.6)
ax0.set_xticks(ix[::(nx//8)])
ax0.set_yticks(day[::len(day)//8])
ax0.set_yticks(day[::len(day)//32],minor=True)
ax0.set_xlabel('site')
ax0.set_ylabel('day')
ax0.set_title(figtitle)
# variance power spectrum
ax1 = fig.add_subplot(gs[0,1:])
ax1.semilogy(wnum, np.array(sp).mean(axis=0))
ax1.set_title("time averaged power spectrum")
ax1.set(xlabel=r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)",title='variance power spectra')
#ax1.set_xscale('log')
#ax1.xaxis.set_major_locator(FixedLocator([240./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi]))
#ax1.xaxis.set_major_formatter(FixedFormatter([r'$\frac{240}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$']))
ax1.xaxis.set_major_locator(FixedLocator([480,240,120,60,30,1]))
ax1.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','1']))
#ax1.set_xlim(0.5/np.pi,wnum[-1])
secax = ax1.secondary_xaxis('top',functions=(wnum2wlen, wlen2wnum))
secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
fig.savefig(figdir/f'{figname}.png',dpi=300)
plt.close()

x2 = np.zeros_like(x0)
for j in range(50):
    print(f"trial {j+1}")
    x1 = xc[0]
    x2[:] = x0 + random.normal(0, scale=1e-4, size=nx)

    e = np.zeros(nsave)
    e[0] = np.sqrt(np.mean((x2 - x1)**2))
    if model=='l05III':
        e2 = np.zeros((2,nsave))
        x1l, x1s = step.decomp(x1)
        x2l, x2s = step.decomp(x2)
        e2[0,0] = np.sqrt(np.mean((x2l - x1l)**2))
        e2[1,0] = np.sqrt(np.mean((x2s - x1s)**2))
    i = 0
    for k in range(nt):
        x2 = step(x2)
        if k%isave==0:
            i=i+1
            x1 = xc[i]
            e[i] = np.sqrt(np.mean((x2 - x1)**2))
            if model=='l05III':
                x1l, x1s = step.decomp(x1)
                x2l, x2s = step.decomp(x2)
                e2[0,i] = np.sqrt(np.mean((x2l - x1l)**2))
                e2[1,i] = np.sqrt(np.mean((x2s - x1s)**2))
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

y = np.log(emean[:nsave//5])
t = np.array(time[:nsave//5])
popt, pcov = opt.curve_fit(fit_func, t, y)

ly = fit_func(t, popt[0], popt[1])

ax[1].scatter(t, y/np.log(10))
ax[1].plot(t, ly/np.log(10), color="tab:orange")
ax[1].set_xlabel("time")
ax[1].set_ylabel("RMSE(log10 scale)")
ax[1].set_title("Leading Lyapnov exponent = {:.2e}".format(popt[0]))

fig.savefig(figdir/f"lyapnov_{figname}.png",dpi=300)
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

    fig.savefig(figdir/f"lyapnov_scale_{figname}.png",dpi=300)