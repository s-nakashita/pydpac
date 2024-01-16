import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "envar", "etkf", "po", "srf", "letkf", "kf", "var", "var_nest",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
if model == "z08":
    perts = ["mlef", "grad", "etkf-fh", "etkf-jh"]#, "po", "srf", "letkf"]
    linestyle = {"mlef":"solid", "grad":"dashed",
     "etkf-fh":"solid", "etkf-jh":"dashed"}
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf-fh":'tab:green',"etkf-jh":'tab:red'}
cmap = "coolwarm"
f = "truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[:na,]
print(xt.shape)
nx_t = xt.shape[1]
t = np.arange(na)
ix_t = np.loadtxt("ix_true.txt")
ix_gm = np.loadtxt("ix_gm.txt")
ix_lam = np.loadtxt("ix_lam.txt")
xt2x = interp1d(ix_t,xt)
xlim = 15.0
for pt in perts:
    fig, axs = plt.subplots(nrows=2,figsize=[10,10],constrained_layout=True)
    #GM
    f = "xagm_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xagm = np.load(f)
    print(xagm.shape)
    f = "xsagm_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xsagm = np.load(f)
    print(xsagm.shape)
    if np.isnan(xagm).any():
        print("divergence in {}".format(pt))
        continue
    xg2xt = interp1d(ix_gm,xagm,fill_value="extrapolate")
    xdgm = xg2xt(ix_t) - xt
    axs[0].plot(ix_t,xdgm.mean(axis=0),c='tab:blue',label='GM,err')
    #axs[0].plot(ix_gm,xsagm.mean(axis=0),c='tab:blue',ls='dashed',label='GM,sprd')
    #LAM
    f = "xalam_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xalam = np.load(f)
    print(xalam.shape)
    f = "xsalam_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xsalam = np.load(f)
    print(xsalam.shape)
    if np.isnan(xalam).any():
        print("divergence in {}".format(pt))
        continue
    i0 = np.argmin(np.abs(ix_t-ix_lam[0]))
    i1 = np.argmin(np.abs(ix_t-ix_lam[-1]))
    xdlam = xalam - xt[:,i0:i1+1]
    xd = xdgm.copy()
    xd[:,i0:i1+1] = xdlam
    axs[0].plot(ix_t,xd.mean(axis=0),c='tab:orange',label='GM+LAM,err')
    #axs[0].plot(ix_lam,xsalam.mean(axis=0),c='tab:orange',ls='dashed',label='LAM,sprd')
    axs[0].vlines([ix_lam[0],ix_lam[-1]],0,1,colors='k',ls='dashdot',transform=axs[0].get_xaxis_transform())
    axs[0].set_xlim(ix_t[0],ix_t[-1])
    axs[0].set_title("state space")
    axs[0].legend()
    axs[0].grid()
    lines = []
    labels = []
    nx = xdgm.shape[1]
    dx = 2.0 * np.pi / nx
    espgm = fft(xdgm,axis=1)
    #freq = fftfreq(nx,dx)[:nx//2]
    freq = np.arange(0,nx//2)
    print(espgm.shape)
    print(freq)
    espgm = 2.0*np.abs(espgm[:,:nx//2].mean(axis=0))/nx
    axs[1].plot(freq,espgm,c='tab:blue')
    lines.append(Line2D([0],[0],color='tab:blue',lw=2))
    labels.append('GM,err')
    #nx = xsagm.shape[1]
    #dx = 2.0 * np.pi / nx
    #espgms = fft(xsagm,axis=1)
    #freq = fftfreq(nx,dx)[:nx//2]
    #axs[1].plot(freq,2.0*np.abs(espgms[:,:nx//2].mean(axis=0))/nx,c='tab:blue',ls='dashed',label='GM,sprd')
    nx = xd.shape[1]
    dx = 2.0 * np.pi / nx
    esp = fft(xd,axis=1)
    #freq = fftfreq(nx,dx)[:nx//2]
    freq = np.arange(0,nx//2)
    print(freq)
    print(esp.shape)
    esp = 2.0*np.abs(esp[:,:nx//2].mean(axis=0))/nx
    axs[1].plot(freq,esp,c='tab:green')
    lines.append(Line2D([0],[0],color='tab:green',lw=2))
    labels.append('GM+LAM,err')
    ax2 = axs[1].twinx()
    espdiff = esp - espgm
    ax2.plot(freq,espdiff,c='red')
    lines.append(Line2D([0],[0],color='red',lw=2))
    labels.append('(GM+LAM)-GM')
    ax2.hlines([0],0,1,colors='orange',transform=ax2.get_yaxis_transform(),zorder=0)
    ax2.tick_params(axis='y',labelcolor='red')
    #xsa = xsagm
    #xsa[:,i0:i1+1] = xsalam
    #esplams = fft(xsalam,axis=1)
    #axs[1].plot(freq,2.0*np.abs(esplams[:,:nx//2].mean(axis=0))/nx,c='tab:orange',ls='dashed',label='LAM,sprd')
    #axs[1].set_xlim(freq[0],freq[-1])
    axs[1].set_yscale("log")
    axs[1].set_title("spectral space")
    axs[1].set_xlabel('wave number')
    """
    def num2len(k):
        #Vectorized 1/x, treating x==0 manually
        k = np.array(k, float)
        near_zero = np.isclose(k, 0)
        x = np.zeros_like(k)
        x[near_zero] = np.inf
        x[~near_zero] = nx * dx / k[~near_zero]
        return x
    len2num = num2len
    print(num2len(freq))
    secax = axs[1].secondary_xaxis('top',functions=(num2len,len2num))
    secax.set_xlabel('wave length')
    #secax.set_xscale('log')
    """
    axs[1].legend(lines,labels)
    axs[1].grid()
#    xd2 = ifft(esp,axis=1)
#    axs[0].plot(xs,xd2[0,],label='reconstructed')
    fig.savefig("{}_errspectra_{}_{}.png".format(model,op,pt))
#    plt.show()