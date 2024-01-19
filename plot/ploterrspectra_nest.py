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
nx_t = ix_t.size
nx_gm = ix_gm.size
nx_lam = ix_lam.size
xt2x = interp1d(ix_t,xt)
xlim = 15.0
Lx_gm = 2.0 * np.pi
nghost = 50 # ghost region for periodicity in LAM
dwindow = (1.0 + np.cos(np.pi*np.arange(1,nghost+1)/nghost))*0.5
Lx_lam = 2.0 * np.pi * (nx_lam + 2*nghost) / nx_t
dx_gm = Lx_gm / nx_gm
dx_lam = Lx_lam / (nx_lam + 2*nghost)
for pt in perts:
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
    fig, axs = plt.subplots(nrows=2,figsize=[10,10],constrained_layout=True)
    xg2xt = interp1d(ix_gm,xagm,fill_value="extrapolate")
    xdgm = xg2xt(ix_t) - xt
    axs[0].plot(ix_t,xdgm.mean(axis=0),c='tab:blue',label='GM,err')
    #if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
    #    axs[0].plot(ix_gm,xsagm.mean(axis=0),c='tab:blue',ls='dashed',label='GM,sprd')
    i0 = np.argmin(np.abs(ix_t-ix_lam[0]))
    i1 = np.argmin(np.abs(ix_t-ix_lam[-1]))
    xdlam = xalam - xt[:,i0:i1+1]
    #xd = xdgm.copy()
    #xd[:,i0:i1+1] = xdlam
    xd = np.zeros((xdlam.shape[0],nx_lam+2*nghost))
    xd[:,nghost:nghost+nx_lam] = xdlam[:,:]
    xd[:,0:nghost] = xdlam[:,0].reshape(-1,1) * dwindow[None,::-1]
    xd[:,nghost+nx_lam:] = xdlam[:,-1].reshape(-1,1) * dwindow[None,:]
    xsa = np.zeros((xsalam.shape[0],nx_lam+2*nghost))
    xsa[:,nghost:nghost+nx_lam] = xsalam[:,:]
    xsa[:,0:nghost] = xsalam[:,0].reshape(-1,1) * dwindow[None,::-1]
    xsa[:,nghost+nx_lam:] = xsalam[:,-1].reshape(-1,1) * dwindow[None,:]
    ix_lam_ext = ix_t[i0-nghost:i0+nx_lam+nghost]
    axs[0].plot(ix_lam,xdlam.mean(axis=0),c='tab:orange',label='LAM,err')
    #if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
    #    axs[0].plot(ix_lam,xsalam.mean(axis=0),c='tab:orange',ls='dashed',label='LAM,sprd')
    axs[0].vlines([ix_lam[0],ix_lam[-1]],0,1,colors='k',ls='dashdot',transform=axs[0].get_xaxis_transform())
    axs[0].set_xlim(ix_t[0],ix_t[-1])
    axs[0].set_xlabel("grid index")
    #if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
    #    axs[0].set_ylabel("error or spread")
    #else:
    axs[0].set_ylabel("error")
    axs[0].set_title("state space")
    axs[0].legend()
    axs[0].grid()
    lines = []
    labels = []
    espgm = fft(xdgm,axis=1)
    freq = fftfreq(nx_gm,dx_gm)[:nx_gm//2] * 2.0 * np.pi
    #freq = np.arange(0,nx//2)
    print(espgm.shape)
    print(freq)
    psdgm = 2.0*np.mean(np.abs(espgm[:,:nx_gm//2])**2,axis=0)*dx_gm*dx_gm/Lx_gm
    axs[1].plot(freq,psdgm,c='tab:blue')
    lines.append(Line2D([0],[0],color='tab:blue',lw=2))
    labels.append('GM,err')
    #nx = xsagm.shape[1]
    #dx = 2.0 * np.pi / nx
    #freq = fftfreq(nx,dx)[:nx//2]
    #if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
    #    espgms = fft(xsagm,axis=1)
    #    psdgm = 2.0*np.mean(np.abs(espgms[:,:nx_gm//2])**2,axis=0)*dx_gm*dx_gm/Lx_gm
    #    axs[1].plot(freq,psdgm,c='tab:blue',ls='dashed',label='GM,sprd')
    nx = xd.shape[1]
    esp = fft(xd,axis=1)
    freq = fftfreq(nx,dx_lam)[:nx_lam//2] * 2.0 * np.pi
    #freq = np.arange(0,nx//2)
    print(freq)
    print(esp.shape)
    psd = 2.0*np.mean(np.abs(esp[:,:nx_lam//2])**2,axis=0)*dx_lam*dx_lam/Lx_lam
    axs[1].plot(freq,psd,c='tab:orange')
    lines.append(Line2D([0],[0],color='tab:orange',lw=2))
    labels.append('LAM,err')
    #ax2 = axs[1].twinx()
    #espdiff = esp - espgm
    #ax2.plot(freq,espdiff,c='red')
    #lines.append(Line2D([0],[0],color='red',lw=2))
    #labels.append('(GM+LAM)-GM')
    #ax2.hlines([0],0,1,colors='orange',transform=ax2.get_yaxis_transform(),zorder=0)
    #ax2.tick_params(axis='y',labelcolor='red')
    #if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
    #    esps = fft(xsa,axis=1)
    #    psd = 2.0*np.mean(np.abs(esps[:,:nx_lam//2])**2,axis=0)*dx_lam*dx_lam/Lx_lam
    #    axs[1].plot(freq,psd,c='tab:orange',ls='dashed',label='LAM,sprd')
    axs[1].set_xlim(freq[0],freq[-1])
    axs[1].set_yscale("log")
    axs[1].set_xlabel("wave number")
    axs[1].set_ylabel("power spectral density")
    axs[1].set_title("spectral space")
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
    fig.suptitle(f"{op} {pt}")
    fig.savefig("{}_errspectra_{}_{}.png".format(model,op,pt))
    plt.show()
