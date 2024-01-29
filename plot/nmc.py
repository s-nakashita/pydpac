import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.interpolate import interp1d
from nmc_tools import corrscale, psd
plt.rcParams['font.size'] = 16
def one_over(k):
    #Vectorized 1/x, treating x==0 manually
    k = np.array(k, float)
    near_zero = np.isclose(k, 0)
    x = np.zeros_like(k)
    x[near_zero] = np.inf
    x[~near_zero] = 1.0 / k[~near_zero]
    return x
inverse = one_over

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
dscl = False
if len(sys.argv)>4:
    dscl = (sys.argv[4]=='T')
print(dscl)
perts = ["mlef", "envar", "etkf", "po", "srf", "letkf", "kf", "var","var_nest",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
cmap = "coolwarm"
f = "truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[:na,]
print(xt.shape)
nx = xt.shape[1]
t = np.arange(na)
ix = np.loadtxt("ix.txt") * 360. / nx
ix_rad = np.deg2rad(ix)
xlim = 15.0
ns = na//10
ne = na
print(f"na={na} ns={ns} ne={ne}")
for pt in perts:
    # 6-hr forecast
    f = "{}_xf_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x6h = np.load(f)
    print(x6h.shape)
    nx = ix.size
    # 12-hr forecast
    f = "{}_xf12_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x12h = np.load(f)
    print(x12h.shape)
    nx = ix.size
    # 24-hr forecast
    f = "{}_xf24_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x24h = np.load(f)
    print(x24h.shape)
    nx = ix.size
    # 48-hr forecast
    f = "{}_xf48_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x48h = np.load(f)
    print(x48h.shape)
    nx = ix.size
    ## 12h - 6h
    #x12m6 = x12h[ns-1:ne-1] - x6h[ns:ne]
    x12m6 = x12h[ns:ne] - x6h[ns:ne]
    #x12m6 = x12m6 - x12m6.mean(axis=1)[:,None]
    print(x12m6.shape)
    B12m6 = np.dot(x12m6.T,x12m6)/float(ne-ns+1)*0.5
    wnum, sp12m6 = psd(x12m6,ix_rad,axis=1)
    ## 24h - 12h
    #x24m12 = x24h[ns-3:ne-3] - x12h[ns-1:ne-1]
    x24m12 = x24h[ns:ne] - x12h[ns:ne]
    #x24m12 = x24m12 - x24m12.mean(axis=1)[:,None]
    print(x24m12.shape)
    B24m12 = np.dot(x24m12.T,x24m12)/float(ne-ns+1)*0.5
    wnum, sp24m12 = psd(x24m12,ix_rad,axis=1)
    ## 48h - 24h
    #x48m24 = x48h[ns-7:ne-7] - x24h[ns-3:ne-3]
    x48m24 = x48h[ns:ne] - x24h[ns:ne]
    #x48m24 = x48m24 - x48m24.mean(axis=1)[:,None]
    print(x48m24.shape)
    B48m24 = np.dot(x48m24.T,x48m24)/float(ne-ns+1)*0.5
    wnum, sp48m24 = psd(x48m24,ix_rad,axis=1)
    ## plot
    fig, axs = plt.subplots(nrows=1,ncols=4,figsize=[12,6],constrained_layout=True,sharey=True)
    mp0 = axs[0].pcolormesh(ix, t[ns:ne], x6h[ns:ne,:], shading='auto',\
        cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[0].set_xticks(ix[::(nx//6)])
    axs[0].set_yticks(t[ns:ne:(na//8)])
    axs[0].set_xlabel("site")
    axs[0].set_ylabel("DA cycle")
    axs[0].set_title("6h")
    p0 = fig.colorbar(mp0,ax=axs[0],orientation="horizontal")
    vlim = max(np.max(x12m6),-np.min(x12m6))
    mp1 = axs[1].pcolormesh(ix, t[ns:ne], x12m6, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[1].set_xticks(ix[::(nx//6)])
    axs[1].set_yticks(t[ns:ne:(na//8)])
    axs[1].set_xlabel("site")
    axs[1].set_title("12h - 6h")
    p1 = fig.colorbar(mp1,ax=axs[1],orientation="horizontal")
    vlim = max(np.max(x24m12),-np.min(x24m12))
    mp2 = axs[2].pcolormesh(ix, t[ns:ne], x24m12, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[2].set_xticks(ix[::(nx//6)])
    axs[2].set_yticks(t[ns:ne:(na//8)])
    axs[2].set_xlabel("site")
    axs[2].set_title("24h - 12h")
    p2 = fig.colorbar(mp2,ax=axs[2],orientation="horizontal")
    vlim = max(np.max(x48m24),-np.min(x48m24))
    mp3 = axs[3].pcolormesh(ix, t[ns:ne], x48m24, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[3].set_xticks(ix[::(nx//6)])
    axs[3].set_yticks(t[ns:ne:(na//8)])
    axs[3].set_xlabel("site")
    axs[3].set_title("48h - 24h")
    p3 = fig.colorbar(mp3,ax=axs[3],orientation="horizontal")
    fig.suptitle("forecast : "+pt+" "+op)
    fig.savefig("{}_xf_{}_{}.png".format(model,op,pt))
    plt.show()
    plt.close()
    fig = plt.figure(figsize=[12,8],constrained_layout=True)
    gs = gridspec.GridSpec(2,1,figure=fig)
    gs0 = gs[0].subgridspec(1,3)
    ax00 = fig.add_subplot(gs0[:,0])
    ax01 = fig.add_subplot(gs0[:,1])
    ax02 = fig.add_subplot(gs0[:,2])
    vlim = max(np.max(B12m6),-np.min(B12m6))
    mp0 = ax00.pcolormesh(ix, ix, B12m6, shading='auto',\
        cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
    ax00.set_xticks(ix[::(nx//6)])
    ax00.set_yticks(ix[::(nx//6)])
    ax00.set_title("12h - 6h")
    ax00.set_aspect(1)
    p0 = fig.colorbar(mp0,ax=ax00,shrink=0.5,pad=0.01) #,orientation="horizontal")
    vlim = max(np.max(B24m12),-np.min(B24m12))
    mp1 = ax01.pcolormesh(ix, ix, B24m12, shading='auto',\
        cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
    ax01.set_xticks(ix[::(nx//6)])
    ax01.set_yticks(ix[::(nx//6)])
    ax01.set_title("24h - 12h")
    ax01.set_aspect(1)
    p1 = fig.colorbar(mp1,ax=ax01,shrink=0.5,pad=0.01) #,orientation="horizontal")
    vlim = max(np.max(B48m24),-np.min(B48m24))
    mp2 = ax02.pcolormesh(ix, ix, B48m24, shading='auto',\
        cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
    ax02.set_xticks(ix[::(nx//6)])
    ax02.set_yticks(ix[::(nx//6)])
    ax02.set_title("48h - 24h")
    ax02.set_aspect(1)
    p2 = fig.colorbar(mp2,ax=ax02,shrink=0.5,pad=0.01) #,orientation="horizontal")
    gs1 = gs[1].subgridspec(1,3)
    ax10 = fig.add_subplot(gs1[:,0])
    ax11 = fig.add_subplot(gs1[:,1])
    ax12 = fig.add_subplot(gs1[:,2])
    ### standard deviation
    data = [
        np.sqrt(np.diag(B12m6)),
        np.sqrt(np.diag(B24m12)),
        np.sqrt(np.diag(B48m24)),
        ]
    labels = [
        "12h - 6h",
        "24h - 12h",
        "48h - 24h"
    ]
    bp0=ax10.boxplot(data, vert=True, sym='+')
    ax10.yaxis.grid(True, linestyle='-', which='major', color='lightgray', alpha=0.5)
    ax10.set_xticks(np.arange(1,len(data)+1))
    ax10.set_xticklabels(labels)
    ax10.set(axisbelow=True,title=r"$\sigma$")
    for i in range(len(data)):
        med = bp0['medians'][i]
        ax10.plot(np.average(med.get_xdata()),np.average(data[i]),color='r',marker='*',markeredgecolor='k')
        s = str(round(np.average(data[i]),3))
        ax10.text(np.average(med.get_xdata()),.95,s,
        transform=ax10.get_xaxis_transform(),ha='center',c='r')
    ### correlation length scale
    L12m6  = corrscale(np.deg2rad(ix),B12m6,cyclic=True)
    L24m12 = corrscale(np.deg2rad(ix),B24m12,cyclic=True)
    L48m24 = corrscale(np.deg2rad(ix),B48m24,cyclic=True)
    data = [
        np.rad2deg(L12m6),
        np.rad2deg(L24m12),
        np.rad2deg(L48m24),
        ]
    bp1=ax11.boxplot(data, vert=True, sym='+')
    ax11.yaxis.grid(True, linestyle='-', which='major', color='lightgray', alpha=0.5)
    ax11.set_xticks(np.arange(1,len(data)+1))
    ax11.set_xticklabels(labels)
    ax11.set(axisbelow=True,title="Length-scale (degree)")
    for i in range(len(data)):
        med = bp1['medians'][i]
        ax11.plot(np.average(med.get_xdata()),np.average(data[i]),color='r',marker='*',markeredgecolor='k')
        s = str(round(np.average(data[i]),3))
        ax11.text(np.average(med.get_xdata()),.95,s,
        transform=ax11.get_xaxis_transform(),ha='center',c='r')
    ### variance spectra
    ax12.plot(wnum,sp12m6,label='12h - 6h')
    ax12.plot(wnum,sp24m12,label='24h - 12h')
    ax12.plot(wnum,sp48m24,label='48h - 24h')
    ax12.set_yscale("log")
    ax12.set(xlabel=r'wave number [radian$^{-1}$]',title='variance power spectra')
    ax12.set_xscale('log')
    ax12.xaxis.set_major_locator(FixedLocator([240./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi]))
    ax12.xaxis.set_major_formatter(FixedFormatter([r'$\frac{240}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$']))
    #ax12.set_xlim(0.5/np.pi,wnum[-1])
    secax = ax12.secondary_xaxis('top',functions=(one_over,inverse))
    secax.set_xlabel('wave length [radian]')
    secax.xaxis.set_major_locator(FixedLocator([np.pi,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
    secax.xaxis.set_major_formatter(FixedFormatter([r'$\pi$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
    ax12.legend()
    fig.suptitle("NMC : "+pt+" "+op)
    fig.savefig("{}_nmc_{}_{}.png".format(model,op,pt))
    plt.show()
    plt.close()
    np.save("{}_B12m6.npy".format(model),B12m6)
    np.save("{}_B24m12.npy".format(model),B24m12)
    np.save("{}_B48m24.npy".format(model),B48m24)
