import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.interpolate import interp1d
from nmc_tools import NMC_tools, wnum2wlen, wlen2wnum
#corrscale, psd, cpsd
sys.path.append(os.path.join(os.path.dirname(__file__),'../analysis'))
from trunc1d import Trunc1d
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
dscl = False
if len(sys.argv)>4:
    dscl = (sys.argv[4]=='T')
print(dscl)
perts = ["mlef", "envar", "envar_nest", "etkf", "po", "srf", "letkf", "kf", "var", "var_nest",\
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
ix_t = np.loadtxt("ix_true.txt") * 2.0 * np.pi / nx
ix_gm = np.loadtxt("ix_gm.txt") * 2.0 * np.pi / nx
ix_lam = np.loadtxt("ix_lam.txt") * 2.0 * np.pi / nx
ix_t_deg = np.rad2deg(ix_t)
ix_gm_deg = np.rad2deg(ix_gm)
ix_lam_deg = np.rad2deg(ix_lam)
tmp_gm2lam = interp1d(ix_gm, np.eye(ix_gm.size), axis=0)
H_gm2lam = tmp_gm2lam(ix_lam)
print(f"H_gm2lam={H_gm2lam.shape}")
nmc_gm = NMC_tools(ix_gm)
nmc_lam = NMC_tools(ix_lam,cyclic=False)
i0 = np.argmin(np.abs(ix_gm - ix_lam[0]))
if ix_gm[i0]<ix_lam[0]: i0+=1
i1 = np.argmin(np.abs(ix_gm - ix_lam[-1]))
if ix_gm[i1]>ix_lam[-1]: i1-=1
ix_gm_crop = ix_gm[i0:i1+1]
nmc_llam = NMC_tools(ix_gm_crop,cyclic=False)
#ftmax = 20.0 / np.pi
#f = trunc_operator(np.arange(ix_lam.size),ix=ix_lam,ftmax=ftmax,first=True,cyclic=False)
ntrunc = 12
trunc_operator = Trunc1d(ix_lam,ntrunc=ntrunc,cyclic=False,nghost=0)
ix_trunc = trunc_operator.ix_trunc
nmc_trunc = NMC_tools(ix_trunc,cyclic=False)

xt2x = interp1d(ix_t, xt)
xlim = 15.0
ns = na//10
ne = na
print(f"na={na} ns={ns} ne={ne}")
for pt in perts:
    #GM
    # 6-hr forecast
    if dscl:
        f = "{}_xfgmonly_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xfgm_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x6hgm = np.load(f)
    print(x6hgm.shape)
    nx = ix_gm.size
    # 12-hr forecast
    if dscl:
        f = "{}_xf12gmonly_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xf12gm_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x12hgm = np.load(f)
    print(x12hgm.shape)
    nx = ix_gm.size
    # 24-hr forecast
    if dscl:
        f = "{}_xf24gmonly_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xf24gm_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x24hgm = np.load(f)
    print(x24hgm.shape)
    nx = ix_gm.size
    # 48-hr forecast
    if dscl:
        f = "{}_xf48gmonly_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xf48gm_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x48hgm = np.load(f)
    print(x48hgm.shape)
    nx = ix_gm.size
    ## 12h - 6h
    x12m6_gm = x12hgm[ns:ne] - x6hgm[ns:ne]
    #x12m6_gm = x12m6_gm - x12m6_gm.mean(axis=1)[:,None]
    print(x12m6_gm.shape)
    B12m6_gm = np.dot(x12m6_gm.T,x12m6_gm)/float(ne-ns+1)*0.5
    wnum, sp12m6_gm = nmc_gm.psd(x12m6_gm,axis=1)
    ## 24h - 12h
    x24m12_gm = x24hgm[ns:ne] - x12hgm[ns:ne]
    #x24m12_gm = x24m12_gm - x24m12_gm.mean(axis=1)[:,None]
    print(x24m12_gm.shape)
    B24m12_gm = np.dot(x24m12_gm.T,x24m12_gm)/float(ne-ns+1)*0.5
    wnum, sp24m12_gm = nmc_gm.psd(x24m12_gm,axis=1)
    ## 48h - 24h
    x48m24_gm = x48hgm[ns:ne] - x24hgm[ns:ne]
    #x48m24_gm = x48m24_gm - x48m24_gm.mean(axis=1)[:,None]
    print(x48m24_gm.shape)
    B48m24_gm = np.dot(x48m24_gm.T,x48m24_gm)/float(ne-ns+1)*0.5
    wnum, sp48m24_gm = nmc_gm.psd(x48m24_gm,axis=1)
    ## plot
    fig, axs = plt.subplots(nrows=1,ncols=4,figsize=[12,6],constrained_layout=True,sharey=True)
    mp0 = axs[0].pcolormesh(ix_gm_deg, t[ns:ne], x6hgm[ns:ne], shading='auto',\
        cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[0].set_xticks(ix_gm_deg[::(nx//6)])
    axs[0].set_yticks(t[ns:ne:(na//8)])
    axs[0].set_xlabel("site")
    axs[0].set_ylabel("DA cycle")
    axs[0].set_title("6h")
    p0 = fig.colorbar(mp0,ax=axs[0],orientation="horizontal")
    vlim = max(np.max(x12m6_gm),-np.min(x12m6_gm))
    mp1 = axs[1].pcolormesh(ix_gm_deg, t[ns:ne], x12m6_gm, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[1].set_xticks(ix_gm_deg[::(nx//6)])
    axs[1].set_yticks(t[ns:ne:(na//8)])
    axs[1].set_xlabel("site")
    axs[1].set_title("12h - 6h")
    p1 = fig.colorbar(mp1,ax=axs[1],orientation="horizontal")
    vlim = max(np.max(x24m12_gm),-np.min(x24m12_gm))
    mp2 = axs[2].pcolormesh(ix_gm_deg, t[ns:ne], x24m12_gm, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[2].set_xticks(ix_gm_deg[::(nx//6)])
    axs[2].set_yticks(t[ns:ne:(na//8)])
    axs[2].set_xlabel("site")
    axs[2].set_title("24h - 12h")
    p2 = fig.colorbar(mp2,ax=axs[2],orientation="horizontal")
    vlim = max(np.max(x48m24_gm),-np.min(x48m24_gm))
    mp3 = axs[3].pcolormesh(ix_gm_deg, t[ns:ne], x48m24_gm, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[3].set_xticks(ix_gm_deg[::(nx//6)])
    axs[3].set_yticks(t[ns:ne:(na//8)])
    axs[3].set_xlabel("site")
    axs[3].set_title("48h - 24h")
    p3 = fig.colorbar(mp3,ax=axs[3],orientation="horizontal")
    fig.suptitle("forecast in GM : "+pt+" "+op)
    fig.savefig("{}_xfgm_{}_{}.png".format(model,op,pt))
    #plt.show(block=False)
    plt.close(fig=fig)
    #
    figsp, axsp = plt.subplots(figsize=[6,8],constrained_layout=True)
    fig = plt.figure(figsize=[12,8],constrained_layout=True)
    gs = gridspec.GridSpec(2,1,figure=fig)
    gs0 = gs[0].subgridspec(1,3)
    ax00 = fig.add_subplot(gs0[:,0])
    ax01 = fig.add_subplot(gs0[:,1])
    ax02 = fig.add_subplot(gs0[:,2])
    vlim = max(np.max(B12m6_gm),-np.min(B12m6_gm))
    mp0 = ax00.pcolormesh(ix_gm_deg, ix_gm_deg, B12m6_gm, shading='auto',\
        cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
    ax00.set_xticks(ix_gm_deg[::(nx//6)])
    ax00.set_yticks(ix_gm_deg[::(nx//6)])
    ax00.set_title("12h - 6h")
    ax00.set_aspect("equal")
    p0 = fig.colorbar(mp0,ax=ax00,shrink=0.5,pad=0.01) #,orientation="horizontal")
    vlim = max(np.max(B24m12_gm),-np.min(B24m12_gm))
    mp1 = ax01.pcolormesh(ix_gm_deg, ix_gm_deg, B24m12_gm, shading='auto',\
        cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
    ax01.set_xticks(ix_gm_deg[::(nx//6)])
    ax01.set_yticks(ix_gm_deg[::(nx//6)])
    ax01.set_title("24h - 12h")
    ax01.set_aspect("equal")
    p1 = fig.colorbar(mp1,ax=ax01,shrink=0.5,pad=0.01) #,orientation="horizontal")
    vlim = max(np.max(B48m24_gm),-np.min(B48m24_gm))
    mp2 = ax02.pcolormesh(ix_gm_deg, ix_gm_deg, B48m24_gm, shading='auto',\
        cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
    ax02.set_xticks(ix_gm_deg[::(nx//6)])
    ax02.set_yticks(ix_gm_deg[::(nx//6)])
    ax02.set_title("48h - 24h")
    ax02.set_aspect("equal")
    p2 = fig.colorbar(mp2,ax=ax02,shrink=0.5,pad=0.01)
    gs1 = gs[1].subgridspec(1,3)
    ax10 = fig.add_subplot(gs1[:,0])
    ax11 = fig.add_subplot(gs1[:,1])
    ax12 = fig.add_subplot(gs1[:,2])
    ### standard deviation
    data = [
        np.sqrt(np.diag(B12m6_gm)),
        np.sqrt(np.diag(B24m12_gm)),
        np.sqrt(np.diag(B48m24_gm)),
        ]
    labels = [
        "12h - 6h",
        "24h - 12h",
        "48h - 24h"
    ]
    bp0=ax12.boxplot(data, vert=True, sym='+')
    ax12.yaxis.grid(True, linestyle='-', which='major', color='lightgray', alpha=0.5)
    ax12.set_xticks(np.arange(1,len(data)+1))
    ax12.set_xticklabels(labels)
    ax12.set(axisbelow=True,title=r"$\sigma$")
    for i in range(len(data)):
        med = bp0['medians'][i]
        ax12.plot(np.average(med.get_xdata()),np.average(data[i]),color='r',marker='*',markeredgecolor='k')
        s = str(round(np.average(data[i]),3))
        ax12.text(np.average(med.get_xdata()),.95,s,
        transform=ax12.get_xaxis_transform(),ha='center',c='r')
    ### correlation length scale
    L12m6_gm  = nmc_gm.corrscale(B12m6_gm)
    L24m12_gm = nmc_gm.corrscale(B24m12_gm)
    L48m24_gm = nmc_gm.corrscale(B48m24_gm)
    data = [
        np.rad2deg(L12m6_gm),
        np.rad2deg(L24m12_gm),
        np.rad2deg(L48m24_gm),
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
    ax10.plot(wnum,sp12m6_gm,label='12h - 6h')
    ax10.plot(wnum,sp24m12_gm,label='24h - 12h')
    ax10.plot(wnum,sp48m24_gm,label='48h - 24h')
    ax10.set_yscale("log")
    ax10.set(xlabel=r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)",title='variance power spectra')
    ax10.set_xscale('log')
    #ax10.xaxis.set_major_locator(FixedLocator([240./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi]))
    #ax10.xaxis.set_major_formatter(FixedFormatter([r'$\frac{240}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$']))
    ax10.xaxis.set_major_locator(FixedLocator([480,240,120,60,8,2,1]))
    ax10.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','8','2','1']))
    #ax10.set_xlim(0.5/np.pi,wnum[-1])
    secax = ax10.secondary_xaxis('top',functions=(wnum2wlen, wlen2wnum))
    secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
    secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi,np.pi/4,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
    secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\pi$',r'$\frac{\pi}{4}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
    ax10.legend()
    fig.suptitle("NMC in GM : "+pt+" "+op)
    if dscl:
        fig.savefig("{}_nmcgmonly_{}_{}.png".format(model,op,pt))
    else:
        fig.savefig("{}_nmcgm_{}_{}.png".format(model,op,pt))
    plt.show(block=False)
    plt.close(fig=fig)
    axsp.plot(wnum, sp48m24_gm, c='gray', label='GM')
    np.save("{}_B12m6_gm.npy".format(model),B12m6_gm)
    np.save("{}_B24m12_gm.npy".format(model),B24m12_gm)
    np.save("{}_B48m24_gm.npy".format(model),B48m24_gm)
    #exit()
    #LAM
    # 6-hr forecast
    if dscl:
        f = "{}_xfdscl_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xflam_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x6hlam = np.load(f)
    print(x6hlam.shape)
    nx = ix_lam.size
    # 12-hr forecast
    if dscl:
        f = "{}_xf12dscl_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xf12lam_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x12hlam = np.load(f)
    print(x12hlam.shape)
    nx = ix_lam.size
    # 24-hr forecast
    if dscl:
        f = "{}_xf24dscl_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xf24lam_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x24hlam = np.load(f)
    print(x24hlam.shape)
    # 48-hr forecast
    if dscl:
        f = "{}_xf48dscl_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xf48lam_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x48hlam = np.load(f)
    print(x48hlam.shape)
    nx = ix_lam.size
    ## 12h - 6h
    x12m6_lam = x12hlam[ns:ne] - x6hlam[ns:ne]
    #x12m6_lam = x12m6_lam - x12m6_lam.mean(axis=1)[:,None]
    print(x12m6_lam.shape)
    B12m6_lam = np.dot(x12m6_lam.T,x12m6_lam)/float(ne-ns+1)*0.5
    wnum, sp12m6_lam, _ = nmc_lam.psd(x12m6_lam, axis=1)
    ## 24h - 12h
    x24m12_lam = x24hlam[ns:ne] - x12hlam[ns:ne]
    #x24m12_lam = x24m12_lam - x24m12_lam.mean(axis=1)[:,None]
    print(x24m12_lam.shape)
    B24m12_lam = np.dot(x24m12_lam.T,x24m12_lam)/float(ne-ns+1)*0.5
    wnum, sp24m12_lam, _ = nmc_lam.psd(x24m12_lam, axis=1)
    ## 48h - 24h
    x48m24_lam = x48hlam[ns:ne] - x24hlam[ns:ne]
    #x48m24_lam = x48m24_lam - x48m24_lam.mean(axis=1)[:,None]
    print(x48m24_lam.shape)
    B48m24_lam = np.dot(x48m24_lam.T,x48m24_lam)/float(ne-ns+1)*0.5
    wnum, sp48m24_lam, _ = nmc_lam.psd(x48m24_lam, axis=1)
    ## plot
    fig, axs = plt.subplots(nrows=1,ncols=4,figsize=[12,6],constrained_layout=True,sharey=True)
    mp0 = axs[0].pcolormesh(ix_lam_deg, t[ns:ne], x6hlam[ns:ne], shading='auto',\
        cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[0].set_xticks(ix_lam_deg[::(nx//6)])
    axs[0].set_yticks(t[ns:ne:(na//8)])
    axs[0].set_xlabel("site")
    axs[0].set_ylabel("DA cycle")
    axs[0].set_title("6h")
    p0 = fig.colorbar(mp0,ax=axs[0],orientation="horizontal")
    vlim = max(np.max(x12m6_lam),-np.min(x12m6_lam))
    mp1 = axs[1].pcolormesh(ix_lam_deg, t[ns:ne], x12m6_lam, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[1].set_xticks(ix_lam_deg[::(nx//6)])
    axs[1].set_yticks(t[ns:ne:(na//8)])
    axs[1].set_xlabel("site")
    axs[1].set_title("12h - 6h")
    p1 = fig.colorbar(mp1,ax=axs[1],orientation="horizontal")
    vlim = max(np.max(x24m12_lam),-np.min(x24m12_lam))
    mp2 = axs[2].pcolormesh(ix_lam_deg, t[ns:ne], x24m12_lam, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[2].set_xticks(ix_lam_deg[::(nx//6)])
    axs[2].set_yticks(t[ns:ne:(na//8)])
    axs[2].set_xlabel("site")
    axs[2].set_title("24h - 12h")
    p2 = fig.colorbar(mp2,ax=axs[2],orientation="horizontal")
    vlim = max(np.max(x48m24_lam),-np.min(x48m24_lam))
    mp3 = axs[3].pcolormesh(ix_lam_deg, t[ns:ne], x48m24_lam, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[3].set_xticks(ix_lam_deg[::(nx//6)])
    axs[3].set_yticks(t[ns:ne:(na//8)])
    axs[3].set_xlabel("site")
    axs[3].set_title("48h - 24h")
    p3 = fig.colorbar(mp3,ax=axs[3],orientation="horizontal")
    fig.suptitle("forecast in LAM : "+pt+" "+op)
    fig.savefig("{}_xflam_{}_{}.png".format(model,op,pt))
    #plt.show(block=False)
    plt.close(fig=fig)
    #
    fig = plt.figure(figsize=[12,8],constrained_layout=True)
    gs = gridspec.GridSpec(2,1,figure=fig)
    gs0 = gs[0].subgridspec(1,3)
    ax00 = fig.add_subplot(gs0[:,0])
    ax01 = fig.add_subplot(gs0[:,1])
    ax02 = fig.add_subplot(gs0[:,2])
    vlim = max(np.max(B12m6_lam),-np.min(B12m6_lam))
    mp0 = ax00.pcolormesh(ix_lam_deg, ix_lam_deg, B12m6_lam, shading='auto',\
        cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
    ax00.set_xticks(ix_lam_deg[::(nx//6)])
    ax00.set_yticks(ix_lam_deg[::(nx//6)])
    ax00.set_title("12h - 6h")
    ax00.set_aspect("equal")
    p0 = fig.colorbar(mp0,ax=ax00,shrink=0.5,pad=0.01) #,orientation="horizontal")
    vlim = max(np.max(B24m12_lam),-np.min(B24m12_lam))
    mp1 = ax01.pcolormesh(ix_lam_deg, ix_lam_deg, B24m12_lam, shading='auto',\
        cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
    ax01.set_xticks(ix_lam_deg[::(nx//6)])
    ax01.set_yticks(ix_lam_deg[::(nx//6)])
    ax01.set_title("24h - 12h")
    ax01.set_aspect("equal")
    p1 = fig.colorbar(mp1,ax=ax01,shrink=0.5,pad=0.01) #,orientation="horizontal")
    vlim = max(np.max(B48m24_lam),-np.min(B48m24_lam))
    mp2 = ax02.pcolormesh(ix_lam_deg, ix_lam_deg, B48m24_lam, shading='auto',\
        cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
    ax02.set_xticks(ix_lam_deg[::(nx//6)])
    ax02.set_yticks(ix_lam_deg[::(nx//6)])
    ax02.set_title("48h - 24h")
    ax02.set_aspect("equal")
    p2 = fig.colorbar(mp2,ax=ax02,shrink=0.5,pad=0.01)
    gs1 = gs[1].subgridspec(1,3)
    ax10 = fig.add_subplot(gs1[:,0])
    ax11 = fig.add_subplot(gs1[:,1])
    ax12 = fig.add_subplot(gs1[:,2])
    ### standard deviation
    data = [
        np.sqrt(np.diag(B12m6_lam)),
        np.sqrt(np.diag(B24m12_lam)),
        np.sqrt(np.diag(B48m24_lam)),
        ]
    labels = [
        "12h - 6h",
        "24h - 12h",
        "48h - 24h"
    ]
    bp0=ax12.boxplot(data, vert=True, sym='+')
    ax12.yaxis.grid(True, linestyle='-', which='major', color='lightgray', alpha=0.5)
    ax12.set_xticks(np.arange(1,len(data)+1))
    ax12.set_xticklabels(labels)
    ax12.set(axisbelow=True,title=r"$\sigma$")
    for i in range(len(data)):
        med = bp0['medians'][i]
        ax12.plot(np.average(med.get_xdata()),np.average(data[i]),color='r',marker='*',markeredgecolor='k')
        s = str(round(np.average(data[i]),3))
        ax12.text(np.average(med.get_xdata()),.95,s,
        transform=ax12.get_xaxis_transform(),ha='center',c='r')
    ### correlation length scale
    L12m6_lam  = nmc_lam.corrscale(B12m6_lam)
    L24m12_lam = nmc_lam.corrscale(B24m12_lam)
    L48m24_lam = nmc_lam.corrscale(B48m24_lam)
    data = [
        np.rad2deg(L12m6_lam),
        np.rad2deg(L24m12_lam),
        np.rad2deg(L48m24_lam),
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
    ax10.plot(wnum,sp12m6_lam,label='12h - 6h')
    ax10.plot(wnum,sp24m12_lam,label='24h - 12h')
    ax10.plot(wnum,sp48m24_lam,label='48h - 24h')
    ax10.set_yscale("log")
    ax10.set(xlabel=r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)",title='variance power spectra')
    ax10.set_xscale('log')
    #ax10.xaxis.set_major_locator(FixedLocator([240./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi]))
    #ax10.xaxis.set_major_formatter(FixedFormatter([r'$\frac{240}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$']))
    ax10.xaxis.set_major_locator(FixedLocator([480,240,120,60,8,2,1]))
    ax10.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','8','2','1']))
    #ax10.set_xlim(0.5/np.pi,wnum[-1])
    secax = ax10.secondary_xaxis('top',functions=(wnum2wlen, wlen2wnum))
    secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
    secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi,np.pi/4,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
    secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\pi$',r'$\frac{\pi}{4}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
    ax10.legend()
    fig.suptitle("NMC in LAM : "+pt+" "+op)
    if dscl:
        fig.savefig("{}_nmcdscl_{}_{}.png".format(model,op,pt))
    else:
        fig.savefig("{}_nmclam_{}_{}.png".format(model,op,pt))
    plt.show(block=False)
    plt.close(fig=fig)
    np.save("{}_B12m6_lam.npy".format(model),B12m6_lam)
    np.save("{}_B24m12_lam.npy".format(model),B24m12_lam)
    np.save("{}_B48m24_lam.npy".format(model),B48m24_lam)
    axsp.plot(wnum,sp48m24_lam,c='blue',ls='dashed',label='LAM')
    #exit()
    #GM-LAM
    ixlist = []
    nmclist = []
    vmatlist=[]
    wnumlist = []
    splist=[]
    gm2lamlist=[]
    sp_gm2lamlist=[]
    ## GM interpolated into nominal LAM grid (H_1:GM=>LAM, H_2=I)
    ixlist.append(ix_lam)
    nmclist.append(nmc_lam)
    ### 12h - 6h
    gm2lam = interp1d(ix_gm,x12hgm[ns:ne],axis=1)
    x12hgm2lam = gm2lam(ix_lam)
    ek12h = x12hgm2lam - x12hlam[ns:ne] #\sim H_1(x^{a}_{GM}) - H_2(x^{t}_{LAM})
    V12m6 = np.dot(ek12h.T,ek12h)/float(ne-ns+1)*0.5
    B12m6_gm2lam = np.dot(ek12h.T,x12m6_lam)/float(ne-ns+1)*0.5
    vmatlist.append(V12m6)
    gm2lamlist.append(B12m6_gm2lam)
    wnum, sp12m6_v, _ = nmc_lam.psd(ek12h,axis=1)
    wnumlist.append(wnum)
    splist.append(sp12m6_v)
    wnum_c, csp12m6 = nmc_lam.cpsd(ek12h,x12m6_lam,axis=1)
    sp_gm2lamlist.append(csp12m6)
    ### 24h - 12h
    gm2lam = interp1d(ix_gm,x24hgm[ns:ne],axis=1)
    x24hgm2lam = gm2lam(ix_lam)
    ek24h = x24hgm2lam - x24hlam[ns:ne] #\sim H_1(x^{a}_{GM}) - H_2(x^{t}_{LAM})
    V24m12 = np.dot(ek24h.T,ek24h)/float(ne-ns+1)*0.5
    B24m12_gm2lam = np.dot(ek24h.T,x24m12_lam)/float(ne-ns+1)*0.5
    vmatlist.append(V24m12)
    gm2lamlist.append(B24m12_gm2lam)
    wnum, sp24m12_v, _ = nmc_lam.psd(ek24h,axis=1)
    splist.append(sp24m12_v)
    wnum_c, csp24m12 = nmc_lam.cpsd(ek24h,x24m12_lam,axis=1)
    sp_gm2lamlist.append(csp24m12)
    ### 48h - 24h
    gm2lam = interp1d(ix_gm,x48hgm[ns:ne],axis=1)
    x48hgm2lam = gm2lam(ix_lam)
    ek48h = x48hgm2lam - x48hlam[ns:ne] #\sim H_1(x^{a}_{GM}) - H_2(x^{t}_{LAM})
    V48m24 = np.dot(ek48h.T,ek48h)/float(ne-ns+1)*0.5
    B48m24_gm2lam = np.dot(ek48h.T,x48m24_lam)/float(ne-ns+1)*0.5
    vmatlist.append(V48m24)
    gm2lamlist.append(B48m24_gm2lam)
    wnum, sp48m24_v, _ = nmc_lam.psd(ek48h,axis=1)
    splist.append(sp48m24_v)
    wnum_c, csp48m24 = nmc_lam.cpsd(ek48h,x48m24_lam,axis=1)
    sp_gm2lamlist.append(csp48m24)
    ## extract GM included in the LAM domain (H_1:crop, H_2:LAM=>cropped GM)
    ixlist.append(ix_gm_crop)
    nmclist.append(nmc_llam)
    ### 12h - 6h
    lam2gm = interp1d(ix_lam,x12hlam[ns:ne],axis=1)
    x12hlam2gm = lam2gm(ix_gm[i0:i1+1])
    ek12h = x12hgm[ns:ne,i0:i1+1] - x12hlam2gm #\sim H_1(x^{a}_{GM}) - H_2(x^{t}_{LAM})
    V12m6 = np.dot(ek12h.T,ek12h)/float(ne-ns+1)*0.5
    B12m6_gm2lam = np.dot(ek12h.T,x12m6_lam)/float(ne-ns+1)*0.5
    vmatlist.append(V12m6)
    gm2lamlist.append(B12m6_gm2lam)
    wnum, sp12m6_v, _ = nmc_llam.psd(ek12h,axis=1)
    wnumlist.append(wnum)
    splist.append(sp12m6_v)
    lam2gm = interp1d(ix_lam,x12m6_lam,axis=1)
    wnum_c, csp12m6 = nmc_llam.cpsd(ek12h,lam2gm(ix_gm[i0:i1+1]),axis=1)
    sp_gm2lamlist.append(csp12m6)
    ### 24h - 12h
    lam2gm = interp1d(ix_lam,x24hlam[ns:ne],axis=1)
    x24hlam2gm = lam2gm(ix_gm[i0:i1+1])
    ek24h = x24hgm[ns:ne,i0:i1+1]- x24hlam2gm #\sim H_1(x^{a}_{GM}) - H_2(x^{t}_{LAM})
    V24m12 = np.dot(ek24h.T,ek24h)/float(ne-ns+1)*0.5
    B24m12_gm2lam = np.dot(ek24h.T,x24m12_lam)/float(ne-ns+1)*0.5
    vmatlist.append(V24m12)
    gm2lamlist.append(B24m12_gm2lam)
    wnum, sp24m12_v, _ = nmc_llam.psd(ek24h,axis=1)
    splist.append(sp24m12_v)
    lam2gm = interp1d(ix_lam,x24m12_lam,axis=1)
    wnum_c, csp24m12 = nmc_llam.cpsd(ek24h,lam2gm(ix_gm[i0:i1+1]),axis=1)
    sp_gm2lamlist.append(csp24m12)
    ### 48h - 24h
    lam2gm = interp1d(ix_lam,x48hlam[ns:ne],axis=1)
    x48hlam2gm = lam2gm(ix_gm[i0:i1+1])
    ek48h = x48hgm[ns:ne,i0:i1+1] - x48hlam2gm #\sim H_1(x^{a}_{GM}) - H_2(x^{t}_{LAM})
    V48m24 = np.dot(ek48h.T,ek48h)/float(ne-ns+1)*0.5
    B48m24_gm2lam = np.dot(ek48h.T,x48m24_lam)/float(ne-ns+1)*0.5
    vmatlist.append(V48m24)
    gm2lamlist.append(B48m24_gm2lam)
    wnum, sp48m24_v, _ = nmc_llam.psd(ek48h,axis=1)
    splist.append(sp48m24_v)
    lam2gm = interp1d(ix_lam,x48m24_lam,axis=1)
    wnum_c, csp48m24 = nmc_llam.cpsd(ek48h,lam2gm(ix_gm[i0:i1+1]),axis=1)
    sp_gm2lamlist.append(csp48m24)
    ## truncated GM into nominal LAM grid (H_1:GM=>LAM@truncation, H_2=truncation)
    ixlist.append(ix_trunc)
    nmclist.append(nmc_trunc)
    ### 12h - 6h
    x12hgm2lam = np.dot(x12hgm[ns:ne],H_gm2lam.T)
    x12hgm_trunc = trunc_operator(x12hgm2lam.T)
    x12hlam_trunc = trunc_operator(x12hlam[ns:ne].T)
    ek12h = x12hgm_trunc.T - x12hlam_trunc.T #\sim H_1(x^{a}_{GM}) - H_2(x^{t}_{LAM})
    V12m6 = np.dot(ek12h.T,ek12h)/float(ne-ns+1)*0.5
    B12m6_gm2lam = np.dot(ek12h.T,x12m6_lam)/float(ne-ns+1)*0.5
    vmatlist.append(V12m6)
    gm2lamlist.append(B12m6_gm2lam)
    wnum, sp12m6_v, _ = nmc_trunc.psd(ek12h,axis=1)
    wnumlist.append(wnum)
    splist.append(sp12m6_v)
    x12m6_trunc = trunc_operator(x12m6_lam.T)
    wnum_c, csp12m6 = nmc_trunc.cpsd(ek12h,x12m6_trunc.T,axis=1)
    sp_gm2lamlist.append(csp12m6)
    ### 24h - 12h
    x24hgm2lam = np.dot(x24hgm[ns:ne],H_gm2lam.T)
    x24hgm_trunc = trunc_operator(x24hgm2lam.T)
    x24hlam_trunc = trunc_operator(x24hlam[ns:ne].T)
    ek24h = x24hgm_trunc.T - x24hlam_trunc.T #\sim H_1(x^{a}_{GM}) - H_2(x^{t}_{LAM})
    V24m12 = np.dot(ek24h.T,ek24h)/float(ne-ns+1)*0.5
    B24m12_gm2lam = np.dot(ek24h.T,x24m12_lam)/float(ne-ns+1)*0.5
    vmatlist.append(V24m12)
    gm2lamlist.append(B24m12_gm2lam)
    wnum, sp24m12_v, _ = nmc_trunc.psd(ek24h,axis=1)
    splist.append(sp24m12_v)
    x24m12_trunc = trunc_operator(x24m12_lam.T)
    wnum_c, csp24m12 = nmc_trunc.cpsd(ek24h,x24m12_trunc.T,axis=1)
    sp_gm2lamlist.append(csp24m12)
    ### 48h - 24h
    x48hgm2lam = np.dot(x48hgm[ns:ne],H_gm2lam.T)
    x48hgm_trunc = trunc_operator(x48hgm2lam.T)
    x48hlam_trunc = trunc_operator(x48hlam[ns:ne].T)
    ek48h = x48hgm_trunc.T - x48hlam_trunc.T #\sim H_1(x^{a}_{GM}) - H_2(x^{t}_{LAM})
    V48m24 = np.dot(ek48h.T,ek48h)/float(ne-ns+1)*0.5
    B48m24_gm2lam = np.dot(ek48h.T,x48m24_lam)/float(ne-ns+1)*0.5
    vmatlist.append(V48m24)
    gm2lamlist.append(B48m24_gm2lam)
    wnum, sp48m24_v, _ = nmc_trunc.psd(ek48h,axis=1)
    splist.append(sp48m24_v)
    x48m24_trunc = trunc_operator(x48m24_lam.T)
    wnum_c, csp48m24 = nmc_trunc.cpsd(ek48h,x48m24_trunc.T,axis=1)
    sp_gm2lamlist.append(csp48m24)
    ##gm2lam = interp1d(ix_gm,x12m6_gm,axis=1)
    #V12m6 = np.dot(x12m6_gm[:,i0:i1+1].T,x12m6_gm[:,i0:i1+1])/float(ne-ns+1)*0.5
    #B12m6_gm2lam = np.dot(x12m6_gm[:,i0:i1+1].T,x12m6_lam)/float(ne-ns+1)*0.5
    ##gm2lam = interp1d(ix_gm,x24m12_gm,axis=1)
    #V24m12 = np.dot(x24m12_gm[:,i0:i1+1].T,x24m12_gm[:,i0:i1+1])/float(ne-ns+1)*0.5
    #B24m12_gm2lam = np.dot(x24m12_gm[:,i0:i1+1].T,x24m12_lam)/float(ne-ns+1)*0.5
    #V48m24 = np.dot(x48m24_gm[:,i0:i1+1].T,x48m24_gm[:,i0:i1+1])/float(ne-ns+1)*0.5
    #B48m24_gm2lam = np.dot(x48m24_gm[:,i0:i1+1].T,x48m24_lam)/float(ne-ns+1)*0.5
    ## plot
    fnamelist = ['highres','lowres','trunc']
    titlelist = [
        'GM interpolated into nominal LAM grid (H_1:GM=>LAM, H_2=I)',
        'extract GM in the LAM domain (H_1:crop, H_2:LAM=>cropped GM)',
        f'truncation at n={ntrunc}'
        ]
    for k in range(len(fnamelist)):
        fname = fnamelist[k]
        title = titlelist[k]
        ixtmp = ixlist[k]
        ixtmp_deg = np.rad2deg(ixtmp)
        nmc = nmclist[k]
        wnum = wnumlist[k]
        n = ixtmp.size
        j = 3*k
        V12m6 = vmatlist[j]
        sp12m6 = splist[j]
        B12m6_gm2lam = gm2lamlist[j]
        csp12m6 = sp_gm2lamlist[j]
        j+=1
        V24m12 = vmatlist[j]
        sp24m12 = splist[j]
        B24m12_gm2lam = gm2lamlist[j]
        csp24m12 = sp_gm2lamlist[j]
        j+=1
        V48m24 = vmatlist[j]
        sp48m24 = splist[j]
        B48m24_gm2lam = gm2lamlist[j]
        csp48m24 = sp_gm2lamlist[j]

        fig = plt.figure(figsize=[12,8],constrained_layout=True)
        gs = gridspec.GridSpec(2,1,figure=fig)
        gs0 = gs[0].subgridspec(1,3)
        ax00 = fig.add_subplot(gs0[:,0])
        ax01 = fig.add_subplot(gs0[:,1])
        ax02 = fig.add_subplot(gs0[:,2])
        vlim = max(np.max(V12m6),-np.min(V12m6))
        mp0 = ax00.pcolormesh(ixtmp_deg, ixtmp_deg, V12m6, shading='auto',\
            cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
        ax00.set_xticks(ixtmp_deg[::(n//6)])
        ax00.set_yticks(ixtmp_deg[::(n//6)])
        ax00.set_title("12h - 6h")
        ax00.set_aspect("equal")
        p0 = fig.colorbar(mp0,ax=ax00,shrink=0.5,pad=0.01) #,orientation="horizontal")
        vlim = max(np.max(V24m12),-np.min(V24m12))
        mp1 = ax01.pcolormesh(ixtmp_deg, ixtmp_deg, V24m12, shading='auto',\
            cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
        ax01.set_xticks(ixtmp_deg[::(n//6)])
        ax01.set_yticks(ixtmp_deg[::(n//6)])
        ax01.set_title("24h - 12h")
        ax01.set_aspect("equal")
        p1 = fig.colorbar(mp1,ax=ax01,shrink=0.5,pad=0.01) #,orientation="horizontal")
        vlim = max(np.max(V48m24),-np.min(V48m24))
        mp2 = ax02.pcolormesh(ixtmp_deg, ixtmp_deg, V48m24, shading='auto',\
            cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
        ax02.set_xticks(ixtmp_deg[::(n//6)])
        ax02.set_yticks(ixtmp_deg[::(n//6)])
        ax02.set_title("48h - 24h")
        ax02.set_aspect("equal")
        p2 = fig.colorbar(mp2,ax=ax02,shrink=0.5,pad=0.01) #,orientation="horizontal")
        gs1 = gs[1].subgridspec(1,3)
        ax10 = fig.add_subplot(gs1[:,0])
        ax11 = fig.add_subplot(gs1[:,1])
        ax12 = fig.add_subplot(gs1[:,2])
        ### standard deviation
        data = [
            np.sqrt(np.diag(V12m6)),
            np.sqrt(np.diag(V24m12)),
            np.sqrt(np.diag(V48m24)),
            ]
        labels = [
            "12h - 6h",
            "24h - 12h",
            "48h - 24h"
        ]
        bp0=ax12.boxplot(data, vert=True, sym='+')
        ax12.yaxis.grid(True, linestyle='-', which='major', color='lightgray', alpha=0.5)
        ax12.set_xticks(np.arange(1,len(data)+1))
        ax12.set_xticklabels(labels)
        ax12.set(axisbelow=True,title=r"$\sigma$")
        for i in range(len(data)):
            med = bp0['medians'][i]
            ax12.plot(np.average(med.get_xdata()),np.average(data[i]),color='r',marker='*',markeredgecolor='k')
            s = str(round(np.average(data[i]),3))
            ax12.text(np.average(med.get_xdata()),.95,s,
            transform=ax12.get_xaxis_transform(),ha='center',c='r')
        ### correlation length scale
        L12m6_v  = nmc.corrscale(V12m6)
        L24m12_v = nmc.corrscale(V24m12)
        L48m24_v = nmc.corrscale(V48m24)
        data = [
            np.rad2deg(L12m6_v),
            np.rad2deg(L24m12_v),
            np.rad2deg(L48m24_v),
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
        ax10.plot(wnum,sp12m6,label='12h - 6h')
        ax10.plot(wnum,sp24m12,label='24h - 12h')
        ax10.plot(wnum,sp48m24,label='48h - 24h')
        ax10.set_yscale("log")
        ax10.set(xlabel=r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)",title='variance power spectra')
        ax10.set_xscale('log')
        #ax10.xaxis.set_major_locator(FixedLocator([240./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi]))
        #ax10.xaxis.set_major_formatter(FixedFormatter([r'$\frac{240}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$']))
        ax10.xaxis.set_major_locator(FixedLocator([480,240,120,60,8,2,1]))
        ax10.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','8','2','1']))
        #ax10.set_xlim(0.5/np.pi,wnum[-1])
        secax = ax10.secondary_xaxis('top',functions=(wnum2wlen, wlen2wnum))
        secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
        secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi,np.pi/4,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
        secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\pi$',r'$\frac{\pi}{4}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
        ax10.legend()
        fig.suptitle(f"NMC for {title} : "+pt+" "+op)
        fig.savefig("{}_nmcv_{}_{}_{}.png".format(model,fname,op,pt))
        plt.show(block=False)
        plt.close(fig=fig)
        np.save("{}_V12m6_{}.npy".format(model,fname),V12m6)
        np.save("{}_V24m12_{}.npy".format(model,fname),V24m12)
        np.save("{}_V48m24_{}.npy".format(model,fname),V48m24)
        ## plot
        fig, axs = plt.subplots(nrows=2,ncols=3,figsize=[12,8],\
            constrained_layout=True)
        vlim = max(np.max(B12m6_gm2lam),-np.min(B12m6_gm2lam))
        mp0 = axs[0,0].pcolormesh(ix_lam_deg, ixtmp_deg, B12m6_gm2lam, shading='auto',\
            cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
        axs[0,0].set_xticks(ix_lam_deg[::(nx//6)])
        axs[0,0].set_yticks(ixtmp_deg[::(n//6)])
        axs[0,0].set_title("12h - 6h")
        #axs[0,0].set_aspect(n/nx)
        p0 = fig.colorbar(mp0,ax=axs[0,0],shrink=0.5,pad=0.01) #,orientation="horizontal")
        vlim = max(np.max(B24m12_gm2lam),-np.min(B24m12_gm2lam))
        mp1 = axs[0,1].pcolormesh(ix_lam_deg, ixtmp_deg, B24m12_gm2lam, shading='auto',\
            cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
        axs[0,1].set_xticks(ix_lam_deg[::(nx//6)])
        axs[0,1].set_yticks(ixtmp_deg[::(n//6)])
        axs[0,1].set_title("24h - 12h")
        #axs[0,1].set_aspect(n/nx)
        p1 = fig.colorbar(mp1,ax=axs[0,1],shrink=0.5,pad=0.01) #,orientation="horizontal")
        vlim = max(np.max(B48m24_gm2lam),-np.min(B48m24_gm2lam))
        mp2 = axs[0,2].pcolormesh(ix_lam_deg, ixtmp_deg, B48m24_gm2lam, shading='auto',\
            cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
        axs[0,2].set_xticks(ix_lam_deg[::(nx//6)])
        axs[0,2].set_yticks(ixtmp_deg[::(n//6)])
        axs[0,2].set_title("48h - 24h")
        #axs[0,2].set_aspect(n/nx)
        p2 = fig.colorbar(mp2,ax=axs[0,2],shrink=0.5,pad=0.01) #,orientation="horizontal")
        #### diagonal
        #axs[1,0].plot(ix_gm[i0:i1+1],np.diag(B12m6_gm2lam),label="12h - 6h")
        #axs[1,0].plot(ix_gm[i0:i1+1],np.diag(B24m12_gm2lam),label="24h - 12h")
        #axs[1,0].set_xticks(ix_gm[i0:i1+1:(n//6)])
        #axs[1,0].set_title("Diagonal")
        #axs[1,0].legend()
        axs[1,0].remove()
        #### row
        #axs[1,1].plot(ix_lam_deg,B12m6_gm2lam[n//2,:],label="12h - 6h")
        #axs[1,1].plot(ix_lam_deg,B24m12_gm2lam[n//2,:],label="24h - 12h")
        #axs[1,1].plot(ix_lam_deg,B48m24_gm2lam[n//2,:],label="48h - 24h")
        #axs[1,1].set_xticks(ix_lam_deg[::(nx//6)])
        #axs[1,1].set_title("Row")
        #axs[1,1].legend()
        ### cross variance spectra
        axs[1,1].plot(wnum,csp12m6,label='12h - 6h')
        axs[1,1].plot(wnum,csp24m12,label='24h - 12h')
        axs[1,1].plot(wnum,csp48m24,label='48h - 24h')
        axs[1,1].set_yscale("log")
        axs[1,1].set(xlabel=r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)",title='variance power spectra')
        axs[1,1].set_xscale('log')
        #axs[1,1].xaxis.set_major_locator(FixedLocator([240./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi]))
        #axs[1,1].xaxis.set_major_formatter(FixedFormatter([r'$\frac{240}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$']))
        axs[1,1].xaxis.set_major_locator(FixedLocator([480,240,120,60,8,2,1]))
        axs[1,1].xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','8','2','1']))
        #axs[1,1].set_xlim(0.5/np.pi,wnum[-1])
        secax = axs[1,1].secondary_xaxis('top',functions=(wnum2wlen, wlen2wnum))
        secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
        secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi,np.pi/4,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
        secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\pi$',r'$\frac{\pi}{4}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
        axs[1,1].legend()
        axs[1,2].remove()
        fig.suptitle(f"NMC for {title} x LAM : "+pt+" "+op)
        fig.savefig("{}_nmcgm2lam_{}_{}_{}.png".format(model,fname,op,pt))
        plt.show(block=False)
        plt.close(fig=fig)
        np.save("{}_B12m6_gm2lam_{}.npy".format(model,fname),B12m6_gm2lam)
        np.save("{}_B24m12_gm2lam_{}.npy".format(model,fname),B24m12_gm2lam)
        np.save("{}_B48m24_gm2lam_{}.npy".format(model,fname),B48m24_gm2lam)
        if fname=='highres':
            axsp.plot(wnum,sp48m24,c='orange',ls='dotted',label=f'V_{fname}')
            axsp.plot(wnum,csp48m24,c='purple',ls='dashdot',label=f'Ekb_{fname}')
        elif fname=='trunc':
            axsp.plot(wnum,sp48m24,c='green',label=f'V_{fname}')
            axsp.plot(wnum,csp48m24,c='magenta',ls='dashdot',label=f'Ekb_{fname}')
    axsp.set_yscale("log")
    axsp.set_ylim(1e-11,15.)
    axsp.set(xlabel=r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)",title='variance power spectra')
    axsp.set_xscale('log')
    #axsp.xaxis.set_major_locator(FixedLocator([240./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi]))
    #axsp.xaxis.set_major_formatter(FixedFormatter([r'$\frac{240}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$']))
    axsp.xaxis.set_major_locator(FixedLocator([480,240,120,60,8,2,1]))
    axsp.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','8','2','1']))
    #axsp.set_xlim(0.5/np.pi,wnum[-1])
    secax = axsp.secondary_xaxis('top',functions=(wnum2wlen, wlen2wnum))
    secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
    secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi,np.pi/4,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
    secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\pi$',r'$\frac{\pi}{4}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
    axsp.grid()
    axsp.legend()
    figsp.savefig("{}_nmc_varsp_{}_{}.png".format(model,op,pt))
    plt.show()
    plt.close(fig=figsp)