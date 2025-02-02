import sys
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, FixedFormatter
from nmc_tools import NMC_tools, wnum2wlen, wlen2wnum
sys.path.append(os.path.join(os.path.dirname(__file__),'../analysis'))
from trunc1d import Trunc1d

plt.rcParams["font.size"] = 12

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = sys.argv[4]
scycle = int(sys.argv[5])
ecycle = int(sys.argv[6])
if ecycle > na: ecycle=na
if model == "z08":
    nx = 81
elif model == "z05":
    nx = 101
elif model == "l96":
    nx = 40
#ix = np.arange(nx)
ix_t = np.loadtxt("ix_true.txt")
nx_t = ix_t.size
ix_gm = np.loadtxt("ix_gm.txt")
nx_gm = ix_gm.size
ix_lam = np.loadtxt("ix_lam.txt")[1:-1]
nx_lam = ix_lam.size
#i0=np.argmin(np.abs(ix_gm-ix_lam[0]))
#if ix_gm[i0]<ix_lam[0]: i0+=1
#i1=np.argmin(np.abs(ix_gm-ix_lam[-1]))
#if ix_gm[i1]>ix_lam[-1]: i1-=1
#nx_gmlam = i1 - i0 + 1
ntrunc = 12
trunc_operator = Trunc1d(ix_lam,ntrunc=ntrunc,ttype='c',cyclic=False,nghost=0)
ix_trunc = trunc_operator.ix_trunc
nx_gmlam = ix_trunc.size

Lx = 2.0 * np.pi
Lx_gm = Lx
Lx_lam = Lx * nx_lam / nx_t
ix_gm_rad = ix_gm * Lx / nx_t
ix_lam_rad = ix_lam * Lx / nx_t
ix_trunc_rad = ix_trunc * Lx / nx_t
print(ix_lam_rad)
print(ix_trunc_rad)
nmc_gm = NMC_tools(ix_gm_rad,cyclic=False,ttype='c')
nmc_lam = NMC_tools(ix_lam_rad,cyclic=False,ttype='c')
nmc_trunc = NMC_tools(ix_trunc_rad,cyclic=False,ttype='c')

ncycle_gm = 0
ncycle_lam = 0
vmat_exist=False
pfgm = None
pflam = None
psd_gm = None
psd_lam = None
for icycle in range(scycle,ecycle+1):
    f = "{}_gm_spf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if os.path.isfile(f):
        spftmp = np.load(f)
        pftmp = spftmp @ spftmp.T
        if ncycle_gm==0:
            pfgm = pftmp
            wnum_gm, psdgm = nmc_gm.psd(spftmp,axis=0,average=True)
        else:
            pfgm = pfgm + pftmp
            wnum_gm, psdtmp = nmc_gm.psd(spftmp, axis=0, average=True)
            psdgm = psdgm + psdtmp
        ncycle_gm += 1
    else:
        f = "data/{2}/{0}_gm_spf_{1}_{2}_cycle{3}.npy".format(model, op, pt, icycle)
        if os.path.isfile(f):
            spftmp = np.load(f)
            pftmp = spftmp @ spftmp.T
            if ncycle_gm==0:
                pfgm = pftmp
                wnum_gm, psdgm = nmc_gm.psd(spftmp,axis=0,average=True)
            else:
                pfgm = pfgm + pftmp
                wnum_gm, psdtmp = nmc_gm.psd(spftmp,axis=0,average=True)
                psdgm = psdgm + psdtmp
            ncycle_gm += 1
    f = "{}_lam_spf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if os.path.isfile(f):
        spftmp = np.load(f)
        pftmp = spftmp @ spftmp.T
        if ncycle_lam==0:
            pflam = pftmp
            wnum_lam, psdlam = nmc_lam.psd(spftmp,axis=0,average=True)
        else:
            pflam = pflam + pftmp
            wnum_lam, psdtmp = nmc_lam.psd(spftmp,axis=0,average=True)
            psdlam = psdlam + psdtmp
        ncycle_lam += 1
    else:
        f = "data/{2}/{0}_lam_spf_{1}_{2}_cycle{3}.npy".format(model, op, pt, icycle)
        if os.path.isfile(f):
            spftmp = np.load(f)
            pftmp = spftmp @ spftmp.T
            if ncycle_lam==0:
                pflam = pftmp
                wnum_lam, psdlam = nmc_lam.psd(spftmp,axis=0,average=True)
            else:
                pflam = pflam + pftmp
                wnum_lam, psdtmp = nmc_lam.psd(spftmp,axis=0,average=True)
                psdlam = psdlam + psdtmp
            ncycle_lam += 1
    f = "{}_lam_svmat_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if os.path.isfile(f):
        svtmp = np.load(f)
        vtmp = svtmp @ svtmp.T
        if not vmat_exist:
            vmat = vtmp
            wnum_v, psdv = nmc_trunc.psd(svtmp,axis=0,average=True)
        else:
            vmat = vmat + vtmp
            wnum_v, psdtmp = nmc_trunc.psd(svtmp,axis=0,average=True)
            psdv = psdv + psdtmp
        vmat_exist=True
    else:
        f = "data/{2}/{0}_lam_svmat_{1}_{2}_cycle{3}.npy".format(model, op, pt, icycle)
        if os.path.isfile(f):
            svtmp = np.load(f)
            vtmp = svtmp @ svtmp.T
            if not vmat_exist:
                vmat = vtmp
                wnum_v, psdv = nmc_trunc.psd(svtmp,axis=0,average=True)
            else:
                vmat = vmat + vtmp
                wnum_v, psdtmp = nmc_trunc.psd(svtmp,axis=0,average=True)
                psdv = psdv + psdtmp
            vmat_exist=True
#    pa = None  
#    f = "{}_pa_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
#    if os.path.isfile(f):
#        pa = np.load(f)
if ncycle_gm==0 and ncycle_lam==0: exit()
nrows=0
ncols=2
if pfgm is not None: 
    pfgm = pfgm / float(ncycle_gm)
    psdgm = psdgm / float(ncycle_gm)
    nrows+=1
if pflam is not None: 
    pflam = pflam / float(ncycle_lam)
    psdlam = psdlam / float(ncycle_lam)
    nrows+=1
cmap = "coolwarm"
if vmat_exist:
    vmat = vmat / float(ncycle_lam)
    psdv = psdv / float(ncycle_lam)
    nrows+=1
#    fig, axs = plt.subplots(ncols=3,figsize=[12,4],constrained_layout=True)
#else:
#    fig, axs = plt.subplots(ncols=2,figsize=[8,4],constrained_layout=True)
figwidth = ncols * 4
figheight = nrows * 4
fig = plt.figure(figsize=[figwidth,figheight],constrained_layout=True)
gs = GridSpec(nrows,ncols,figure=fig)
axpsd = fig.add_subplot(gs[nrows-2,1])
irow = 0
if pfgm is not None:
    ax0 = fig.add_subplot(gs[irow,0])
    irow+=1
    ymax = np.max(pfgm)
    ymin = np.min(pfgm)
    ylim = max(ymax, np.abs(ymin))
    mp0=ax0.pcolormesh(ix_gm,ix_gm,pfgm,shading='auto',\
        cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
    ax0.set_aspect("equal")
    ax0.set_xticks(ix_gm[::(nx_gm//8)])
    ax0.set_yticks(ix_gm[::(nx_gm//8)])
    ax0.set_title(r"$\mathrm{trace}(\mathbf{P}^\mathrm{f}_\mathrm{GM})/N=$"+f"{np.mean(np.diag(pfgm)):.3f}")
    fig.colorbar(mp0, ax=ax0, pad=0.01, shrink=0.6) #orientation="horizontal")
    axpsd.loglog(wnum_gm,psdgm,marker='x',label='GM')
#else:
#    axs[0].remove()

if pflam is not None:
    ax1 = fig.add_subplot(gs[irow,0])
    irow += 1
    ymax = np.max(pflam)
    ymin = np.min(pflam)
    ylim = max(ymax, np.abs(ymin))
    mp1=ax1.pcolormesh(ix_lam,ix_lam,pflam,shading='auto',\
        cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
    ax1.set_aspect("equal")
    ax1.set_xticks(ix_lam[::(nx_lam//8)])
    ax1.set_yticks(ix_lam[::(nx_lam//8)])
    ax1.set_title(r"$\mathrm{trace}(\mathbf{P}^\mathrm{f}_\mathrm{LAM})/N=$"+f"{np.mean(np.diag(pflam)):.3f}")
    fig.colorbar(mp1, ax=ax1, pad=0.01, shrink=0.6) #orientation="horizontal")
    axpsd.loglog(wnum_lam,psdlam,marker='x',label='LAM')
#else:
#    axs[1].remove()

if vmat_exist:
    ax2 = fig.add_subplot(gs[irow,0])
    ymax = np.max(vmat)
    ymin = np.min(vmat)
    ylim = max(ymax, np.abs(ymin))
    #mp2=axs[2].pcolormesh(ix_gm[i0:i1+1],ix_gm[i0:i1+1],vmat,shading='auto',\
    mp2=ax2.pcolormesh(ix_trunc,ix_trunc,vmat,shading='auto',\
            cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
    ax2.set_aspect("equal")
    #axs[2].set_xticks(ix_gm[i0:i1+1:(nx_gmlam//8)])
    #axs[2].set_yticks(ix_gm[i0:i1+1:(nx_gmlam//8)])
    #axs[2].set_xticks(ix_lam[::(nx_lam//8)])
    #axs[2].set_yticks(ix_lam[::(nx_lam//8)])
    ax2.set_xticks(ix_trunc[::(nx_gmlam//8)])
    ax2.set_yticks(ix_trunc[::(nx_gmlam//8)])
    ax2.set_title(r"$\mathrm{trace}(\mathbf{V})/N=$"+f"{np.mean(np.diag(vmat)):.3f}")
    fig.colorbar(mp2, ax=ax2, pad=0.01, shrink=0.6) #orientation="horizontal")
    axpsd.loglog(wnum_v,psdv,marker='x',label='V')
axpsd.set_ylabel("Spectral density of ensemble spread")
axpsd.set_xlabel(r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)")
axpsd.xaxis.set_major_locator(FixedLocator([480,240,120,60,30,1]))
axpsd.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','1']))
secax = axpsd.secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
axpsd.grid()
axpsd.legend()

fig.suptitle(f"{pt}, cycle={scycle}-{ecycle}")
fig.savefig("{}_pf_{}_{}_cycle{}-{}.png".format(model,op,pt,scycle,ecycle))
