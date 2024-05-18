import sys
import os
import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.colors import Normalize
from nmc_tools import psd, wnum2wlen, wlen2wnum
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__),'../analysis'))
from trunc1d import Trunc1d

plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 24

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])

t = np.arange(na)+1
ns = 40 # spinup

datadir = Path(f'/Volumes/FF520/nested_envar/data/{model}')
#preGMpt = 'envar'
#dscldir = datadir / 'var_vs_envar_dscl_m80obs30'
lamdir  = datadir / 'var_vs_envar_preGM_m80obs30'

perts = ["envar", "envar_nest","var","var_nest"]
labels = {"envar":"EnVar", "envar_nest":"Nested EnVar", "var":"3DVar", "var_nest":"Nested 3DVar"}
linecolor = {"envar":'tab:orange',"envar_nest":'tab:green',"var":"tab:olive","var_nest":"tab:brown"}

ix_t = np.loadtxt(lamdir/"ix_true.txt")
ix_gm = np.loadtxt(lamdir/"ix_gm.txt")
ix_lam = np.loadtxt(lamdir/"ix_lam.txt")[1:-1]
nx_t = ix_t.size
nx_gm = ix_gm.size
nx_lam = ix_lam.size
xlim = 15.0
nghost = 0 # ghost region for periodicity in LAM
ix_t_rad = ix_t * 2.0 * np.pi / nx_t
ix_gm_rad = ix_gm * 2.0 * np.pi / nx_t
ix_lam_rad = ix_lam * 2.0 * np.pi / nx_t
Lx_gm = 2.0 * np.pi
#dwindow = (1.0 + np.cos(np.pi*np.arange(1,nghost+1)/nghost))*0.5
Lx_lam = 2.0 * np.pi * nx_lam / nx_t

ntrunc = 12
trunc_operator = Trunc1d(ix_lam,ntrunc=ntrunc,cyclic=False,nghost=0)
ix_trunc = trunc_operator.ix_trunc
nx_gmlam = ix_trunc.size

#figsp, axsp = plt.subplots(figsize=[10,8],constrained_layout=True)
#psd_dict = {}

pt="envar_nest"
scycle = 40
ecycle = 1000
ncycle_lam = 0
vmat_exist = False
for icycle in range(scycle,ecycle+1):
    f = lamdir/"{}_lam_spf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if f.exists():
        spftmp = np.load(f)
        pftmp = spftmp @ spftmp.T
        if ncycle_lam==0:
            pflam = pftmp
            #wnum_lam, psdlam, _ = psd(spftmp,ix_lam_rad,axis=0,\
            #    cyclic=False,nghost=0,average=True,detrend=True)
        else:
            pflam = pflam + pftmp
            #wnum_lam, psdtmp, _ = psd(spftmp,ix_lam_rad,axis=0,\
            #    cyclic=False,nghost=0,average=True,detrend=True)
            #psdlam = psdlam + psdtmp
        ncycle_lam += 1
    else:
        f = lamdir/"data/{2}/{0}_lam_spf_{1}_{2}_cycle{3}.npy".format(model, op, pt, icycle)
        if f.exists():
            spftmp = np.load(f)
            pftmp = spftmp @ spftmp.T
            if ncycle_lam==0:
                pflam = pftmp
                #wnum_lam, psdlam, _ = psd(spftmp,ix_lam_rad,axis=0,\
                #    cyclic=False,nghost=0,average=True,detrend=True)
            else:
                pflam = pflam + pftmp
                #wnum_lam, psdtmp, _ = psd(spftmp,ix_lam_rad,axis=0,\
                #    cyclic=False,nghost=0,average=True,detrend=True)
                #psdlam = psdlam + psdtmp
            ncycle_lam += 1
    f = lamdir/"{}_lam_svmat_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if f.exists():
        svtmp = np.load(f)
        vtmp = svtmp @ svtmp.T
        if not vmat_exist:
            vmat = vtmp
            #wnum_v, psdv, _ = psd(svtmp,ix_trunc_rad,axis=0,\
            #    cyclic=False,nghost=0,average=True,detrend=True)
        else:
            vmat = vmat + vtmp
            #wnum_v, psdtmp, _ = psd(svtmp,ix_trunc_rad,axis=0,\
            #    cyclic=False,nghost=0,average=True,detrend=True)
            #psdv = psdv + psdtmp
        vmat_exist=True
    else:
        f = lamdir/"data/{2}/{0}_lam_svmat_{1}_{2}_cycle{3}.npy".format(model, op, pt, icycle)
        if f.exists():
            svtmp = np.load(f)
            vtmp = svtmp @ svtmp.T
            if not vmat_exist:
                vmat = vtmp
                #wnum_v, psdv, _ = psd(svtmp,ix_trunc_rad,axis=0,\
                #    cyclic=False,nghost=0,average=True,detrend=True)
            else:
                vmat = vmat + vtmp
                #wnum_v, psdtmp, _ = psd(svtmp,ix_trunc_rad,axis=0,\
                #    cyclic=False,nghost=0,average=True,detrend=True)
                #psdv = psdv + psdtmp
            vmat_exist=True
    if ncycle_lam == 1 and vmat_exist:
        cmat = spftmp @ svtmp.T
    elif vmat_exist:
        cmat = cmat + (spftmp @ svtmp.T)
if ncycle_lam == 0: exit()
pflam = pflam / float(ncycle_lam)
if vmat_exist:
    vmat = vmat / float(ncycle_lam)
    cmat = cmat / float(ncycle_lam)
fig, axs = plt.subplots(nrows=2,ncols=2,figsize=[12,12],constrained_layout=True)
vlim = 0.15
#vlim = max(np.max(pflam),-np.min(pflam))
mp00 = axs[0,0].pcolormesh(ix_lam,ix_lam,pflam,shading='auto',\
    cmap='coolwarm',norm=Normalize(-vlim,vlim))
fig.colorbar(mp00,ax=axs[0,0],pad=0.01,shrink=0.6)
axs[0,0].set_title(r'$\mathbf{P}^\mathrm{f}$')
#vlim = max(np.max(vmat),-np.min(vmat))
mp11 = axs[1,1].pcolormesh(ix_trunc,ix_trunc,vmat,shading='auto',\
    cmap='coolwarm',norm=Normalize(-vlim,vlim))
fig.colorbar(mp11,ax=axs[1,1],pad=0.01,shrink=0.6)
axs[1,1].set_title(r'$\mathbf{P}^\mathrm{v}$')
#vlim = max(np.max(cmat),-np.min(cmat))
mp01 = axs[0,1].pcolormesh(ix_trunc,ix_lam,cmat,shading='auto',\
    cmap='coolwarm',norm=Normalize(-vlim,vlim))
fig.colorbar(mp01,ax=axs[0,1],pad=0.01,shrink=0.6)
axs[0,1].set_title(r'$\mathbf{X}^\mathrm{b}(\mathbf{Z}^\mathrm{b})^\mathrm{T}$')
mp10 = axs[1,0].pcolormesh(ix_lam,ix_trunc,cmat.T,shading='auto',\
    cmap='coolwarm',norm=Normalize(-vlim,vlim))
fig.colorbar(mp10,ax=axs[1,0],pad=0.01,shrink=0.6)
axs[1,0].set_title(r'$\mathbf{Z}^\mathrm{b}(\mathbf{X}^\mathrm{b})^\mathrm{T}$')
for ax in axs.flatten():
    ax.set_aspect("equal")
    ax.set_xticks(ix_lam[::(nx_lam//8)])
    ax.set_yticks(ix_lam[::(nx_lam//8)])

#fig.suptitle(f"{labels[pt]}, cycle={scycle}-{ecycle}")
fig.savefig(lamdir/"{}_pf_{}_{}_cycle{}-{}.png".format(model,op,pt,scycle,ecycle))
plt.show()
