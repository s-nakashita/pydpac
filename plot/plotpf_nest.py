import sys
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
sys.path.append(os.path.join(os.path.dirname(__file__),'../analysis'))
from trunc1d import Trunc1d
plt.rcParams["font.size"] = 12

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = sys.argv[4]
scycle = int(sys.argv[5])
ecycle = int(sys.argv[6])
if model == "z08":
    nx = 81
elif model == "z05":
    nx = 101
elif model == "l96":
    nx = 40
#ix = np.arange(nx)
ix_gm = np.loadtxt("ix_gm.txt")
nx_gm = ix_gm.size
ix_lam = np.loadtxt("ix_lam.txt")
nx_lam = ix_lam.size
#i0=np.argmin(np.abs(ix_gm-ix_lam[0]))
#if ix_gm[i0]<ix_lam[0]: i0+=1
#i1=np.argmin(np.abs(ix_gm-ix_lam[-1]))
#if ix_gm[i1]>ix_lam[-1]: i1-=1
#nx_gmlam = i1 - i0 + 1
ntrunc = 12
trunc_operator = Trunc1d(ix_lam,ntrunc=ntrunc,cyclic=False,nghost=0)
ix_trunc = trunc_operator.ix_trunc
nx_gmlam = ix_trunc.size

ncycle = 0
vmat_exist=False
for icycle in range(scycle,ecycle+1):
    cmap = "coolwarm"
    f = "{}_gm_spf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if os.path.isfile(f):
        spftmp = np.load(f)
        pftmp = spftmp @ spftmp.T
        if ncycle==0:
            pfgm = pftmp
        else:
            pfgm = pfgm + pftmp
    f = "{}_lam_spf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if os.path.isfile(f):
        spftmp = np.load(f)
        pftmp = spftmp @ spftmp.T
        if ncycle==0:
            pflam = pftmp
        else:
            pflam = pflam + pftmp
        ncycle += 1
    f = "{}_lam_svmat_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if os.path.isfile(f):
        svtmp = np.load(f)
        vtmp = svtmp @ svtmp.T
        if not vmat_exist:
            vmat = vtmp
        else:
            vmat = vmat + vtmp
        vmat_exist=True
#    pa = None  
#    f = "{}_pa_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
#    if os.path.isfile(f):
#        pa = np.load(f)
if ncycle==0: exit()
pfgm = pfgm / float(ncycle)
pflam = pflam / float(ncycle)
if vmat_exist:
    vmat = vmat / float(ncycle)
    fig, axs = plt.subplots(ncols=3,figsize=[12,6],constrained_layout=True)
else:
    fig, axs = plt.subplots(ncols=2,figsize=[8,6],constrained_layout=True)
ymax = np.max(pfgm)
ymin = np.min(pfgm)
ylim = max(ymax, np.abs(ymin))
mp0=axs[0].pcolormesh(ix_gm,ix_gm,pfgm,shading='auto',\
        cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
axs[0].set_aspect("equal")
axs[0].set_xticks(ix_gm[::(nx_gm//8)])
axs[0].set_yticks(ix_gm[::(nx_gm//8)])
axs[0].set_title(r"$\mathrm{trace}(\mathbf{P}^\mathrm{f}_\mathrm{GM})/N=$"+f"{np.mean(np.diag(pfgm)):.3f}")
fig.colorbar(mp0, ax=axs[0], pad=0.01, shrink=0.6) #orientation="horizontal")
ymax = np.max(pflam)
ymin = np.min(pflam)
ylim = max(ymax, np.abs(ymin))
mp1=axs[1].pcolormesh(ix_lam,ix_lam,pflam,shading='auto',\
        cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
axs[1].set_aspect("equal")
axs[1].set_xticks(ix_lam[::(nx_lam//8)])
axs[1].set_yticks(ix_lam[::(nx_lam//8)])
axs[1].set_title(r"$\mathrm{trace}(\mathbf{P}^\mathrm{f}_\mathrm{LAM})/N=$"+f"{np.mean(np.diag(pflam)):.3f}")
fig.colorbar(mp1, ax=axs[1], pad=0.01, shrink=0.6) #orientation="horizontal")
if vmat_exist:
    ymax = np.max(vmat)
    ymin = np.min(vmat)
    ylim = max(ymax, np.abs(ymin))
    #mp2=axs[2].pcolormesh(ix_gm[i0:i1+1],ix_gm[i0:i1+1],vmat,shading='auto',\
    mp2=axs[2].pcolormesh(ix_trunc,ix_trunc,vmat,shading='auto',\
            cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
    axs[2].set_aspect("equal")
    #axs[2].set_xticks(ix_gm[i0:i1+1:(nx_gmlam//8)])
    #axs[2].set_yticks(ix_gm[i0:i1+1:(nx_gmlam//8)])
    #axs[2].set_xticks(ix_lam[::(nx_lam//8)])
    #axs[2].set_yticks(ix_lam[::(nx_lam//8)])
    axs[2].set_xticks(ix_trunc[::(nx_gmlam//8)])
    axs[2].set_yticks(ix_trunc[::(nx_gmlam//8)])
    axs[2].set_title(r"$\mathrm{trace}(\mathbf{V})/N=$"+f"{np.mean(np.diag(vmat)):.3f}")
    fig.colorbar(mp2, ax=axs[2], pad=0.01, shrink=0.6) #orientation="horizontal")
fig.savefig("{}_pf_{}_{}_cycle{}-{}.png".format(model,op,pt,scycle,ecycle))
