import sys
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
plt.rcParams["font.size"] = 16
op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = sys.argv[4]
ix_t = np.loadtxt("ix_true.txt")
ix_gm = np.loadtxt("ix_gm.txt")
ix_lam = np.loadtxt("ix_lam.txt")
icycle=na-1
cmap = "coolwarm"
pagm = None
f = "{}_pagm_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
if os.path.isfile(f):
    pagm = np.load(f)
palam = None  
f = "{}_palam_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
if os.path.isfile(f):
    palam = np.load(f)
rhogm = None
f = "{}_rhogm_{}_{}.npy".format(model, op, pt)
if os.path.isfile(f):
    rhogm = np.load(f)
rholam = None  
f = "{}_rholam_{}_{}.npy".format(model, op, pt)
if os.path.isfile(f):
    rholam = np.load(f)
if rhogm is not None and rholam is not None:
    rhoplot=True
    fig, axs = plt.subplots(nrows=2,ncols=2,figsize=[10,10],constrained_layout=True)
else:
    rhoplot=False
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=[10,5],constrained_layout=True)
plot = False
if pagm is not None:
    plot = True
    ymax = np.max(pagm)
    ymin = np.min(pagm)
    ylim = max(ymax, np.abs(ymin))
    if rhoplot:
        ax = axs[0,0]
        axr = axs[1,0]
        mp1=axr.pcolormesh(ix_gm,ix_gm,rhogm,shading='auto',norm=Normalize(vmin=0.0, vmax=1.0))
        axr.set_aspect("equal")
        axr.set_xticks(ix_gm[::(ix_gm.size//8)])
        axr.set_yticks(ix_gm[::(ix_gm.size//8)])
        axr.invert_yaxis()
        axr.set_title(r"$\rho$ in GM")
        fig.colorbar(mp1, ax=axr,orientation="vertical",shrink=0.6,pad=0.01)
    else:
        ax = axs[0]
    mp0=ax.pcolormesh(ix_gm,ix_gm,pagm,shading='auto',cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
    ax.set_aspect("equal")
    ax.set_xticks(ix_gm[::(ix_gm.size//8)])
    ax.set_yticks(ix_gm[::(ix_gm.size//8)])
    ax.invert_yaxis()
    ax.set_title("Pa in GM")
    fig.colorbar(mp0, ax=ax,orientation="vertical",shrink=0.6,pad=0.01)
if palam is not None:
    plot = True
    ymax = np.max(palam)
    ymin = np.min(palam)
    ylim = max(ymax, np.abs(ymin))
    if rhoplot:
        ax = axs[0,1]
        axr = axs[1,1]
        mp3=axr.pcolormesh(ix_lam,ix_lam,rholam,shading='auto',norm=Normalize(vmin=0.0, vmax=1.0))
        axr.set_aspect("equal")
        axr.set_xticks(ix_lam[::(ix_lam.size//8)])
        axr.set_yticks(ix_lam[::(ix_lam.size//8)])
        axr.invert_yaxis()
        axr.set_title(r"$\rho$ in LAM")
        fig.colorbar(mp3, ax=axr,orientation="vertical",shrink=0.6,pad=0.01)
    else:
        ax = axs[1]
    mp2=ax.pcolormesh(ix_lam,ix_lam,palam,shading='auto',cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
    ax.set_aspect("equal")
    ax.set_xticks(ix_lam[::(ix_lam.size//8)])
    ax.set_yticks(ix_lam[::(ix_lam.size//8)])
    ax.invert_yaxis()
    ax.set_title("Pa in LAM")
    fig.colorbar(mp2, ax=ax,orientation="vertical",shrink=0.6,pad=0.01)

if plot:
    fig.savefig("{}_cov_{}_{}_cycle{}.png".format(model,op,pt,icycle))
