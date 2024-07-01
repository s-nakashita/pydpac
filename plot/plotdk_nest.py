import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
plt.rcParams['font.size'] = 16
from methods import perts

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
dscl = False
if len(sys.argv)>4:
    dscl = (sys.argv[4]=='T')
print(dscl)
cmap = "coolwarm"
f = "truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[:na,]
print(xt.shape)
nx = xt.shape[1]
t = np.arange(na)
ix_t = np.loadtxt("ix_true.txt")
ix_gm = np.loadtxt("ix_gm.txt")
ix_lam = np.loadtxt("ix_lam.txt")
i0 = np.argmin(np.abs(ix_gm - ix_lam[0]))
if ix_gm[i0]<ix_lam[0]: i0+=1
i1 = np.argmin(np.abs(ix_gm - ix_lam[-1]))
if ix_gm[i1]>ix_lam[-1]: i1-=1
tmp_lam2gm = interp1d(ix_lam,np.eye(ix_lam.size),axis=0)
JH2 = tmp_lam2gm(ix_gm[i0:i1+1])
xlim = 15.0
for pt in perts:
    #GM
    ## nature and analysis
    fig, axs = plt.subplots(nrows=1,ncols=3,figsize=[12,6],\
        constrained_layout=True,sharey=True)
    if dscl:
        f = "{}_xfgmonly_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xfgm_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xfgm = np.load(f)
    print(xfgm.shape)
    nx = ix_gm.size
    mp0 = axs[0].pcolormesh(ix_gm, t, xfgm, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[0].set_xticks(ix_gm[::(nx//8)])
    axs[0].set_yticks(t[::max(1,na//8)])
    axs[0].set_xlabel("site")
    axs[0].set_title("GM forecast")
    p0 = fig.colorbar(mp0,ax=axs[0],orientation="horizontal")
    axs[0].vlines([ix_lam[0],ix_lam[-1]],0,1,\
            colors='black',linestyle='dashdot',transform=axs[0].get_xaxis_transform())
    #LAM
    if dscl:
        f = "{}_xfdscl_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xflam_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xflam = np.load(f)
    print(xflam.shape)
    nx = ix_lam.size
    mp1 = axs[1].pcolormesh(ix_lam, t, xflam, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    p1 = fig.colorbar(mp1,ax=axs[1],orientation="horizontal")
    axs[1].set_xticks(ix_lam[::(nx//8)])
    axs[1].set_yticks(t[::max(1,na//8)])
    axs[1].set_xlabel("site")
    axs[1].set_title("LAM forecast")
    axs[1].set_xlim(ix_lam[0],ix_lam[-1])
    #dk
    xd = xfgm[:,i0:i1+1] - (JH2 @ xflam[:,:].T).T
    vlim = np.nanmax(np.abs(xd))
    mp2 = axs[2].pcolormesh(ix_gm[i0:i1+1], t, xd, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[2].set_xticks(ix_gm[i0:i1+1:8])
    axs[2].set_yticks(t[::max(1,na//8)])
    axs[2].set_xlabel("site")
    p2 = fig.colorbar(mp2,ax=axs[2],orientation="horizontal")
    axs[2].set_xlim(ix_gm[i0],ix_gm[i1])
    fig.suptitle("forecast difference : "+pt+" "+op)
    if dscl:
        fig.savefig("{}_dkdscl_{}_{}.png".format(model,op,pt))
    else:
        fig.savefig("{}_dk_{}_{}.png".format(model,op,pt))