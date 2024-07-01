import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import collections
plt.rcParams['font.size'] = 16
from methods import perts, linecolor

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0}
try:
    with open("params.txt","r") as f:
        ptype=f.readline()[:-1]
        vargm=[]
        varlam=[]
        while(True):
            tmp=f.readline()[:-1]
            if tmp=='': break
            tmp2 = tmp.split()
            vargm.append(tmp2[0])
            varlam.append(tmp2[1])
#            var.append(f"g{tmp2[0]}l{tmp2[1]}")
except FileNotFoundError:
    print("not found params.txt")
    exit()
## remove overlap
vargm = [k for k, v in collections.Counter(vargm).items() if v > 1]
varlam = [k for k, v in collections.Counter(varlam).items() if v > 1]
print(vargm)
print(varlam)
ix_gm = np.loadtxt('ix_gm.txt')
ix_lam = np.loadtxt('ix_lam.txt')
#y = np.ones(len(var)) * sigma[op]
methods = []
nsuccess_gm = {}
xdmean_gm = {}
xsmean_gm = {}
nsuccess_lam = {}
xdmean_lam = {}
xsmean_lam = {}
for pt in perts:
    i = 0
    ns = 0
    nsuccess_gm[pt] = {}
    xdmean_gm[pt] = {}
    xsmean_gm[pt] = {}
    nsuccess_lam[pt] = {}
    xdmean_lam[pt] = {}
    xsmean_lam[pt] = {}
    for ivargm in vargm:
        j = 0
        nsuccess_gm[pt][ivargm] = {}
        xdmean_gm[pt][ivargm] = {}
        xsmean_gm[pt][ivargm] = {}
        nsuccess_lam[pt][ivargm] = {}
        xdmean_lam[pt][ivargm] = {}
        xsmean_lam[pt][ivargm] = {}
        for ivarlam in varlam:
            ivar=f"g{ivargm}l{ivarlam}"
            ## GM
            f = "{}_xdmean_gm_{}_{}_{}.txt".format(model, op, pt, ivar)
            if not os.path.isfile(f):
                print("not exist {}".format(f))
                xdmean_gm[pt][ivargm][ivarlam] = None
                nsuccess_gm[pt][ivargm][ivarlam] = 0
            else:
                data = np.loadtxt(f)
                e = data[:,1:]
                nsuccess_gm[pt][ivargm][ivarlam] = data[0,0]
                xdmean_gm[pt][ivargm][ivarlam] = e
                ns += 1
            f = "{}_xsmean_gm_{}_{}_{}.txt".format(model, op, pt, ivar)
            if not os.path.isfile(f):
                print("not exist {}".format(f))
                xsmean_gm[pt][ivargm][ivarlam] = None
            else:
                data = np.loadtxt(f)
                e = data[:,1:]
                xsmean_gm[pt][ivargm][ivarlam] = e
            ## LAM
            f = "{}_xdmean_lam_{}_{}_{}.txt".format(model, op, pt, ivar)
            if not os.path.isfile(f):
                print("not exist {}".format(f))
                xdmean_lam[pt][ivargm][ivarlam] = None
                nsuccess_lam[pt][ivargm][ivarlam] = 0
            else:
                data = np.loadtxt(f)
                e = data[:,1:]
                nsuccess_lam[pt][ivargm][ivarlam] = data[0,0]
                xdmean_lam[pt][ivargm][ivarlam] = e
            f = "{}_xsmean_lam_{}_{}_{}.txt".format(model, op, pt, ivar)
            if not os.path.isfile(f):
                print("not exist {}".format(f))
                xsmean_lam[pt][ivargm][ivarlam] = None
            else:
                data = np.loadtxt(f)
                e = data[:,1:]
                xsmean_lam[pt][ivargm][ivarlam] = e
    #ax.plot(x, e, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    if ns > 0:
        methods.append(pt)
i=0
for ivargm in vargm:
    ## GM
    nfigs = len(varlam)
    ncols = 2
    nrows = int(np.ceil(nfigs / ncols))
    figwidth = 10
    figheight = 3*nrows - 1
    fig, axs = plt.subplots(figsize=[figwidth,figheight],nrows=nrows,ncols=ncols,sharex=True,constrained_layout=True)
    ymax=0.0
    for ivarlam, ax in zip(varlam,axs.flatten()):
        for pt in methods:
            xd_gm = xdmean_gm[pt][ivargm][ivarlam]
            xs_gm = xsmean_gm[pt][ivargm][ivarlam]
            ns_gm = nsuccess_gm[pt][ivargm][ivarlam]
            if xd_gm is not None and xs_gm is not None:
                ax.plot(ix_gm,xd_gm[0,],lw=2.0,c=linecolor[pt],label=pt)
                ax.plot(ix_gm,xs_gm[0,],ls='dashed',c=linecolor[pt])
                ymax=max(ymax,np.max(xd_gm[0,]),np.max(xs_gm[0,]))
        ax.plot(ix_gm,np.ones(ix_gm.size)*sigma[op],c='k',ls='dotted')
        ax.vlines([ix_lam[0],ix_lam[-1]],0,1,colors='gray',alpha=0.5,transform=ax.get_xaxis_transform())
        ax.set_title(f"{ptype}={ivargm}:GM, {ivarlam}:LAM : #{int(ns_gm):d}")
    for i in range(nrows):
        axs[i,0].set_ylabel("RMSE or SPREAD")
    for ax in axs.flatten():
        ax.set_ylim(0.0,ymax)
    if len(axs.flatten())>len(varlam):
        axs[nrows-1,ncols-1].remove()
    axs[0,0].legend()
    fig.suptitle(op+", GM")
    fig.savefig("{}_xdmean_gm_{}_g{}_{}.png".format(model, ptype, ivargm, op))
    ## LAM
    fig, axs = plt.subplots(figsize=[figwidth,figheight],nrows=nrows,ncols=ncols,sharex=True,constrained_layout=True)
    ymax=0.0
    for ivarlam, ax in zip(varlam,axs.flatten()):
        for pt in methods:
            xd_lam = xdmean_lam[pt][ivargm][ivarlam]
            xs_lam = xsmean_lam[pt][ivargm][ivarlam]
            ns_lam = nsuccess_lam[pt][ivargm][ivarlam]
            if xd_lam is not None or xs_lam is not None:
                ax.plot(ix_lam,xd_lam[0,],lw=2.0,c=linecolor[pt],label=pt)
                ax.plot(ix_lam,xs_lam[0,],ls='dashed',c=linecolor[pt])
                ymax=max(ymax,np.max(xd_lam[0,]),np.max(xs_lam[0,]))
        ax.plot(ix_lam,np.ones(ix_lam.size)*sigma[op],c='k',ls='dotted')
        ax.set_title(f"{ptype}={ivargm}:GM, {ivarlam}:LAM : #{int(ns_lam):d}")
    for i in range(nrows):
        axs[i,0].set_ylabel("RMSE or SPREAD")
    for ax in axs.flatten():
        ax.set_ylim(0.0,ymax)
    if len(axs.flatten())>len(varlam):
        axs[nrows-1,ncols-1].remove()
    axs[0,0].legend()
    fig.suptitle(op+", LAM")
    fig.savefig("{}_xdmean_lam_{}_g{}_{}.png".format(model, ptype, ivargm, op))
