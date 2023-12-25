import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
dscl = False
if len(sys.argv)>4:
    dscl = (sys.argv[4]=='T')
print(dscl)
perts = ["mlef", "mlefw", "etkf", "po", "srf", "letkf", "kf", "var","var_nest",\
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
ix_t = np.loadtxt("ix_true.txt")
ix_gm = np.loadtxt("ix_gm.txt")
ix_lam = np.loadtxt("ix_lam.txt")
xt2x = interp1d(ix_t, xt)
xlim = 15.0
for pt in perts:
    #GM
    ## nature and analysis
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=[10,6],\
        constrained_layout=True,sharey=True)
    nx = ix_t.size
    mp0 = axs[0].pcolormesh(ix_t, t, xt, shading='auto',\
        cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[0].set_xticks(ix_t[::(nx//8)])
    axs[0].set_yticks(t[::max(1,na//8)])
    axs[0].set_xlabel("site")
    axs[0].set_ylabel("DA cycle")
    axs[0].set_title("nature")
    p0 = fig.colorbar(mp0,ax=axs[0],orientation="horizontal")
    if dscl:
        f = "xagmonly_{}_{}.npy".format(op, pt)
    else:
        f = "xagm_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xagm = np.load(f)
    print(xagm.shape)
    nx = ix_gm.size
    mp1 = axs[1].pcolormesh(ix_gm, t, xagm, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[1].set_xticks(ix_gm[::(nx//8)])
    axs[1].set_yticks(t[::max(1,na//8)])
    axs[1].set_xlabel("site")
    axs[1].set_title("analysis")
    p1 = fig.colorbar(mp1,ax=axs[1],orientation="horizontal")
    for ax in axs.flatten():
        ax.vlines([ix_lam[0],ix_lam[-1]],0,1,\
            colors='black',linestyle='dashdot',transform=ax.get_xaxis_transform())
    fig.suptitle("nature and analysis in GM : "+pt+" "+op)
    if dscl:
        fig.savefig("{}_xagmonly_{}_{}.png".format(model,op,pt))
    else:
        fig.savefig("{}_xagm_{}_{}.png".format(model,op,pt))
    plt.close()
    ## error and spread
    fig2 = plt.figure(figsize=[10,7],constrained_layout=True)
    gs0 = gridspec.GridSpec(1, 2, figure=fig2)
    gs00 = gs0[0].subgridspec(5, 1)
    ax00 = fig2.add_subplot(gs00[1:, :])
    ax01 = fig2.add_subplot(gs00[0, :])
    xt2gm = xt2x(ix_gm)
    xd = xagm - xt2gm
    vlim = np.nanmax(np.abs(xd))
    mp2 = ax00.pcolormesh(ix_gm, t, xd, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-vlim, vmax=vlim))
    ax00.set_xticks(ix_gm[::(nx//8)])
    ax00.set_yticks(t[::max(1,na//8)])
    ax00.set_xlabel("site")
    p2 = fig2.colorbar(mp2,ax=ax00,orientation="horizontal")
    ax01.plot(ix_gm,np.nanmean(np.abs(xd),axis=0))
    ax01.set_xlim(ix_gm[0],ix_gm[-1])
    ax01.set_xticks(ix_gm[::(nx//8)])
    ax01.set_xticklabels([])
    ax01.set_title("error")
    for ax in [ax00,ax01]:
        ax.vlines([ix_lam[0],ix_lam[-1]],0,1,\
            colors='black',linestyle='dashdot',transform=ax.get_xaxis_transform())
    if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
        if dscl:
            f = "xsagmonly_{}_{}.npy".format(op, pt)
        else:
            f = "xsagm_{}_{}.npy".format(op, pt)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        xsagm = np.load(f)
        print(xsagm.shape)
        gs01 = gs0[1].subgridspec(5, 1)
        ax10 = fig2.add_subplot(gs01[1:, :])
        ax11 = fig2.add_subplot(gs01[0, :])
        mp3 = ax10.pcolormesh(ix_gm, t, xsagm, shading='auto')
        ax10.set_xticks(ix_gm[::(nx//8)])
        ax10.set_yticks(t[::max(1,na//8)])
        ax10.set_xlabel("site")
        p3 = fig2.colorbar(mp3,ax=ax10,orientation="horizontal")
        ax11.plot(ix_gm,np.nanmean(xsagm,axis=0))
        ax11.set_xlim(ix_gm[0],ix_gm[-1])
        ax11.set_xticks(ix_gm[::(nx//8)])
        ax11.set_xticklabels([])
        ax11.set_title("spread")
        for ax in [ax10,ax11]:
            ax.vlines([ix_lam[0],ix_lam[-1]],0,1,\
            colors='black',linestyle='dashdot',transform=ax.get_xaxis_transform())
        fig2.suptitle("error and spread in GM : "+pt+" "+op)
    else:
        fig2.suptitle("error in GM : "+pt+" "+op)
    if dscl:
        fig2.savefig("{}_xdgmonly_{}_{}.png".format(model,op,pt))
    else:
        fig2.savefig("{}_xdgm_{}_{}.png".format(model,op,pt))
    plt.close()
    #LAM
    ## nature and analysis
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=[10,6],\
        constrained_layout=True,sharey=True)
    nx = ix_t.size
    mp0 = axs[0].pcolormesh(ix_t, t, xt, shading='auto',\
        cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[0].set_yticks(t[::max(1,na//8)])
    axs[0].set_xlabel("site")
    axs[0].set_ylabel("DA cycle")
    axs[0].set_title("nature")
    p0 = fig.colorbar(mp0,ax=axs[0],orientation="horizontal")
    if dscl:
        f = "xadscl_{}_{}.npy".format(op, pt)
    else:
        f = "xalam_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xalam = np.load(f)
    print(xalam.shape)
    nx = ix_lam.size
    axs[0].set_xticks(ix_lam[::(nx//8)])
    mp1 = axs[1].pcolormesh(ix_lam, t, xalam, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    p1 = fig.colorbar(mp1,ax=axs[1],orientation="horizontal")
    axs[1].set_xticks(ix_lam[::(nx//8)])
    axs[1].set_yticks(t[::max(1,na//8)])
    axs[1].set_xlabel("site")
    axs[1].set_title("analysis")
    for ax in axs.flatten():
        ax.set_xlim(ix_lam[0],ix_lam[-1])
    fig.suptitle("nature and analysis in LAM : "+pt+" "+op)
    if dscl:
        fig.savefig("{}_xadscl_{}_{}.png".format(model,op,pt))
    else:
        fig.savefig("{}_xalam_{}_{}.png".format(model,op,pt))
    plt.close()
    ## error and spread
    fig2 = plt.figure(figsize=[10,7],constrained_layout=True)
    gs0 = gridspec.GridSpec(1, 2, figure=fig2)
    gs00 = gs0[0].subgridspec(5, 1)
    ax00 = fig2.add_subplot(gs00[1:, :])
    ax01 = fig2.add_subplot(gs00[0, :])
    xt2lam = xt2x(ix_lam)
    xd = xalam - xt2lam
    vlim = np.nanmax(np.abs(xd))
    mp2 = ax00.pcolormesh(ix_lam, t, xd, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-vlim, vmax=vlim))
    ax00.set_xticks(ix_lam[::(nx//8)])
    ax00.set_yticks(t[::max(1,na//8)])
    ax00.set_xlabel("site")
    p2 = fig2.colorbar(mp2,ax=ax00,orientation="horizontal")
    ax01.plot(ix_lam,np.nanmean(np.abs(xd),axis=0))
    ax01.set_xticks(ix_lam[::(nx//8)])
    ax01.set_xticklabels([])
    ax01.set_title("error")
    for ax in [ax00,ax01]:
        ax.set_xlim(ix_lam[0],ix_lam[-1])
    if pt != "kf" and pt != "var" and pt != "4dvar":
        if dscl:
            f = "xsadscl_{}_{}.npy".format(op, pt)
        else:
            f = "xsalam_{}_{}.npy".format(op, pt)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        xsalam = np.load(f)
        print(xsalam.shape)
        gs01 = gs0[1].subgridspec(5, 1)
        ax10 = fig2.add_subplot(gs01[1:, :])
        ax11 = fig2.add_subplot(gs01[0, :])
        mp3 = ax10.pcolormesh(ix_lam, t, xsalam, shading='auto')
        ax10.set_xticks(ix_lam[::(nx//8)])
        ax10.set_yticks(t[::max(1,na//8)])
        ax10.set_xlabel("site")
        p3 = fig2.colorbar(mp3,ax=ax10,orientation="horizontal")
        ax11.plot(ix_lam,np.nanmean(xsalam,axis=0))
        ax11.set_xticks(ix_lam[::(nx//8)])
        ax11.set_xticklabels([])
        ax11.set_title("spread")
        for ax in [ax10,ax11]:
            ax.set_xlim(ix_lam[0],ix_lam[-1])
        fig2.suptitle("error and spread in LAM : "+pt+" "+op)
    else:
        fig2.suptitle("error in LAM : "+pt+" "+op)
    if dscl:
        fig2.savefig("{}_xddscl_{}_{}.png".format(model,op,pt))
    else:
        fig2.savefig("{}_xdlam_{}_{}.png".format(model,op,pt))