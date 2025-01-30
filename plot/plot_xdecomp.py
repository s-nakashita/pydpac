import sys
import os
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
sys.path.append(os.path.join(os.path.dirname(__file__),'../analysis'))
from trunc1d import Trunc1d
from methods import perts, linecolor

plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
preGM = False
if len(sys.argv)>4:
    preGM = (sys.argv[4]=='T')

cmap = "coolwarm"
f = "truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[:na,]
print(xt.shape)
t = np.arange(na)
ix_t = np.loadtxt('ix_true.txt')
nx_t = ix_t.size
ix_t_rad = ix_t * 2.0 * np.pi / nx_t
xt2x = interp1d(ix_t,xt)
ix_gm = np.loadtxt('ix_gm.txt')
nx_gm = ix_gm.size
ix_gm_rad = ix_gm * 2.0 * np.pi / nx_t
ix_lam = np.loadtxt('ix_lam.txt')
nx_lam = ix_lam.size
ix_lam_rad = ix_lam * 2.0 * np.pi / nx_t

truncope_t = Trunc1d(ix_t_rad,ttype='c',resample=False)
truncope_g = Trunc1d(ix_gm_rad,ttype='c',resample=False)
truncope_l = Trunc1d(ix_lam_rad,ttype='c',resample=False)

kthres = [6.,30.]
ncols = len(kthres) + 2
fig, axs = plt.subplots(ncols=ncols,figsize=[12,6],sharey=True,constrained_layout=True)
mp0 = axs[0].pcolormesh(ix_t,t,xt,shading='auto',\
    cmap=cmap,norm=Normalize(-15,15))
axs[0].set_xticks(ix_t[::(nx_t//6)])
axs[0].set_yticks(t[::(na//8)])
axs[0].set_xlabel("site")
axs[0].set_ylabel("DA cycle")
axs[0].set_title("full")
p0 = fig.colorbar(mp0, ax=axs[0], orientation='vertical',shrink=0.6,pad=0.01)
_, xtdecomp = truncope_t.scale_decomp(xt.T,kthres=kthres)
for i,xd in enumerate(xtdecomp):
    mp1 = axs[i+1].pcolormesh(ix_t,t,xd.T,shading='auto',\
    cmap=cmap,norm=Normalize(-15,15))
    axs[i+1].set_xticks(ix_t[::(nx_t//6)])
    axs[i+1].set_xlabel("site")
    #axs[i+1].set_yticks(t[::(na//8)])
    #axs[i+1].set_ylabel("DA cycle")
axs[1].set_title(r"$k \leq$"+f"{kthres[0]}")
axs[2].set_title(f"{kthres[0]}"+r"$\leq k \leq$"+f"{kthres[1]}")
axs[3].set_title(f"{kthres[1]}"+r"$\leq k$")
fig.suptitle("nature")
fig.savefig("{}_nature_decomp.png".format(model),dpi=300)
#plt.show()
plt.close()

errgm_decomp = {}
errlam_decomp = {}
for pt in perts:
    if not preGM:
        #GM
        f = "xagm_{}_{}.npy".format(op, pt)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        xagm = np.load(f)
        
        fig, axs = plt.subplots(ncols=ncols,figsize=[12,6],sharey=True,constrained_layout=True)
        mp0 = axs[0].pcolormesh(ix_gm,t,xagm,shading='auto',\
            cmap=cmap,norm=Normalize(-15,15))
        axs[0].set_xticks(ix_gm[::(nx_gm//6)])
        axs[0].set_yticks(t[::(na//8)])
        axs[0].set_xlabel("site")
        axs[0].set_ylabel("DA cycle")
        axs[0].set_title("full")
        p0 = fig.colorbar(mp0, ax=axs[0], orientation='vertical',shrink=0.6,pad=0.01)
        _, xgdecomp = truncope_g.scale_decomp(xagm.T,kthres=kthres)
        for i,xd in enumerate(xgdecomp):
            mp1 = axs[i+1].pcolormesh(ix_gm,t,xd.T,shading='auto',\
            cmap=cmap,norm=Normalize(-15,15))
            axs[i+1].set_xticks(ix_gm[::(nx_gm//6)])
            axs[i+1].set_xlabel("site")
            #axs[i+1].set_yticks(t[::(na//8)])
            #axs[i+1].set_ylabel("DA cycle")
        axs[1].set_title(r"$k \leq$"+f"{kthres[0]}")
        axs[2].set_title(f"{kthres[0]}"+r"$\leq k \leq$"+f"{kthres[1]}")
        axs[3].set_title(f"{kthres[1]}"+r"$\leq k$")
        fig.suptitle(f"GM, {op}, {pt}")
        fig.savefig("{}_xagm_decomp_{}_{}.png".format(model,op,pt),dpi=300)
        #plt.show()
        plt.close()

        ##diff
        xd = xagm - xt2x(ix_gm)
        #vlim = max(xd.max(),-xd.min())
        vlim = 5.0
        fig, axs = plt.subplots(ncols=ncols,figsize=[12,6],sharey=True,constrained_layout=True)
        mp0 = axs[0].pcolormesh(ix_gm,t,xd,shading='auto',\
            cmap=cmap,norm=Normalize(-vlim,vlim))
        axs[0].set_xticks(ix_gm[::(nx_gm//6)])
        axs[0].set_yticks(t[::(na//8)])
        axs[0].set_xlabel("site")
        axs[0].set_ylabel("DA cycle")
        axs[0].set_title("full")
        p0 = fig.colorbar(mp0, ax=axs[0], orientation='vertical',shrink=0.6,pad=0.01)
        #xddecomp = scale_decomp(xd,ix=ix_gm_rad, kthres=kthres)
        err = []
        for i, xgd in enumerate(xgdecomp):
            xtd = xtdecomp[i].T
            xtd2x = interp1d(ix_t,xtd)
            xd = xgd.T - xtd2x(ix_gm)
            mp1 = axs[i+1].pcolormesh(ix_gm,t,xd,shading='auto',\
            cmap=cmap,norm=Normalize(-vlim,vlim))
            axs[i+1].set_xticks(ix_gm[::(nx_gm//6)])
            axs[i+1].set_xlabel("site")
            err.append(np.sqrt(np.mean(xd**2,axis=1)))
        axs[1].set_title(r"$k \leq$"+f"{kthres[0]}")
        axs[2].set_title(f"{kthres[0]}"+r"$\leq k \leq$"+f"{kthres[1]}")
        axs[3].set_title(f"{kthres[1]}"+r"$\leq k$")
        fig.suptitle(f"GM error, {op}, {pt}")
        fig.savefig("{}_xdgm_decomp_{}_{}.png".format(model,op,pt),dpi=300)
        #plt.show()
        plt.close()
        errgm_decomp[pt] = err

    #LAM
    f = "xalam_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xalam = np.load(f)
    
    fig, axs = plt.subplots(ncols=ncols,figsize=[12,6],sharey=True,constrained_layout=True)
    mp0 = axs[0].pcolormesh(ix_lam,t,xalam,shading='auto',\
        cmap=cmap,norm=Normalize(-15,15))
    axs[0].set_xticks(ix_lam[::(nx_lam//6)])
    axs[0].set_yticks(t[::(na//8)])
    axs[0].set_xlabel("site")
    axs[0].set_ylabel("DA cycle")
    axs[0].set_title("full")
    p0 = fig.colorbar(mp0, ax=axs[0], orientation='vertical',shrink=0.6,pad=0.01)
    _, xldecomp = truncope_l.scale_decomp(xalam.T,kthres=kthres)
    for i,xd in enumerate(xldecomp):
        mp1 = axs[i+1].pcolormesh(ix_lam,t,xd.T,shading='auto',\
        cmap=cmap,norm=Normalize(-15,15))
        axs[i+1].set_xticks(ix_lam[::(nx_lam//6)])
        axs[i+1].set_xlabel("site")
        #axs[i+1].set_yticks(t[::(na//8)])
        #axs[i+1].set_ylabel("DA cycle")
    axs[1].set_title(r"$k \leq$"+f"{kthres[0]}")
    axs[2].set_title(f"{kthres[0]}"+r"$\leq k \leq$"+f"{kthres[1]}")
    axs[3].set_title(f"{kthres[1]}"+r"$\leq k$")
    fig.suptitle(f"LAM, {op}, {pt}")
    fig.savefig("{}_xalam_decomp_{}_{}.png".format(model,op,pt),dpi=300)
    plt.show()
    plt.close()

    ##diff
    xd = xalam - xt2x(ix_lam)
    #vlim = max(xd.max(),-xd.min())
    vlim = 5.0
    fig, axs = plt.subplots(ncols=ncols,figsize=[12,6],sharey=True,constrained_layout=True)
    mp0 = axs[0].pcolormesh(ix_lam,t,xd,shading='auto',\
        cmap=cmap,norm=Normalize(-vlim,vlim))
    axs[0].set_xticks(ix_lam[::(nx_lam//6)])
    axs[0].set_yticks(t[::(na//8)])
    axs[0].set_xlabel("site")
    axs[0].set_ylabel("DA cycle")
    axs[0].set_title("full")
    p0 = fig.colorbar(mp0, ax=axs[0], orientation='vertical',shrink=0.6,pad=0.01)
    #xddecomp = scale_decomp(xd,ix=ix_gm_rad, kthres=kthres)
    err = []
    for i, xld in enumerate(xldecomp):
        xtd = xtdecomp[i].T
        xtd2x = interp1d(ix_t,xtd)
        xd = xld.T - xtd2x(ix_lam)
        mp1 = axs[i+1].pcolormesh(ix_lam,t,xd,shading='auto',\
        cmap=cmap,norm=Normalize(-vlim,vlim))
        axs[i+1].set_xticks(ix_lam[::(nx_lam//6)])
        axs[i+1].set_xlabel("site")
        err.append(np.sqrt(np.mean(xd**2,axis=1)))
    axs[1].set_title(r"$k \leq$"+f"{kthres[0]}")
    axs[2].set_title(f"{kthres[0]}"+r"$\leq k \leq$"+f"{kthres[1]}")
    axs[3].set_title(f"{kthres[1]}"+r"$\leq k$")
    fig.suptitle(f"LAM error, {op}, {pt}")
    fig.savefig("{}_xdlam_decomp_{}_{}.png".format(model,op,pt),dpi=300)
    plt.show()
    plt.close()
    errlam_decomp[pt] = err

if not preGM:
    figgm, axsgm = plt.subplots(nrows=len(kthres)+1,figsize=[8,8],sharex=True,constrained_layout=True)
    for pt in errgm_decomp.keys():
        err = errgm_decomp[pt]
        for i, e in enumerate(err):
            emean = e.mean()
            axsgm[i].plot(t,e,c=linecolor[pt],label=pt+f'={emean:.3f}')
    axsgm[0].set_title(r"$k \leq$"+f"{kthres[0]}")
    axsgm[1].set_title(f"{kthres[0]}"+r"$\leq k \leq$"+f"{kthres[1]}")
    axsgm[2].set_title(f"{kthres[1]}"+r"$\leq k$")
    axsgm[2].set_xlabel("DA cycle")
    for ax in axsgm:
        ax.legend(loc='upper left',bbox_to_anchor=(1.01,0.95))
    figgm.suptitle(f"GM error, {op}")
    figgm.savefig("{}_xdgm_decomp_1d_{}.png".format(model,op),dpi=300)
    plt.show()
    plt.close()

figlam, axslam = plt.subplots(nrows=len(kthres)+1,figsize=[8,8],sharex=True,constrained_layout=True)
for pt in errlam_decomp.keys():
    err = errlam_decomp[pt]
    for i, e in enumerate(err):
        emean = e.mean()
        axslam[i].plot(t,e,c=linecolor[pt],label=pt+f'={emean:.3f}')
axslam[0].set_title(r"$k \leq$"+f"{kthres[0]}")
axslam[1].set_title(f"{kthres[0]}"+r"$\leq k \leq$"+f"{kthres[1]}")
axslam[2].set_title(f"{kthres[1]}"+r"$\leq k$")
axslam[2].set_xlabel("DA cycle")
for ax in axslam:
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,0.95))
figlam.suptitle(f"LAM error, {op}")
figlam.savefig("{}_xdlam_decomp_1d_{}.png".format(model,op),dpi=300)
plt.show()
plt.close()
