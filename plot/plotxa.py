import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
plt.rcParams['font.size'] = 16
from methods import perts

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
cmap = "coolwarm"
f = "truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[:na,]
print(xt.shape)
nx_t = xt.shape[1]
t = np.arange(na)
xs_t = np.loadtxt('ix_t.txt')
xt2mod = interp1d(xs_t,xt,axis=1)
xs = np.loadtxt('ix.txt')
if model=='kdvb' or model=='burgers':
    if model == "burgers":
        sigma = {"linear": 8.0e-2, "quadratic": 1.0e-3, "cubic": 1.0e-3, "quartic": 1.0e-2, \
        "quadratic-nodiff": 1.0e-3, "cubic-nodiff": 1.0e-3, "quartic-nodiff": 1.0e-2}
    elif model == "kdvb":
        sigma = {"linear": 0.05, "quadratic": 0.05}
    sig = sigma[op]
    obsfile=f'obs_{op}_{int(sig*1e4)}.npy'
    yobs = np.load(obsfile)
    # plot curves
    xlim = 1.0
    plot_interval = na // 8
    colors = plt.get_cmap('plasma')(np.linspace(0,1,11))
    for pt in perts:
        ## nature and analysis
        f = "xa_{}_{}.npy".format(op, pt)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        xa = np.load(f)
        print(xa.shape)
        nx = xa.shape[1]
        fig, axs = plt.subplots(figsize=[8,6],nrows=2,sharex=True,constrained_layout=True)
        lines = []
        labels = []
        lines.append(Line2D([0],[0],lw=3.0,c=colors[0],alpha=0.3))
        labels.append('truth')
        lines.append(Line2D([0],[0],c=colors[0]))
        labels.append('analysis')
        icol = 0
        for j in range(0,na,plot_interval):
            axs[0].plot(xs_t, xt[j],color=colors[icol],lw=3.0,alpha=0.3)
            axs[0].plot(xs,xa[j],color=colors[icol])
            lines.append(Line2D([0],[0],c=colors[icol]))
            labels.append(f'c{j:02d}')
            axs[0].plot(yobs[j,:,0],yobs[j,:,1],lw=0.0,marker='X',c=colors[icol])
            icol += 1
        fig.legend(lines,labels,ncol=2,loc='upper right')
        ## error and spread
        xd = xa - xt2mod(xs)
        f = "xsa_{}_{}.npy".format(op, pt)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        xsa = np.load(f)
        print(xsa.shape)
        lines = []
        labels = []
        lines.append(Line2D([0],[0],c=colors[0],ls='dashed'))
        labels.append('error')
        lines.append(Line2D([0],[0],c=colors[0],ls='dotted'))
        labels.append('spread')
        icol = 0
        for j in range(0,na,plot_interval):
            axs[1].plot(xs, xd[j],color=colors[icol],ls='dashed')
            axs[1].plot(xs,xsa[j],color=colors[icol],ls='dotted')
            icol += 1
        axs[1].legend(lines,labels)
        fig.suptitle(pt+" "+op)
        fig.savefig("{}_xa_{}_{}.png".format(model,op,pt))
        plt.close()
else:
    # plot hövmeller
    xlim = 15.0
    for pt in perts:
        ## nature and analysis
        f = "xa_{}_{}.npy".format(op, pt)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        xa = np.load(f)
        print(xa.shape)
        nx = xa.shape[1]
        intmod = nx_t // nx
        xs = np.arange(0,nx_t,intmod)
        fig, axs = plt.subplots(nrows=1,ncols=2,figsize=[10,6],constrained_layout=True,sharey=True)
        mp0 = axs[0].pcolormesh(xs_t, t, xt, shading='auto',\
            cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
        axs[0].set_xticks(xs_t[::(nx_t//8)])
        axs[0].set_xlabel("site")
        axs[0].set_ylabel("DA cycle")
        axs[0].set_title("nature")
        p0 = fig.colorbar(mp0,ax=axs[0],orientation="horizontal")
        mp1 = axs[1].pcolormesh(xs, t, xa, shading='auto', \
        cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
        axs[1].set_title("analysis")
        axs[1].set_xticks(xs[::(nx//8)])
        axs[1].set_xlabel("site")
        for ax in axs:
            ax.set_yticks(t[::(na//8)])
            ax.set_ylim(t[-1],t[0])
        p1 = fig.colorbar(mp1,ax=axs[1],orientation="horizontal")
        fig.suptitle("nature, analysis : "+pt+" "+op)
        fig.savefig("{}_xa_{}_{}.png".format(model,op,pt))
        plt.close()
        ## error and spread
        fig2 = plt.figure(figsize=[10,7],constrained_layout=True)
        gs0 = gridspec.GridSpec(1,2,figure=fig2)
        gs00 = gs0[0].subgridspec(5,1)
        ax00 = fig2.add_subplot(gs00[1:,:])
        ax01 = fig2.add_subplot(gs00[0,:])
        xd = xa - xt2mod(xs)
        vlim = np.max(np.abs(xd))
        mp2 = ax00.pcolormesh(xs, t, xd, shading='auto', \
        cmap=cmap, norm=Normalize(vmin=-vlim, vmax=vlim))
        ax00.set_xticks(xs[::(nx//8)])
        ax00.set_yticks(t[::(na//8)])
        ax00.set_ylim(t[-1],t[0])
        ax00.set_xlabel("site")
        p2 = fig.colorbar(mp2,ax=ax00,orientation="horizontal")
        ax01.plot(xs, np.abs(xd).mean(axis=0))
        ax01.set_xticks(xs[::(nx//8)])
        ax01.set_xticklabels([])
        ax01.set_title("error")
        for ax in [ax00,ax01]:
            ax.set_xlim(xs[0],xs[-1])
    #    if pt != "kf" and pt != "var" and pt != "4dvar":
        f = "xsa_{}_{}.npy".format(op, pt)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        xsa = np.load(f)
        print(xsa.shape)
        gs01 = gs0[1].subgridspec(5, 1)
        ax10 = fig2.add_subplot(gs01[1:, :])
        ax11 = fig2.add_subplot(gs01[0, :])
        mp3 = ax10.pcolormesh(xs, t, xsa, shading='auto')
        ax10.set_xticks(xs[::(nx//8)])
        ax10.set_yticks(t[::max(1,na//8)])
        ax10.set_ylim(t[-1],t[0])
        ax10.set_xlabel("site")
        p3 = fig2.colorbar(mp3,ax=ax10,orientation="horizontal")
        ax11.plot(xs,xsa.mean(axis=0))
        ax11.set_xticks(xs[::(nx//8)])
        ax11.set_xticklabels([])
        for ax in [ax10,ax11]:
            ax.set_xlim(xs[0],xs[-1])
        if pt != "kf" and pt != "var" and pt != "4dvar":
            ax11.set_title("spread")
            fig2.suptitle("error and spread : "+pt+" "+op)
        else:
            ax11.set_title("analysis error standard deviation")
            fig2.suptitle("error and stdv : "+pt+" "+op)
        fig2.savefig("{}_xd_{}_{}.png".format(model,op,pt))
