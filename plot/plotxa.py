import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "etkf", "po", "srf", "letkf", "kf", "var"]
if model == "z08":
    perts = ["mlef", "grad", "etkf-fh", "etkf-jh"]#, "po", "srf", "letkf"]
    linestyle = {"mlef":"solid", "grad":"dashed",
     "etkf-fh":"solid", "etkf-jh":"dashed"}
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf-fh":'tab:green',"etkf-jh":'tab:red'}
cmap = "coolwarm"
f = "truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[:na,]
print(xt.shape)
nx = xt.shape[1]
t = np.arange(na)
xs = np.arange(nx)
xlim = 15.0
for pt in perts:
    ## nature and analysis
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=[10,6],constrained_layout=True,sharey=True)
    mp0 = axs[0].pcolormesh(xs, t, xt, shading='auto',\
        cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[0].set_xticks(xs[::(nx//8)])
    axs[0].set_yticks(t[::(na//8)])
    axs[0].set_xlabel("site")
    axs[0].set_ylabel("DA cycle")
    axs[0].set_title("nature")
    p0 = fig.colorbar(mp0,ax=axs[0],orientation="horizontal")
    f = "{}_xa_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xa = np.load(f)
    print(xa.shape)
    mp1 = axs[1].pcolormesh(xs, t, xa, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[1].set_xticks(xs[::(nx//8)])
    axs[1].set_yticks(t[::(na//8)])
    axs[1].set_xlabel("site")
    axs[1].set_title("analysis")
    p1 = fig.colorbar(mp1,ax=axs[1],orientation="horizontal")
    fig.suptitle("nature, analysis : "+pt+" "+op)
    fig.savefig("{}_xa_{}_{}.png".format(model,op,pt))
    plt.close()
    ## error and spread
    fig2 = plt.figure(figsize=[10,7],constrained_layout=True)
    gs0 = gridspec.GridSpec(1,2,figure=fig2)
    gs00 = gs0[0].subgridspec(5,1)
    ax00 = fig2.add_subplot(gs00[1:,:])
    ax01 = fig2.add_subplot(gs01[0,:])
    xd = xa - xt
    vlim = np.max(np.abs(xd))
    mp2 = ax00.pcolormesh(xs, t, xd, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-vlim, vmax=vlim))
    ax00.set_xticks(xs[::(nx//8)])
    ax00.set_yticks(t[::(na//8)])
    ax00.set_xlabel("site")
    p2 = fig.colorbar(mp2,ax=ax00,orientation="horizontal")
    ax01.plot(xs, np.abs(xd).mean(axis=0))
    ax01.set_xticks(xs[::(nx//8)])
    ax01.set_xticklabels([])
    ax01.set_title("error")
    f = "{}_xsa_{}_{}.npy".format(model, op, pt)
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
    ax10.set_xlabel("site")
    p3 = fig2.colorbar(mp3,ax=ax10,orientation="horizontal")
    ax11.plot(xs,xsa.mean(axis=0))
    ax11.set_xticks(xs[::(nx//8)])
    ax11.set_xticklabels([])
    ax11.set_title("spread")
    for ax in [ax00,ax01,ax10,ax11]:
        ax.set_xlim(xs[0],xs[-1])
    fig2.suptitle("error and spread : "+pt+" "+op)
    fig2.savefig("{}_xd_{}_{}.png".format(model,op,pt))