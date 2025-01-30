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
model_error = False
if len(sys.argv)>4:
    model_error = (sys.argv[4]=='True')
cmap = "coolwarm"
f = "truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[:na,]
print(xt.shape)
nx_t = xt.shape[1]
t = np.arange(na)
xs_t = np.arange(nx_t)
xt2mod = interp1d(xs_t,xt,axis=1)
xlim = 15.0
for pt in perts:
    ## nature and analysis
    f = "{}_xa_{}_{}.npy".format(model, op, pt)
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
    axs[0].set_yticks(t[::(na//8)])
    axs[0].set_xlabel("site")
    axs[0].set_ylabel("DA cycle")
    axs[0].set_title("nature")
    p0 = fig.colorbar(mp0,ax=axs[0],orientation="horizontal")
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
    ax01 = fig2.add_subplot(gs00[0,:])
    xd = xa - xt2mod(xs)
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
    for ax in [ax00,ax01]:
        ax.set_xlim(xs[0],xs[-1])
#    if pt != "kf" and pt != "var" and pt != "4dvar":
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
    for ax in [ax10,ax11]:
        ax.set_xlim(xs[0],xs[-1])
    if pt != "kf" and pt != "var" and pt != "4dvar":
        ax11.set_title("spread")
        fig2.suptitle("error and spread : "+pt+" "+op)
    else:
        ax11.set_title("analysis error standard deviation")
        fig2.suptitle("error and stdv : "+pt+" "+op)
    fig2.savefig("{}_xd_{}_{}.png".format(model,op,pt))
