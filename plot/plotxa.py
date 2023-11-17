import sys
import os
import matplotlib.pyplot as plt
import numpy as np
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
xt = np.load(f)
print(xt.shape)
nx = xt.shape[1]
t = np.arange(na)
xs = np.arange(nx)
xlim = 15.0
for pt in perts:
    fig, axs = plt.subplots(ncols=3,figsize=[12,6],constrained_layout=True,sharey=True)
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
    xd = xa - xt
    mp1 = axs[1].pcolormesh(xs, t, xa, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[1].set_xticks(xs[::(nx//8)])
    axs[1].set_yticks(t[::(na//8)])
    axs[1].set_xlabel("site")
    axs[1].set_title("analysis")
    vlim = np.max(np.abs(xd))
    p1 = fig.colorbar(mp1,ax=axs[1],orientation="horizontal")
    mp2 = axs[2].pcolormesh(xs, t, xd, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[2].set_xticks(xs[::(nx//8)])
    axs[2].set_yticks(t[::(na//8)])
    axs[2].set_xlabel("site")
    axs[2].set_title("error")
    p2 = fig.colorbar(mp2,ax=axs[2],orientation="horizontal")
    fig.suptitle("nature, analysis, error "+pt+" "+op)
    fig.savefig("{}_xa_{}_{}.png".format(model,op,pt))