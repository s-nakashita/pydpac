import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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
t = np.arange(na+1)
xs = np.arange(xt.shape[1]+1)
for pt in perts:
    fig, ax = plt.subplots()
    #mappable0 = ax[0].pcolor(xs, t, xt, cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    #ax[0].set_xticks(xs[::10])
    #ax[0].set_yticks(t[::50])
    #ax[0].set_xlabel("site")
    #ax[0].set_ylabel("DA cycle")
    #ax[0].set_title("truth")
    #pp = fig.colorbar(mappable0,ax=ax[0],orientation="horizontal")
    f = "{}_ua_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xa = np.load(f)
    print(xa.shape)
    xd = xa - xt
    #xmax = np.max(xd)*1.01
    #xmin = np.min(xd)*1.01
    #xlim = max(np.abs(xmin),xmax)
    xlim = 15.0
    mappable0 = ax.pcolor(xs, t, xd, cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    ax.set_xticks(xs[::10])
    ax.set_yticks(t[::50])
    ax.set_xlabel("site")
    ax.set_ylabel("DA cycle")
    ax.set_title("analysis error "+pt+" "+op)
    pp = fig.colorbar(mappable0,ax=ax,orientation="vertical")
    fig.savefig("{}_xa_{}_{}.png".format(model,op,pt))