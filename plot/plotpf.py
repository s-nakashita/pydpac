import sys
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
plt.rcParams["font.size"] = 12
op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = sys.argv[4]
scycle = int(sys.argv[5])
ecycle = int(sys.argv[6])
if model == "z08":
    nx = 81
elif model == "z05":
    nx = 101
elif model == "l96":
    nx = 40
#ix = np.arange(nx)
ix = np.loadtxt("ix.txt")
nx = ix.size
ncycle = 0
pf = np.zeros((nx,nx))
for icycle in range(scycle,ecycle+1):
    cmap = "coolwarm"
    f = "{}_pf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if os.path.isfile(f):
        tmp = np.load(f)
        pf = pf + tmp
        ncycle += 1
#    pa = None  
#    f = "{}_pa_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
#    if os.path.isfile(f):
#        pa = np.load(f)
pf = pf / float(ncycle)
fig, ax = plt.subplots(figsize=[6,6],constrained_layout=True)
plot = False
if ncycle > 0:
        plot = True
        ymax = np.max(pf)
        ymin = np.min(pf)
        ylim = max(ymax, np.abs(ymin))
        mappable=ax.pcolormesh(ix,ix,pf,shading='auto',\
            cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
        ax.set_aspect("equal")
        ax.set_xticks(ix[::(nx//8)])
        ax.set_yticks(ix[::(nx//8)])
        ax.set_title("Pf")
        fig.colorbar(mappable, ax=ax, pad=0.01, shrink=0.6) #orientation="horizontal")
        fig.savefig("{}_pf_{}_{}_cycle{}-{}.png".format(model,op,pt,scycle,ecycle))
