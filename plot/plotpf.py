import sys
import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = sys.argv[4]
if model == "z08" or model == "z05":
    nx = 81
elif model == "l96":
    nx = 40
x = np.arange(nx)
for icycle in range(5):
    cmap = "coolwarm"
    pf = None
    f = "{}_pf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if os.path.isfile(f):
        pf = np.load(f)
    pa = None  
    f = "{}_pa_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if os.path.isfile(f):
        pa = np.load(f)
    
    fig, ax = plt.subplots(nrows=1,ncols=2)
    plot = False
    if pf is not None:
        plot = True
        ymax = np.max(pf)
        ymin = np.min(pf)
        ylim = max(ymax, np.abs(ymin))
        mappable=ax[0].matshow(pf,cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
        ax[0].set_aspect("equal")
        ax[0].set_xticks(x[::5])
        ax[0].set_yticks(x[::5])
        #ax[0, 0].invert_xaxis()
        ax[0].invert_yaxis()
        ax[0].set_title("Pf")
        fig.colorbar(mappable, ax=ax[0],orientation="horizontal")
    if pa is not None:
        plot = True
        ymax = np.max(pa)
        ymin = np.min(pa)
        ylim = max(ymax, np.abs(ymin))
        mappable=ax[1].matshow(pa,cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
        ax[1].set_aspect("equal")
        ax[1].set_xticks(x[::5])
        ax[1].set_yticks(x[::5])
        #ax[1, 0].invert_xaxis()
        ax[1].invert_yaxis()
        ax[1].set_title("Pa")
        fig.colorbar(mappable, ax=ax[1],orientation="horizontal")
    if plot:
        fig.tight_layout()
        fig.savefig("{}_cov_{}_{}_cycle{}.png".format(model,op,pt,icycle))
