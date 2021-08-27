import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
plt.rcParams['savefig.dpi'] = 300

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = sys.argv[4]
if model == "z08" or model == "z05":
    nx = 81
elif model == "l96":
    nx = 40
x = np.arange(nx) 
fig, ax = plt.subplots()
cmap = "Reds"
f = "{}_rho_{}_{}.npy".format(model, op, pt)
if os.path.isfile(f):
    rho = np.load(f)
    ymax = np.max(rho)
    ymin = np.min(rho)
    print("rho max={} min={}".format(ymax,ymin))
    mappable=ax.matshow(rho,cmap=cmap,norm=Normalize(vmin=ymin, vmax=ymax))
    ax.set_aspect("equal")
    ax.set_xticks(x[::5])
    ax.set_yticks(x[::5])
    ax.invert_yaxis()
    ax.set_title(u"\u03c1")
    fig.colorbar(mappable, ax=ax,orientation="vertical")
    fig.savefig("{}_rho_{}_{}.png".format(model,op,pt))

for icycle in range(5):
#icycle = 0
    fig, ax = plt.subplots(2,2)
    cmap = "coolwarm"
    pf = None
    f = "{}_pf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if os.path.isfile(f):
        pf = np.load(f)

    lpf = None        
    f = "{}_lpf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if os.path.isfile(f):
        lpf = np.load(f)

    spf = None
    f = "{}_spf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if os.path.isfile(f):
        spf = np.load(f)

    lspf = None 
    f = "{}_lspf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if os.path.isfile(f):
        lspf = np.load(f)

    plot = False
    if pf is not None:
        plot = True
        ymax = np.max(pf)
        ymin = np.min(pf)
        ylim = max(ymax, np.abs(ymin))
        mappable=ax[0, 0].matshow(pf,cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
        ax[0, 0].set_aspect("equal")
        ax[0, 0].set_xticks(x[::5])
        ax[0, 0].set_yticks(x[::5])
        #ax[0, 0].invert_xaxis()
        ax[0, 0].invert_yaxis()
        ax[0, 0].set_title("Pf")
        fig.colorbar(mappable, ax=ax[0,0],orientation="vertical")
    else:
        ax[0, 0].remove()
    if lpf is not None:
        plot = True
        ymax = np.max(lpf)
        ymin = np.min(lpf)
        ylim = max(ymax, np.abs(ymin))
        mappable=ax[1, 0].matshow(lpf,cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
        ax[1, 0].set_aspect("equal")
        ax[1, 0].set_xticks(x[::5])
        ax[1, 0].set_yticks(x[::5])
        #ax[1, 0].invert_xaxis()
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_title("localized Pf")
        fig.colorbar(mappable, ax=ax[1,0],orientation="vertical")
    else:
        ax[1, 0].remove()
    if spf is not None:
        plot = True
        nmem = spf.shape[1]
        y = np.arange(nmem) 
        ymax = np.max(spf)
        ymin = np.min(spf)
        ylim = max(ymax, np.abs(ymin))
        mappable=ax[0, 1].matshow(spf,cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
        #ax[0, 1].set_aspect(1.2)
        ax[0, 1].set_yticks(x[::5])
        ax[0, 1].set_xticks(y[::5])
        ax[0, 1].invert_yaxis()
        if pt[0:4] == "mlef":
            ax[0, 1].set_title("sqrtPf")
        else:
            ax[0, 1].set_title("dXf")
        fig.colorbar(mappable, ax=ax[0,1],orientation="vertical")
    else:
        ax[0, 1].remove()
    if lspf is not None:
        plot = True
        nmem = lspf.shape[1]
        y = np.arange(nmem) 
        ymax = np.max(lspf)
        ymin = np.min(lspf)
        ylim = max(ymax, np.abs(ymin))
        mappable=ax[1, 1].matshow(lspf,cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
        #ax[1, 1].set_aspect(1.2)
        #ax[1, 1].set_yticks(x[::10])
        #ax[1, 1].set_xticks(y[::10])
        ax[1, 1].invert_yaxis()
        if pt[0:4] == "mlef":
            ax[1, 1].set_title("localized sqrtPf")
        else:
            ax[1, 1].set_title("localized dXf")
        fig.colorbar(mappable, ax=ax[1,1],orientation="vertical")
    else:
        ax[1, 1].remove()
    #axpos = ax[0, 1].get_position()
    #cbar_ax = fig.add_axes([0.90, axpos.y0/4, 0.02, 2*axpos.height])
    #mappable = ScalarMappable(cmap=cmap)
    #fig.colorbar(mappable, cax=cbar_ax)
    if plot:
        fig.tight_layout()
        fig.savefig("{}_lpf_{}_{}_cycle{}.png".format(model,op,pt,icycle))
