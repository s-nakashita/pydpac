import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
if model == "z08" or model == "z05":
    nx = 81
    perts = ["mlef", "grad", "etkf-jh", "etkf-fh"]
    cycle = [1, 2, 3, 4]
    #na = 20
elif model == "l96":
    nx = 40
    perts = ["mlef"]
    #na = 300
x = np.arange(nx+1) + 1
for pt in perts:
    fig, ax = plt.subplots()
    cmap = "Reds"
    f = "{}_rho_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    rho = np.load(f)
    ymax = np.max(rho)
    ymin = np.min(rho)
    mappable=ax.pcolor(x,x,rho,cmap=cmap,norm=Normalize(vmin=ymin, vmax=ymax))
    ax.set_aspect("equal")
    ax.set_xticks(x[::5])
    ax.set_yticks(x[::5])
    ax.set_title(u"\u03c1")
    fig.colorbar(mappable, ax=ax,orientation="vertical")
    fig.savefig("{}_rho_{}_{}.png".format(model,op,pt))

    for icycle in [0, 50, 100, 150]:
        fig, ax = plt.subplots(2,2)
        cmap = "coolwarm"
        f = "{}_pf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        pf = np.load(f)
        
        f = "{}_lpf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        lpf = np.load(f)
    
        f = "{}_spf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        spf = np.load(f)
        
        f = "{}_lspf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        lspf = np.load(f)

        nmem = spf.shape[1]
        y = np.arange(nmem+1) + 1

        ymax = np.max(pf)
        ymin = np.min(pf)
        ymax = max(ymax, np.max(lpf))
        ymin = min(ymin, np.min(lpf))
        ylim = max(ymax, np.abs(ymin))
        mappable=ax[0, 0].pcolor(x,x,pf,cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
        ax[0, 0].set_aspect("equal")
        ax[0, 0].set_xticks(x[::5])
        ax[0, 0].set_yticks(x[::5])
        ax[0, 0].set_title("Pf")
        fig.colorbar(mappable, ax=ax[0,0],orientation="vertical")
        mappable=ax[0, 1].pcolor(x,x,lpf,cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
        ax[0, 1].set_aspect("equal")
        ax[0, 1].set_xticks(x[::5])
        ax[0, 1].set_yticks(x[::5])
        ax[0, 1].set_title("localized Pf")
        fig.colorbar(mappable, ax=ax[0,1],orientation="vertical")
        ymax = np.max(spf)
        ymin = np.min(spf)
        ymax = max(ymax, np.max(lspf))
        ymin = min(ymin, np.min(lspf))
        ylim = max(ymax, np.abs(ymin))
        mappable=ax[1, 0].pcolor(y,x,spf,cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
        ax[1, 0].set_aspect(1.2)
        ax[1, 0].set_yticks(x[::5])
        ax[1, 0].set_xticks(y[::5])
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_title("sqrtPf")
        fig.colorbar(mappable, ax=ax[1,0],orientation="vertical")
        mappable=ax[1, 1].pcolor(y,x,lspf,cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
        ax[1, 1].set_aspect(1.2)
        ax[1, 1].set_yticks(x[::5])
        ax[1, 1].set_xticks(y[::5])
        ax[1, 1].invert_yaxis()
        ax[1, 1].set_title("localized sqrtPf")
        fig.colorbar(mappable, ax=ax[1,1],orientation="vertical")
        #axpos = ax[0, 1].get_position()
        #cbar_ax = fig.add_axes([0.90, axpos.y0/4, 0.02, 2*axpos.height])
        #mappable = ScalarMappable(cmap=cmap)
        #fig.colorbar(mappable, cax=cbar_ax)
        fig.tight_layout()
        fig.savefig("{}_pf+lpf_{}_{}_cycle{}.png".format(model,op,pt,icycle))
