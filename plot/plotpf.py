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
    perts = ["mlef", "grad", "etkf-jh", "etkf-fh"]
    cycle = [1, 2, 3, 4]
    #na = 20
elif model == "l96":
    nx = 40
    perts = ["mlef", "etkf", "po", "srf"]
    #na = 300
x = np.arange(nx+1) + 1
for icycle in [1, 50, 100, 150, 200, 250]:
    cmap = "coolwarm"
    f = "{}_pf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    pf = np.load(f)
        
    f = "{}_spf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    spf = np.load(f)
        
    nmem = spf.shape[1]
    lam, v = la.eigh(pf)
    lam = lam[::-1]
    v = v[:,::-1]
    print(lam)
    if pt == "mlef":
        lspf = v[:,:nmem] @ np.diag(np.sqrt(lam[:nmem]))
    else:
        lspf = v[:,:nmem] @ np.diag(np.sqrt(lam[:nmem])) * np.sqrt(nmem-1)
    pfr = v[:,:nmem] @ np.diag(lam[:nmem]) @ v[:,:nmem].T
    y = np.arange(nmem+1) + 1

    fig, ax = plt.subplots()
    mode = np.arange(lam.size) + 1
    ax.bar(mode, lam, width=0.35)
    ax.set_xticks(mode[::5])
    ax.set_title("eigenvalues")
    fig.savefig("{}_pfeig_{}_{}_cycle{}.png".format(model,op,pt,icycle))
    
    fig, ax = plt.subplots(2,2)
    ymax = np.max(pf)
    ymin = np.min(pf)
    ylim = max(ymax, np.abs(ymin))
    mappable=ax[0, 0].pcolor(x,x,pf,cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
    ax[0, 0].set_aspect("equal")
    ax[0, 0].set_xticks(x[::5])
    ax[0, 0].set_yticks(x[::5])
    #ax[0, 0].invert_xaxis()
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_title("Pf")
    fig.colorbar(mappable, ax=ax[0,0],orientation="vertical")
    mappable=ax[0, 1].pcolor(x,x,pfr,cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
    ax[0, 1].set_aspect("equal")
    ax[0, 1].set_xticks(x[::5])
    ax[0, 1].set_yticks(x[::5])
    #ax[1, 0].invert_xaxis()
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_title("Pf (reconstructed)")
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
    if pt == "mlef":
        ax[1, 0].set_title("sqrtPf")
    else:
        ax[1, 0].set_title("dXf")
    fig.colorbar(mappable, ax=ax[1,0],orientation="vertical")
    
    mappable=ax[1, 1].pcolor(y,x,lspf,cmap=cmap,norm=Normalize(vmin=-ylim, vmax=ylim))
    ax[1, 1].set_aspect(1.2)
    ax[1, 1].set_yticks(x[::5])
    ax[1, 1].set_xticks(y[::5])
    ax[1, 1].invert_yaxis()
    if pt == "mlef":
        ax[1, 1].set_title("reconstructed sqrtPf")
    else:
        ax[1, 1].set_title("reconstructed dXf")
    fig.colorbar(mappable, ax=ax[1,1],orientation="vertical")
    
    #axpos = ax[0, 1].get_position()
    #cbar_ax = fig.add_axes([0.90, axpos.y0/4, 0.02, 2*axpos.height])
    #mappable = ScalarMappable(cmap=cmap)
    #fig.colorbar(mappable, cax=cbar_ax)
    fig.tight_layout()
    fig.savefig("{}_pf_{}_{}_cycle{}.png".format(model,op,pt,icycle))
