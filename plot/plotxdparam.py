import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
from methods import perts, linecolor

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0}
if len(sys.argv)>4:
    ptype = sys.argv[4]
    if ptype == "loc":
        var = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    elif ptype == "infl":
        var = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
    elif ptype == "nobs":
        var = [40, 35, 30, 25, 20, 15, 10]
    elif ptype == "nmem":
        var = [40, 35, 30, 25, 20, 15, 10, 5]
    elif ptype == "nt":
        var = [1, 2, 3, 4, 5, 6, 7, 8]
    elif ptype == "a_window":
        perts = ["4dvar","4dletkf","4dmlefbe","4dmlefbm","4dmlefcw","4dmlefy"]
        linecolor = {"var":"tab:olive",
        "letkf":"tab:blue",
        "mlefbe":"tab:red","mlefbm":"tab:pink",
        "mlefcw":"tab:green","mlefy":"tab:orange"}
        var = [1, 2, 3, 4, 5, 6, 7, 8]
try:
    with open("params.txt","r") as f:
        ptype=f.readline()[:-1]
        var=[]
        while(True):
            tmp=f.readline()[:-1]
            if tmp=='': break
            if ptype=="infl" or ptype=="sigo" or ptype=="sigb" or ptype=="lb":
                var.append(float(tmp))
            else:
                var.append(int(tmp))
except FileNotFoundError:
    print("not found params.txt")
    pass
ix = np.loadtxt('ix.txt')
#y = np.ones(len(var)) * sigma[op]
methods = []
nsuccess = {}
xdmean = {}
xsmean = {}
for pt in perts:
    i = 0
    j = 0
    nsuccess[pt] = {}
    xdmean[pt] = {}
    xsmean[pt] = {}
    for ivar in var:
        f = "{}_xdmean_{}_{}_{}.txt".format(model, op, pt, ivar)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            xdmean[pt][ivar] = None
            nsuccess[pt][ivar] = 0
        else:
            data = np.loadtxt(f)
            e = data[:,1:]
            nsuccess[pt][ivar] = data[0,0]
            xdmean[pt][ivar] = e
            j += 1
        if pt != "kf" and pt != "var" and pt != "4dvar":
            f = "{}_xsmean_{}_{}_{}.txt".format(model, op, pt, ivar)
            if not os.path.isfile(f):
                print("not exist {}".format(f))
                xsmean[pt][ivar] = None
            else:
                data = np.loadtxt(f)
                e = data[:,1:]
                xsmean[pt][ivar] = e
        else:
            xsmean[pt][ivar] = None
    #ax.plot(x, e, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    if j > 0:
        methods.append(pt)
i=0
nfigs = len(var)
ncols = 2
nrows = int(np.ceil(nfigs / ncols))
figwidth = 10
figheight = 3*nrows - 1
fig, axs = plt.subplots(figsize=[figwidth,figheight],nrows=nrows,ncols=ncols,sharex=True,constrained_layout=True)
for ivar, ax in zip(var,axs.flatten()):
    for pt in methods:
        xd = xdmean[pt][ivar]
        xs = xsmean[pt][ivar]
        ns = nsuccess[pt][ivar]
        if xd is not None:
            ax.plot(ix,xd[0,],lw=2.0,c=linecolor[pt],label=pt)
            ax.plot(ix,np.ones(ix.size)*sigma[op],c='k',ls='dotted')
        if xs is not None:
            ax.plot(ix,xs[0,],ls='dashed',c=linecolor[pt])
    ax.set_title(f"{ptype}={ivar} : #{int(ns):d}")
for i in range(nrows):
    axs[i,0].set_ylabel("RMSE or SPREAD")
axs[0,0].legend()
if len(axs.flatten())>len(var):
    axs[nrows-1,ncols-1].remove()
fig.suptitle(op)
fig.savefig("{}_xdmean_{}_{}.png".format(model, ptype, op))
