import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0}
perts = ["mlef", "envar", "envar_nest", "etkf", "po", "srf", "letkf", "kf", "var", "var_nest",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
linecolor = {"mlef":'tab:blue',"envar":'tab:orange',"envar_nest":'tab:green',"etkf":'tab:green', "po":'tab:red',\
        "srf":"tab:pink", "letkf":"tab:purple", "kf":"tab:cyan", "var":"tab:olive","var_nest":"tab:brown",\
        "mlefcw":"tab:green","mlefy":"tab:orange","mlefbe":"tab:red","mlefbm":"tab:pink"}
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
            if ptype=="infl":
                var.append(float(tmp))
            elif ptype=="sigb":
                tmp2 = tmp.split()
                var.append(f"g{tmp2[0]}l{tmp2[1]}")
            else:
                var.append(int(tmp))
except FileNotFoundError:
    print("not found params.txt")
    pass
ix_gm = np.loadtxt('ix_gm.txt')
ix_lam = np.loadtxt('ix_lam.txt')
#y = np.ones(len(var)) * sigma[op]
methods = []
nsuccess_gm = {}
xdmean_gm = {}
xsmean_gm = {}
nsuccess_lam = {}
xdmean_lam = {}
xsmean_lam = {}
for pt in perts:
    i = 0
    j = 0
    nsuccess_gm[pt] = {}
    xdmean_gm[pt] = {}
    xsmean_gm[pt] = {}
    nsuccess_lam[pt] = {}
    xdmean_lam[pt] = {}
    xsmean_lam[pt] = {}
    for ivar in var:
        ## GM
        f = "{}_xdmean_gm_{}_{}_{}.txt".format(model, op, pt, ivar)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            xdmean_gm[pt][ivar] = None
            nsuccess_gm[pt][ivar] = 0
        else:
            data = np.loadtxt(f)
            e = data[:,1:]
            nsuccess_gm[pt][ivar] = data[0,0]
            xdmean_gm[pt][ivar] = e
            j += 1
        f = "{}_xsmean_gm_{}_{}_{}.txt".format(model, op, pt, ivar)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            xsmean_gm[pt][ivar] = None
        else:
            data = np.loadtxt(f)
            e = data[:,1:]
            xsmean_gm[pt][ivar] = e
        ## LAM
        f = "{}_xdmean_lam_{}_{}_{}.txt".format(model, op, pt, ivar)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            xdmean_lam[pt][ivar] = None
            nsuccess_lam[pt][ivar] = 0
        else:
            data = np.loadtxt(f)
            e = data[:,1:]
            nsuccess_lam[pt][ivar] = data[0,0]
            xdmean_lam[pt][ivar] = e
        f = "{}_xsmean_lam_{}_{}_{}.txt".format(model, op, pt, ivar)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            xsmean_lam[pt][ivar] = None
        else:
            data = np.loadtxt(f)
            e = data[:,1:]
            xsmean_lam[pt][ivar] = e
    #ax.plot(x, e, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    if j > 0:
        methods.append(pt)
i=0
## GM
nfigs = len(var)
ncols = 2
nrows = int(np.ceil(nfigs / ncols))
figwidth = 10
figheight = 3*nrows - 1
fig, axs = plt.subplots(figsize=[figwidth,figheight],nrows=nrows,ncols=ncols,sharex=True,constrained_layout=True)
ymax=0.0
for ivar, ax in zip(var,axs.flatten()):
    for pt in methods:
        xd_gm = xdmean_gm[pt][ivar]
        xs_gm = xsmean_gm[pt][ivar]
        ns_gm = nsuccess_gm[pt][ivar]
        if xd_gm is not None and xs_gm is not None:
            ax.plot(ix_gm,xd_gm[0,],lw=2.0,c=linecolor[pt],label=pt)
            ax.plot(ix_gm,xs_gm[0,],ls='dashed',c=linecolor[pt])
            ymax=max(ymax,np.max(xd_gm[0,]),np.max(xs_gm[0,]))
    ax.plot(ix_gm,np.ones(ix_gm.size)*sigma[op],c='k',ls='dotted')
    ax.vlines([ix_lam[0],ix_lam[-1]],0,1,colors='gray',alpha=0.5,transform=ax.get_xaxis_transform())
    ax.set_title(f"{ptype}={ivar} : #{int(ns_gm):d}")
for i in range(nrows):
    axs[i,0].set_ylabel("RMSE or SPREAD")
for ax in axs.flatten():
    ax.set_ylim(0.0,ymax)
if len(axs.flatten())>len(var):
    axs[nrows-1,ncols-1].remove()
axs[0,0].legend()
fig.suptitle(op+", GM")
fig.savefig("{}_xdmean_gm_{}_{}.png".format(model, ptype, op))
## LAM
fig, axs = plt.subplots(figsize=[figwidth,figheight],nrows=nrows,ncols=ncols,sharex=True,constrained_layout=True)
ymax=0.0
for ivar, ax in zip(var,axs.flatten()):
    for pt in methods:
        xd_lam = xdmean_lam[pt][ivar]
        xs_lam = xsmean_lam[pt][ivar]
        ns_lam = nsuccess_lam[pt][ivar]
        if xd_lam is not None or xs_lam is not None:
            ax.plot(ix_lam,xd_lam[0,],lw=2.0,c=linecolor[pt],label=pt)
            ax.plot(ix_lam,xs_lam[0,],ls='dashed',c=linecolor[pt])
            ymax=max(ymax,np.max(xd_lam[0,]),np.max(xs_lam[0,]))
    ax.plot(ix_lam,np.ones(ix_lam.size)*sigma[op],c='k',ls='dotted')
    ax.set_title(f"{ptype}={ivar} : #{int(ns_lam):d}")
for i in range(nrows):
    axs[i,0].set_ylabel("RMSE or SPREAD")
for ax in axs.flatten():
    ax.set_ylim(0.0,ymax)
if len(axs.flatten())>len(var):
    axs[nrows-1,ncols-1].remove()
axs[0,0].legend()
fig.suptitle(op+", LAM")
fig.savefig("{}_xdmean_lam_{}_{}.png".format(model, ptype, op))
