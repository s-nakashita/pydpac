import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
from matplotlib.lines import Line2D

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "envar", "var","var_nest",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4dvar", "4dmlef"]
linecolor = {"mlef":'tab:blue',"envar":'tab:orange', "var":"tab:olive","var_nest":"tab:brown",\
        "mlefcw":"tab:green","mlefy":"tab:orange","mlefbe":"tab:red","mlefbm":"tab:pink"}
marker = {"3d":"o","4d":"x"}
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(12,8),constrained_layout=True)
ax1_2 = ax[1].twinx()
i = 0
vmax = 0.0
lplot = False
lines = []
labels = []
lines2 = []
labels2 = []
for pt in perts:
    f = "{}_jh_{}_{}.txt".format(model, op, pt)
    if os.path.isfile(f):
        j = np.loadtxt(f)
        f = "{}_gh_{}_{}.txt".format(model, op, pt)
        g = np.loadtxt(f)
        f = "{}_niter_{}_{}.txt".format(model, op, pt)
        niter = np.loadtxt(f)
        cycles = np.arange(j.shape[0])
        lplot=True
    else:
        cycles = []
        j = []
        g = []
        niter = []
        for icycle in range(na):
            f = "{}_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle)
            if not os.path.isfile(f): continue
            j1 = np.loadtxt(f)
            if j1.ndim == 2:
                j.append(j1[-1,:])
            else:
                j.append(j1)
            f = "{}_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle)
            if not os.path.isfile(f): continue
            g1 = np.loadtxt(f)
            if g1.ndim == 1:
                g.append(g1[-1])
                niter.append(g1.size)
            else:
                g.append(g1)
                niter.append(1)
            #
            cycles.append(icycle)
        if len(cycles) == 0:
            print("not exist {}".format(pt))
            continue
        lplot=True
        j = np.array(j)
        g = np.array(g)
        niter = np.array(niter)
        np.savetxt("{}_jh_{}_{}.txt".format(model, op, pt), j)
        np.savetxt("{}_gh_{}_{}.txt".format(model, op, pt), g)
        np.savetxt("{}_niter_{}_{}.txt".format(model, op, pt), niter)
    print("{}, mean J = {}".format(pt,np.mean(j.sum(axis=1))))
    print("{}, mean dJ = {}".format(pt,np.mean(g)))
    ax[0].plot(cycles, j[:,0], linestyle="solid", color=linecolor[pt]) #, label=pt+",Jb")
    lines.append(Line2D([0],[0],color=linecolor[pt]))
    labels.append(pt+",Jb")
    ax[0].plot(cycles, j[:,1], linestyle="dashed", color=linecolor[pt]) #, label=pt+",Jo")
    lines.append(Line2D([0],[0],color=linecolor[pt],linestyle="dashed"))
    labels.append(pt+",Jo")
    ax[1].plot(cycles, g, linestyle="dashdot", color=linecolor[pt]) #,label=pt+r",$\nabla$J")
    lines2.append(Line2D([0],[0],color=linecolor[pt],linestyle="dashdot"))
    labels2.append(pt+r",$\nabla$J")
    ax1_2.bar(cycles, niter, color=linecolor[pt], alpha=0.5)
if lplot:
    ax[0].set(xlabel="analysis cycles", ylabel="J",
        title=op)
    ax[1].set(xlabel="analysis cycles", ylabel=r"$\nabla$J",
        title=op)
    ax1_2.tick_params(axis='y',labelcolor='red')
    ax1_2.set_ylabel('iteration number',color='red')
    ax1_2.set_ylim(0,30)
    for ax1 in ax.flatten():
        ax1.set_yscale("log")
        ax1.set_xlim(-1,100)
    ax[0].legend(lines,labels,ncol=2)
    ax[1].legend(lines2,labels2)
    fig.savefig("{}_jh+gh_{}.png".format(model, op))
