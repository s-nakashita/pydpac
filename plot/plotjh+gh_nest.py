import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "envar", "envar_nest", "var","var_nest",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4dvar", "4dmlef"]
linecolor = {"mlef":'tab:blue',"envar":'tab:orange',"envar_nest":'tab:green',"var":"tab:olive","var_nest":"tab:brown",\
        "mlefcw":"tab:green","mlefy":"tab:orange","mlefbe":"tab:red","mlefbm":"tab:pink"}
marker = {"3d":"o","4d":"x"}
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
fig = plt.figure(figsize=(12,6),constrained_layout=True)
gs0 = gridspec.GridSpec(2, 5, figure=fig)
ax0 = fig.add_subplot(gs0[0,:4])
ax1 = fig.add_subplot(gs0[1,:4])
axl = fig.add_subplot(gs0[:, 4])
ax = [ax0, ax1]
fig2, ax2 = plt.subplots(nrows=2,ncols=2,figsize=(12,8),constrained_layout=True)
i = 0
vmax = 0.0
lplot = False
lines = []
labels = []
lines2 = []
labels2 = []
for pt in perts:
    f = "{}_gm_jh_{}_{}.txt".format(model, op, pt)
    if os.path.isfile(f):
        j_gm = np.loadtxt(f)
        f = "{}_gm_gh_{}_{}.txt".format(model, op, pt)
        g_gm = np.loadtxt(f)
        f = "{}_gm_niter_{}_{}.txt".format(model, op, pt)
        niter_gm = np.loadtxt(f)
        f = "{}_lam_jh_{}_{}.txt".format(model, op, pt)
        j_lam = np.loadtxt(f)
        f = "{}_lam_gh_{}_{}.txt".format(model, op, pt)
        g_lam = np.loadtxt(f)
        f = "{}_lam_niter_{}_{}.txt".format(model, op, pt)
        niter_lam = np.loadtxt(f)
        cycles = np.arange(j_gm.shape[0])
        lplot=True
    else:
        cycles = []
        j_gm = []
        g_gm = []
        j_lam = []
        g_lam = []
        niter_gm = []
        niter_lam = []
        for icycle in range(na):
            #GM
            f = "{}_gm_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle)
            if not os.path.isfile(f): continue
            j = np.loadtxt(f)
            if j.ndim == 2:
                j_gm.append(j[-1,:])
            else:
                j_gm.append(j)
            f = "{}_gm_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle)
            if not os.path.isfile(f): continue
            g = np.loadtxt(f)
            if g.ndim == 1:
                g_gm.append(g[-1])
                niter_gm.append(g.size)
            else:
                g_gm.append(g)
                niter_gm.append(1)
            #LAM
            f = "{}_lam_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle)
            if not os.path.isfile(f): continue
            j = np.loadtxt(f)
            if j.ndim == 2:
                j_lam.append(j[-1,:])
            else:
                j_lam.append(j)
            f = "{}_lam_gh_{}_{}_cycle{}.txt".format(model, op, pt, icycle)
            if not os.path.isfile(f): continue
            g = np.loadtxt(f)
            if g.ndim == 1:
                g_lam.append(g[-1])
                niter_lam.append(g.size)
            else:
                g_lam.append(g)
                niter_lam.append(1)
            #
            cycles.append(icycle)
        if len(cycles) == 0:
            print("not exist {}".format(pt))
            continue
        lplot=True
        j_gm = np.array(j_gm)
        g_gm = np.array(g_gm)
        niter_gm = np.array(niter_gm)
        j_lam = np.array(j_lam)
        g_lam = np.array(g_lam)
        niter_lam = np.array(niter_lam)
        np.savetxt("{}_gm_jh_{}_{}.txt".format(model, op, pt), j_gm)
        np.savetxt("{}_gm_gh_{}_{}.txt".format(model, op, pt), g_gm)
        np.savetxt("{}_gm_niter_{}_{}.txt".format(model, op, pt), niter_gm)
        np.savetxt("{}_lam_jh_{}_{}.txt".format(model, op, pt), j_lam)
        np.savetxt("{}_lam_gh_{}_{}.txt".format(model, op, pt), g_lam)
        np.savetxt("{}_lam_niter_{}_{}.txt".format(model, op, pt), niter_lam)
    print("{}, GM mean J = {}".format(pt,np.mean(j_gm.sum(axis=1))))
    print("{}, GM mean dJ = {}".format(pt,np.mean(g_gm)))
    print("{}, LAM mean J = {}".format(pt,np.mean(j_lam.sum(axis=1))))
    print("{}, LAM mean dJ = {}".format(pt,np.mean(g_lam)))
    ax[0].plot(cycles, j_gm[:,0], linestyle="solid", color=linecolor[pt]) #, label=pt+",Jb")
    lines.append(Line2D([0],[0],color=linecolor[pt]))
    labels.append(pt+",Jb")
    ax[0].plot(cycles, j_gm[:,1], linestyle="dashed", color=linecolor[pt]) #, label=pt+",Jo")
    lines.append(Line2D([0],[0],color=linecolor[pt],linestyle="dashed"))
    labels.append(pt+",Jo")
    ax[1].plot(cycles, j_lam[:,0], linestyle="solid", color=linecolor[pt]) #, label=pt+",Jb")
    #lines.append(Line2D([0],[0],color=linecolor[pt]))
    #labels.append(pt+",Jb")
    ax[1].plot(cycles, j_lam[:,1], linestyle="dashed", color=linecolor[pt]) #, label=pt+",Jo")
    #lines.append(Line2D([0],[0],color=linecolor[pt],linestyle="dashed"))
    #labels.append(pt+",Jo")
    if pt=="var_nest" or pt=="envar_nest":
        ax[1].plot(cycles, j_lam[:,2], linestyle="dotted", color=linecolor[pt]) #, label=pt+",Jk")
        lines.append(Line2D([0],[0],color=linecolor[pt],linestyle="dotted"))
        labels.append(pt+",Jk")
    ax2[0,0].plot(cycles, g_gm, linestyle="dashdot", color=linecolor[pt]) #,label=pt+r",$\nabla$J")
    lines2.append(Line2D([0],[0],color=linecolor[pt],linestyle="dashdot"))
    labels2.append(pt+r",$\nabla$J")
    ax2[0,1].bar(cycles, niter_gm, color=linecolor[pt], alpha=0.5)
    ax2[1,0].plot(cycles, g_lam, linestyle="dashdot", color=linecolor[pt]) #,label=pt+r",$\nabla$J")
    ax2[1,1].bar(cycles, niter_lam, color=linecolor[pt], alpha=0.5)
    #lines.append(Line2D([0],[0],color=linecolor[pt],linestyle="dashdot"))
    #labels.append(pt+r",$\nabla$J")
if lplot:
    ax[0].set(xlabel="analysis cycles", ylabel="J",
        title=op+" GM")
    ax2[0,0].set(xlabel="analysis cycles", ylabel=r"$\nabla$J",
        title=op+" GM")
    ax2[0,1].set_ylabel('iteration number',color='red')
    ax[1].set(xlabel="analysis cycles", ylabel="J",
        title=op+" LAM")
    ax2[1,0].set(xlabel="analysis cycles", ylabel=r"$\nabla$J",
        title=op+" LAM")
    ax2[1,1].set_ylabel('iteration number',color='red')
    for ax1 in ax.flatten():
        ax1.set_yscale("log")
        ax1.set_xlim(-1,100)
    for ax1 in ax2[:,0]:
        ax1.set_yscale("log")
        ax1.set_xlim(-1,100)
    for ax1 in ax2[:,1]:
        ax1.set_xlim(-1,100)
    axl.legend(lines,labels,loc='center')
    ax2[1,0].legend(lines2,labels2)
    fig.savefig("{}_jh_{}.png".format(model, op))
    fig2.savefig("{}_gh_{}.png".format(model, op))
