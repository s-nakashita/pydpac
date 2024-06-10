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
perts = ["mlef", "envar", "mlef_nest", "mlef_nestc",\
    "envar_nest", "envar_nestc", "var","var_nest",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4dvar", "4dmlef"]
linecolor = {"mlef":'tab:blue',"mlef_nest":'tab:purple',"mlef_nestc":'tab:cyan',\
    "envar":'tab:orange',"envar_nest":'tab:green',"envar_nestc":'lime',"var":"tab:olive","var_nest":"tab:brown",\
    "mlefcw":"tab:green","mlefy":"tab:orange","mlefbe":"tab:red","mlefbm":"tab:pink"}
marker = {"3d":"o","4d":"x"}
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
# J all
fig = plt.figure(figsize=(12,6),constrained_layout=True)
gs0 = gridspec.GridSpec(2, 5, figure=fig)
ax0 = fig.add_subplot(gs0[0,:4])
ax1 = fig.add_subplot(gs0[1,:4])
axl0 = fig.add_subplot(gs0[0, 4])
axl1 = fig.add_subplot(gs0[1, 4])
ax = [ax0, ax1]
# J each
fige = plt.figure(figsize=(12,12),constrained_layout=True)
gse = gridspec.GridSpec(3,2,figure=fige)
axe0_gm = fige.add_subplot(gse[0,0])
axe1_gm = fige.add_subplot(gse[1,0])
axe0_lam = fige.add_subplot(gse[0,1])
axe1_lam = fige.add_subplot(gse[1,1])
axe2_lam = fige.add_subplot(gse[2,1])
# dJ & iteration
fig2, ax2 = plt.subplots(nrows=2,ncols=2,figsize=(12,8),constrained_layout=True)
i = 0
vmax = 0.0
lplot = False
lines0 = []
labels0 = []
lines1 = []
labels1 = []
lines2 = []
labels2 = []
lines3 = []
labels3 = []
plot_Jk = False
for pt in perts:
    #GM
    f = "{}_gm_jh_{}_{}.txt".format(model, op, pt)
    if os.path.isfile(f):
        j_gm = np.loadtxt(f)
        f = "{}_gm_gh_{}_{}.txt".format(model, op, pt)
        g_gm = np.loadtxt(f)
        f = "{}_gm_niter_{}_{}.txt".format(model, op, pt)
        niter_gm = np.loadtxt(f)
        cycles_gm = np.arange(j_gm.shape[0]).tolist()
        lplot=True
    else:
        cycles_gm = []
        j_gm = []
        g_gm = []
        niter_gm = []
        for icycle in range(na):
            #GM
            f = "{}_gm_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle)
            if os.path.isfile(f):
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
                cycles_gm.append(icycle)
        if len(j_gm) > 0 and len(g_gm) > 0 and len(niter_gm) > 0:
            j_gm = np.array(j_gm)
            g_gm = np.array(g_gm)
            niter_gm = np.array(niter_gm)
            np.savetxt("{}_gm_jh_{}_{}.txt".format(model, op, pt), j_gm)
            np.savetxt("{}_gm_gh_{}_{}.txt".format(model, op, pt), g_gm)
            np.savetxt("{}_gm_niter_{}_{}.txt".format(model, op, pt), niter_gm)
    #LAM
    f = "{}_lam_jh_{}_{}.txt".format(model, op, pt)
    if os.path.isfile(f):
        j_lam = np.loadtxt(f)
        f = "{}_lam_gh_{}_{}.txt".format(model, op, pt)
        g_lam = np.loadtxt(f)
        f = "{}_lam_niter_{}_{}.txt".format(model, op, pt)
        niter_lam = np.loadtxt(f)
        nspinup = max(len(cycles_gm) - j_lam.shape[0], 0)
        cycles_lam = np.arange(nspinup,j_lam.shape[0]+nspinup)
        lplot=True
    else:
        cycles_lam = []
        j_lam = []
        g_lam = []
        niter_lam = []
        for icycle in range(na):
            #LAM
            f = "{}_lam_jh_{}_{}_cycle{}.txt".format(model, op, pt, icycle)
            if os.path.isfile(f):
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
                cycles_lam.append(icycle)
            #
            #cycles.append(icycle)
        if len(j_lam) > 0 and len(g_lam) > 0 and len(niter_lam) > 0:
            j_lam = np.array(j_lam)
            g_lam = np.array(g_lam)
            niter_lam = np.array(niter_lam)
            np.savetxt("{}_lam_jh_{}_{}.txt".format(model, op, pt), j_lam)
            np.savetxt("{}_lam_gh_{}_{}.txt".format(model, op, pt), g_lam)
            np.savetxt("{}_lam_niter_{}_{}.txt".format(model, op, pt), niter_lam)
    if len(cycles_gm) == 0 and len(cycles_lam) == 0:
        print("not exist {}".format(pt))
        continue
    lplot=True
    if len(cycles_gm) > 0:
        print("{}, GM mean J = {}".format(pt,np.mean(j_gm.sum(axis=1))))
        print("{}, GM mean dJ = {}".format(pt,np.mean(g_gm)))
        ax[0].plot(cycles_gm, np.sum(j_gm,axis=1), linestyle="solid", color=linecolor[pt]) #, label=pt+",Jb")
        lines0.append(Line2D([0],[0],color=linecolor[pt]))
        labels0.append(pt+f",{np.mean(np.sum(j_gm,axis=1)):.3e}") #+",Jb")
        axe0_gm.plot(cycles_gm, j_gm[:,0], color=linecolor[pt], label=pt)
        axe1_gm.plot(cycles_gm, j_gm[:,1], color=linecolor[pt]) #, label=pt+",Jo")
        #lines.append(Line2D([0],[0],color=linecolor[pt],linestyle="dashed"))
        #labels.append(pt+",Jo")
    if len(cycles_lam) > 0:
        print("{}, LAM mean J = {}".format(pt,np.mean(j_lam.sum(axis=1))))
        print("{}, LAM mean dJ = {}".format(pt,np.mean(g_lam)))
        ax[1].plot(cycles_lam, np.sum(j_lam,axis=1), linestyle="solid", color=linecolor[pt]) #, label=pt+",Jb")
        lines1.append(Line2D([0],[0],color=linecolor[pt]))
        labels1.append(pt+f",{np.mean(np.sum(j_lam,axis=1)):.3e}")
        axe0_lam.plot(cycles_lam, j_lam[:,0], color=linecolor[pt], label=pt)
        axe1_lam.plot(cycles_lam, j_lam[:,1], color=linecolor[pt]) #, label=pt+",Jo")
        lines3.append(Line2D([0],[0],color=linecolor[pt]))
        labels3.append(pt)
        if pt=="var_nest" or pt=="envar_nest" or pt=="mlef_nest":
            plot_Jk=True
            axe2_lam.plot(cycles_lam, j_lam[:,2], color=linecolor[pt]) #, label=pt+",Jk")
            #lines.append(Line2D([0],[0],color=linecolor[pt],linestyle="dotted"))
            #labels.append(pt+",Jk")
    if len(cycles_gm) > 0:
        ax2[0,0].plot(cycles_gm, g_gm, linestyle="dashdot", color=linecolor[pt]) #,label=pt+r",$\nabla$J")
        lines2.append(Line2D([0],[0],color=linecolor[pt],linestyle="dashdot"))
        labels2.append(pt+r",$\nabla$J")
        ax2[0,1].bar(cycles_gm, niter_gm, color=linecolor[pt], alpha=0.5)
    if len(cycles_lam) > 0:
        ax2[1,0].plot(cycles_lam, g_lam, linestyle="dashdot", color=linecolor[pt]) #,label=pt+r",$\nabla$J")
        ax2[1,1].bar(cycles_lam, niter_lam, color=linecolor[pt], alpha=0.5)
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
    for ax1 in ax:
        ax1.set_yscale("log")
        #ax1.set_xlim(-1,100)
    for ax1 in ax2[:,0]:
        ax1.set_yscale("log")
        #ax1.set_xlim(-1,100)
    #for ax1 in ax2[:,1]:
    #    ax1.set_xlim(-1,100)
    axl0.axis("off")
    axl0.legend(lines0,labels0,loc='center')
    axl1.axis("off")
    axl1.legend(lines1,labels1,loc='center')
    ax2[1,0].legend(lines2,labels2)
    fig.savefig("{}_jh_{}.png".format(model, op))
    fig2.savefig("{}_gh_{}.png".format(model, op))

    axe0_gm.set(ylabel="Jb",
        title=op+" GM")
    axe1_gm.set(xlabel="analysis cycles", ylabel="Jo")
    axe0_lam.set(ylabel="Jb",
        title=op+" LAM")
    if not plot_Jk:
        axe2_lam.axis("off")
        axe1_lam.set(xlabel="analysis cycles", ylabel="Jo")
    else:
        axe1_lam.set(ylabel="Jo")
        axe2_lam.set(xlabel="analysis cycles", ylabel="Jk")
        axe2_lam.set_yscale("log")
    for ax in [axe0_gm, axe1_gm, axe0_lam, axe1_lam]:
        ax.set_yscale("log")
    axl = fige.add_subplot(gse[2,0])
    axl.axis("off")
    axl.legend(lines3,labels3,loc='center')
    fige.savefig("{}_jheach_{}.png".format(model,op))