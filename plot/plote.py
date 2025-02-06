import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
from methods import perts, linecolor, iinflist, infltype, inflcolor

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = None
linfl = False
if len(sys.argv) > 4:
    pt = sys.argv[4]
if len(sys.argv) > 5 and sys.argv[5]=='infl':
    linfl = True
nspinup = na // 5
marker = {"3d":"o","4d":"x","3ds":"x","4ds":"^"}
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
x = np.arange(na) + 1
if model == "burgers":
    sigma = {"linear": 8.0e-2, "quadratic": 1.0e-3, "cubic": 1.0e-3, "quartic": 1.0e-2, \
    "quadratic-nodiff": 1.0e-3, "cubic-nodiff": 1.0e-3, "quartic-nodiff": 1.0e-2}
elif model == "kdvb":
    sigma = {"linear": 0.05, "quadratic": 0.05}
elif model == "qg":
    sigma = {"linear": 4.0}
if pt is not None:
    perts = list(set(perts))
    perts = [pt, pt+"be", pt+"bm", "l"+pt+"0", "l"+pt+"1", "l"+pt+"2", 'letkf']
    linecolor = {pt:'tab:blue',pt+"be":'tab:orange',pt+"bm":'tab:green',
    "l"+pt+"0":'tab:cyan', "l"+pt+"1":'tab:pink', "l"+pt+"2":'tab:purple', 'letkf':'tab:red'}
if linfl:
    linecolor = inflcolor
    linestyle = ['solid', 'dashed', 'dashdot', 'dotted']
y = np.ones(x.size) * sigma[op]
if model == "qg":
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(12,10),constrained_layout=True)
    figf, axf = plt.subplots(nrows=2,ncols=1,figsize=(12,10),constrained_layout=True)
    fig2, ax2 = plt.subplots(nrows=2,ncols=1,figsize=(12,10),constrained_layout=True)
    figf2, axf2 = plt.subplots(nrows=2,ncols=1,figsize=(12,10),constrained_layout=True)
else:
    fig, ax = plt.subplots(figsize=(12,5),constrained_layout=True)
    figf, axf = plt.subplots(figsize=(12,5),constrained_layout=True)
    fig2, ax2 = plt.subplots(figsize=(12,5),constrained_layout=True)
    figf2, axf2 = plt.subplots(figsize=(12,5),constrained_layout=True)
#ax2 = ax.twinx()
nspinup = na//5
i = 0
f = "enda_{}.txt".format(op)
try:
    e = np.loadtxt(f)
    if np.isnan(e).any():
        print("divergence in NoDA")
    else:
        print("NoDA, mean RMSE = {}".format(np.mean(e[nspinup:])))
        ax.plot(x, e, linestyle='dotted', color='gray', label='NoDA')
except OSError or FileNotFoundError:
    print("not exist {}".format(f))
ymax = 0.0
yfmax = 0.0
if not linfl:
    for pt in perts:
        if pt[:2] != "4d":
            c = linecolor[pt]
        else:
            c = linecolor[pt[2:]]
        label = pt
        ## analysis
        f = "e_{}_{}.txt".format(op, pt)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        e = np.loadtxt(f)
        if np.isnan(e).any():
            print("divergence in {}".format(pt))
            continue
        if model == "qg":
            e = e.reshape(-1,2)
            eavg = np.mean(e[nspinup:,:],axis=0)
            print("{}, analysis RMSE = {}".format(pt,eavg[1]))
        else:
            eavg = np.mean(e[nspinup:])
            print("{}, analysis RMSE = {}".format(pt,eavg))
        ymax = max(np.max(e),ymax)
        #ax.plot(x, e, linestyle=linestyle[pt], color=c, c)
        if model == "qg":
            if pt[:2] != "4d":
                for i in range(2):
                    ax[i].plot(x, e[:,i], linestyle="solid", color=c, label=label+f'={eavg[i]:.3f}')
            else:
                for i in range(2):
                    ax[i].plot(x, e[:,i], linestyle="dashed", color=c, label=label+f'={eavg[i]:.3f}')
        else:
            if model=="kdvb" or model=="burgers":
                label1=label+f'={eavg:.2e}'
            else:
                label1=label+f'={eavg:.3f}'
            if pt[:2] != "4d":
                ax.plot(x, e, marker=marker["3d"], color=c, label=label1)
            else:
                ax.plot(x, e, marker=marker["4d"], color=c, label=label1)
    #    if pt!="kf" and pt!="var" and pt!="4dvar":
        f = "stda_{}_{}.txt".format(op, pt)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        stda = np.loadtxt(f)
        if model == "qg":
            stda = stda.reshape(-1,2)
            for i in range(2):
                ax[i].plot(x, stda[:,i], linestyle="dashed", color=c, label=label+" stdv.")
        else:
            if pt[:2] != "4d":
                ax.plot(x, stda, linestyle="dashed", marker=marker["3ds"], color=c, label=label+" stdv.")
            else:
                ax.plot(x, stda, linestyle="dashed", marker=marker["4ds"], color=c, label=label+" stdv.")
        if model == "qg":
            if e.shape[1] > na:
                for i in range(2):
                    if pt[:2] != "4d":
                        ax2[i].plot(x[1:], stda[:,i]/e[1:,i], marker=marker["3d"], color=c, label=label)
                    else:
                        ax2[i].plot(x[1:], stda[:,i]/e[1:,i], marker=marker["4d"], color=c, label=label)
            else:
                for i in range(2):
                    if pt[:2] != "4d":
                        ax2[i].plot(x, stda[:,i]/e[:,i], marker=marker["3d"], color=c, label=label)
                    else:
                        ax2[i].plot(x, stda[:,i]/e[:,i], marker=marker["4d"], color=c, label=label)
        else:
            if e.size > na:
                if pt[:2] != "4d":
                    ax2.plot(x[1:], stda/e[1:], marker=marker["3d"], color=c, label=label)
                else:
                    ax2.plot(x[1:], stda/e[1:], marker=marker["4d"], color=c, label=label)
            else:
                if pt[:2] != "4d":
                    ax2.plot(x, stda/e, marker=marker["3d"], color=c, label=label)
                else:
                    ax2.plot(x, stda/e, marker=marker["4d"], color=c, label=label)
        ## forecast
        f = "ef_{}_{}.txt".format(op, pt)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        ef = np.loadtxt(f)
        if np.isnan(ef).any():
            print("divergence in {}".format(pt))
            continue
        if model == "qg":
            ef = ef.reshape(-1,2)
            efavg = np.mean(ef[nspinup:,:],axis=0)
            print("{}, forecast RMSE = {}".format(pt,efavg[1]))
        else:
            efavg = np.mean(ef[nspinup:])
            print("{}, forecast RMSE = {}".format(pt,efavg))
        yfmax = max(np.max(ef),yfmax)
        #ax.plot(x, e, linestyle=linestyle[pt], color=c, label=label)
        if model == "qg":
            if pt[:2] != "4d":
                for i in range(2):
                    axf[i].plot(x, ef[:,i], linestyle="solid", color=c, label=label+f'={efavg[i]:.3f}')
            else:
                for i in range(2):
                    axf[i].plot(x, ef[:,i], linestyle="dashed", color=c, label=label+f'={efavg[i]:.3f}')
        else:
            if model=="kdvb" or model=="burgers":
                label1=label+f'={efavg:.2e}'
            else:
                label1=label+f'={efavg:.3f}'
            if pt[:2] != "4d":
                axf.plot(x, ef, marker=marker["3d"], color=c, label=label1)
            else:
                axf.plot(x, ef, marker=marker["4d"], color=c, label=label1)
    #    if pt!="kf" and pt!="var" and pt!="4dvar":
        f = "stdf_{}_{}.txt".format(op, pt)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        stdf = np.loadtxt(f)
        if model == "qg":
            stdf = stdf.reshape(-1,2)
            for i in range(2):
                axf[i].plot(x, stdf[:,i], linestyle="dashed", color=c, label=label+" stdv.")
        else:
            if pt[:2] != "4d":
                axf.plot(x, stdf, linestyle="dashed", marker=marker["3ds"], color=c, label=label+" stdv.")
            else:
                axf.plot(x, stdf, linestyle="dashed", marker=marker["4ds"], color=c, label=label+" stdv.")
        if model == "qg":
            if ef.shape[1] > na:
                for i in range(2):
                    if pt[:2] != "4d":
                        axf2[i].plot(x[1:], stdf[:,i]/ef[1:,i], marker=marker["3d"], color=c, label=label)
                    else:
                        axf2[i].plot(x[1:], stdf[:,i]/ef[1:,i], marker=marker["4d"], color=c, label=label)
            else:
                for i in range(2):
                    if pt[:2] != "4d":
                        axf2[i].plot(x, stdf[:,i]/ef[:,i], marker=marker["3d"], color=c, label=label)
                    else:
                        axf2[i].plot(x, stdf[:,i]/ef[:,i], marker=marker["4d"], color=c, label=label)
        else:
            if ef.size > na:
                if pt[:2] != "4d":
                    axf2.plot(x[1:], stdf/ef[1:], marker=marker["3d"], color=c, label=label)
                else:
                    axf2.plot(x[1:], stdf/ef[1:], marker=marker["4d"], color=c, label=label)
            else:
                if pt[:2] != "4d":
                    axf2.plot(x, stdf/ef, marker=marker["3d"], color=c, label=label)
                else:
                    axf2.plot(x, stdf/ef, marker=marker["4d"], color=c, label=label)
else:
    for iinf in iinflist:
        itype = infltype[iinf]
        c = linecolor[iinf]
        label=itype
        ## analysis
        f = "e_{}_{}_{}.txt".format(op, pt, iinf)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        e = np.loadtxt(f)
        if np.isnan(e).any():
            print("divergence in {}-{}".format(pt,itype))
            continue
        if model == "qg":
            e = e.reshape(-1,2)
            eavg = np.mean(e[nspinup:,:],axis=0)
            print("{}-{}, analysis RMSE = {}".format(pt,itype,eavg[1]))
        else:
            eavg = np.mean(e[nspinup:])
            print("{}-{}, analysis RMSE = {}".format(pt,itype,eavg))
        ymax = max(np.max(e),ymax)
        #ax.plot(x, e, linestyle=linestyle[pt], color=c, label=label)
        if model == "qg":
            if pt[:2] != "4d":
                for i in range(2):
                    ax[i].plot(x, e[:,i], linestyle="solid", color=c, label=itype+f'={eavg[i]:.3f}')
            else:
                for i in range(2):
                    ax[i].plot(x, e[:,i], linestyle="dashed", color=c, label=itype+f'={eavg[i]:.3f}')
        else:
            if pt[:2] != "4d":
                ax.plot(x, e, marker=marker["3d"], color=c, label=itype+f'={eavg:.3f}')
            else:
                ax.plot(x, e, marker=marker["4d"], color=c, label=itype+f'={eavg:.3f}')
    #    if pt!="kf" and pt!="var" and pt!="4dvar":
        f = "stda_{}_{}_{}.txt".format(op, pt, iinf)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        stda = np.loadtxt(f)
        if model == "qg":
            stda = stda.reshape(-1,2)
        #    for i in range(2):
        #        ax[i].plot(x, stda[:,i], linestyle="dashed", color=c) #, label=itype+" stdv.")
        #else:
        #    if pt[:2] != "4d":
        #        ax.plot(x, stda, linestyle="dashed", marker=marker["3ds"], color=c) #, label=itype+" stdv.")
        #    else:
        #        ax.plot(x, stda, linestyle="dashed", marker=marker["4ds"], color=c) #, label=itype+" stdv.")
        if model == "qg":
            if e.shape[1] > na:
                for i in range(2):
                    if pt[:2] != "4d":
                        ax2[i].plot(x[1:], stda[:,i]/e[1:,i], marker=marker["3d"], color=c, label=itype)
                    else:
                        ax2[i].plot(x[1:], stda[:,i]/e[1:,i], marker=marker["4d"], color=c, label=itype)
            else:
                for i in range(2):
                    if pt[:2] != "4d":
                        ax2[i].plot(x, stda[:,i]/e[:,i], marker=marker["3d"], color=c, label=itype)
                    else:
                        ax2[i].plot(x, stda[:,i]/e[:,i], marker=marker["4d"], color=c, label=itype)
        else:
            if e.size > na:
                if pt[:2] != "4d":
                    ax2.plot(x[1:], stda/e[1:], marker=marker["3d"], color=c, label=itype)
                else:
                    ax2.plot(x[1:], stda/e[1:], marker=marker["4d"], color=c, label=itype)
            else:
                if pt[:2] != "4d":
                    ax2.plot(x, stda/e, marker=marker["3d"], color=c, label=itype)
                else:
                    ax2.plot(x, stda/e, marker=marker["4d"], color=c, label=itype)
        ## forecast
        f = "ef_{}_{}_{}.txt".format(op, pt, iinf)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        ef = np.loadtxt(f)
        if np.isnan(ef).any():
            print("divergence in {}".format(pt))
            continue
        if model == "qg":
            ef = ef.reshape(-1,2)
            efavg = np.mean(ef[nspinup:,:],axis=0)
            print("{}, forecast RMSE = {}".format(pt,efavg[1]))
        else:
            efavg = np.mean(ef[nspinup:])
            print("{}, forecast RMSE = {}".format(pt,efavg))
        yfmax = max(np.max(ef),yfmax)
        #ax.plot(x, e, linestyle=linestyle[pt], color=c, label=label)
        if model == "qg":
            if pt[:2] != "4d":
                for i in range(2):
                    axf[i].plot(x, ef[:,i], linestyle="solid", color=c, label=itype+f'={efavg[i]:.3f}')
            else:
                for i in range(2):
                    axf[i].plot(x, ef[:,i], linestyle="dashed", color=c, label=itype+f'={efavg[i]:.3f}')
        else:
            if pt[:2] != "4d":
                axf.plot(x, ef, marker=marker["3d"], color=c, label=itype+f'={efavg:.3f}')
            else:
                axf.plot(x, ef, marker=marker["4d"], color=c, label=itype+f'={efavg:.3f}')
    #    if pt!="kf" and pt!="var" and pt!="4dvar":
        f = "stdf_{}_{}_{}.txt".format(op, pt, iinf)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        stdf = np.loadtxt(f)
        if model == "qg":
            stdf = stdf.reshape(-1,2)
        #    for i in range(2):
        #        axf[i].plot(x, stdf[:,i], linestyle="dashed", color=c) #, label=label+" stdv.")
        #else:
        #    if pt[:2] != "4d":
        #        axf.plot(x, stdf, linestyle="dashed", marker=marker["3ds"], color=c) #, label=label+" stdv.")
        #    else:
        #        axf.plot(x, stdf, linestyle="dashed", marker=marker["4ds"], color=c) #, label=label+" stdv.")
        if model == "qg":
            if ef.shape[1] > na:
                for i in range(2):
                    if pt[:2] != "4d":
                        axf2[i].plot(x[1:], stdf[:,i]/ef[1:,i], marker=marker["3d"], color=c, label=label)
                    else:
                        axf2[i].plot(x[1:], stdf[:,i]/ef[1:,i], marker=marker["4d"], color=c, label=label)
            else:
                for i in range(2):
                    if pt[:2] != "4d":
                        axf2[i].plot(x, stdf[:,i]/ef[:,i], marker=marker["3d"], color=c, label=label)
                    else:
                        axf2[i].plot(x, stdf[:,i]/ef[:,i], marker=marker["4d"], color=c, label=label)
        else:
            if ef.size > na:
                if pt[:2] != "4d":
                    axf2.plot(x[1:], stdf/ef[1:], marker=marker["3d"], color=c, label=label)
                else:
                    axf2.plot(x[1:], stdf/ef[1:], marker=marker["4d"], color=c, label=label)
            else:
                if pt[:2] != "4d":
                    axf2.plot(x, stdf/ef, marker=marker["3d"], color=c, label=label)
                else:
                    axf2.plot(x, stdf/ef, marker=marker["4d"], color=c, label=label)
    #f = "{}_e_{}-nodiff_{}.txt".format(model, op, pt)
    #if not os.path.isfile(f):
    #    print("not exist {}".format(f))
    #    continue
    #e = np.loadtxt(f)
    #if np.isnan(e).any():
    #    continue
    #ax.plot(x, e, linestyle="dashed", color=c, label="{}-nodiff".format(pt))
    #i += 1
# observation error (loosely dashed)
if model == "qg":
    ax[1].plot(x, y, linestyle=(0, (5, 10)), color='black')
    ax[0].set(xlabel="analysis cycle", ylabel="RMSE",
        title=op+r" $q$")
    ax[1].set(xlabel="analysis cycle", ylabel="RMSE",
        title=op+r" $\psi$")
    ax2[0].set(xlabel="analysis cycle", ylabel="Pa/RMSE",
        title=op+r" $q$")
    ax2[1].set(xlabel="analysis cycle", ylabel="Pa/RMSE",
        title=op+r" $\psi$")
    axf[1].plot(x, y, linestyle=(0, (5, 10)), color='black')
    axf[0].set(xlabel="forecast cycle", ylabel="RMSE",
        title=op+r" $q$")
    axf[1].set(xlabel="forecast cycle", ylabel="RMSE",
        title=op+r" $\psi$")
    axf2[0].set(xlabel="forecast cycle", ylabel="Pf/RMSE",
        title=op+r" $q$")
    axf2[1].set(xlabel="forecast cycle", ylabel="Pf/RMSE",
        title=op+r" $\psi$")
    for i in range(2):
        if len(x) > 50:
            ax[i].set_xticks(x[::len(x)//10])
            ax[i].set_xticks(x[::len(x)//20], minor=True)
            ax2[i].set_xticks(x[::len(x)//10])
            ax2[i].set_xticks(x[::len(x)//20], minor=True)
            axf[i].set_xticks(x[::len(x)//10])
            axf[i].set_xticks(x[::len(x)//20], minor=True)
            axf2[i].set_xticks(x[::len(x)//10])
            axf2[i].set_xticks(x[::len(x)//20], minor=True)
        else:
            ax[i].set_xticks(x[::5])
            ax[i].set_xticks(x, minor=True)
            ax2[i].set_xticks(x[::5])
            ax2[i].set_xticks(x, minor=True)
            axf[i].set_xticks(x[::5])
            axf[i].set_xticks(x, minor=True)
            axf2[i].set_xticks(x[::5])
            axf2[i].set_xticks(x, minor=True)
        #ax2.set_yscale("log")
        ax[i].legend()
        ax2[i].legend()
        axf[i].legend()
        axf2[i].legend()
else:
    ax.plot(x, y, linestyle=(0, (5, 10)), color='black')
    ax.set(xlabel="analysis cycle", ylabel="RMSE",
        title=op)
    ax2.set(xlabel="analysis cycle", ylabel="Pa/RMSE",
        title=op)
    axf.plot(x, y, linestyle=(0, (5, 10)), color='black')
    axf.set(xlabel="forecast cycle", ylabel="RMSE",
        title=op)
    axf2.set(xlabel="forecast cycle", ylabel="Pf/RMSE",
        title=op)
    if model=='kdvb' or model=="burgers":
        #ax.set_ylim(-0.01,0.2)
        ax.set_yscale("log")
        axf.set_yscale("log")
    else:
        ax.set_ylim(-0.01,ymax)
        axf.set_ylim(-0.01,yfmax)
    if len(x) > 50:
        ax.set_xticks(x[::len(x)//10])
        ax.set_xticks(x[::len(x)//20], minor=True)
        ax2.set_xticks(x[::len(x)//10])
        ax2.set_xticks(x[::len(x)//20], minor=True)
        axf.set_xticks(x[::len(x)//10])
        axf.set_xticks(x[::len(x)//20], minor=True)
        axf2.set_xticks(x[::len(x)//10])
        axf2.set_xticks(x[::len(x)//20], minor=True)
    else:
        ax.set_xticks(x[::5])
        ax.set_xticks(x, minor=True)
        ax2.set_xticks(x[::5])
        ax2.set_xticks(x, minor=True)
        axf.set_xticks(x[::5])
        axf.set_xticks(x, minor=True)
        axf2.set_xticks(x[::5])
        axf2.set_xticks(x, minor=True)
    #ax2.set_yscale("log")
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,0.9))
    ax2.legend(loc='upper left',bbox_to_anchor=(1.01,0.9))
    axf.legend(loc='upper left',bbox_to_anchor=(1.01,0.9))
    axf2.legend(loc='upper left',bbox_to_anchor=(1.01,0.9))
if linfl:
    fig.savefig("{}_e_{}_{}.png".format(model, op, pt))
    fig2.savefig("{}_e+stda_{}_{}.png".format(model, op, pt))
    figf.savefig("{}_ef_{}_{}.png".format(model, op, pt))
    figf2.savefig("{}_ef+stdf_{}_{}.png".format(model, op, pt))
else:
    fig.savefig("{}_e_{}.png".format(model, op))
    fig2.savefig("{}_e+stda_{}.png".format(model, op))
    figf.savefig("{}_ef_{}.png".format(model, op))
    figf2.savefig("{}_ef+stdf_{}.png".format(model, op))
#fig.savefig("{}_e_{}+nodiff.pdf".format(model, op))
