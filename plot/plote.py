import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "envar", "etkf", "po", "srf", "letkf", "kf", "var",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
linecolor = {"mlef":'tab:blue',"envar":'tab:orange',"etkf":'tab:green', "po":'tab:red',\
        "srf":"tab:pink", "letkf":"tab:purple", "kf":"tab:cyan", "var":"tab:olive"}
marker = {"3d":"o","4d":"x","3ds":"x","4ds":"^"}
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
x = np.arange(na) + 1
if model == "z08":
    #perts = ["mlef", "etkf", "po", "srf"]
    perts = ["mlef-fh", "mlef-jh", "etkf-fh", "etkf-jh"]#, "var"]
    linecolor = {"mlef-fh":'tab:blue',"mlef-jh":'tab:orange',"etkf-fh":'tab:green',"etkf-jh":'tab:red',
     "var":"tab:cyan"}
    #perts = ["mlef-fh", "mlef-jh", "mlefw-fh", "mlefw-jh"]
    #linecolor = {"mlef-fh":'tab:blue',"mlef-jh":'tab:orange',"mlefw-fh":'tab:green',"mlefw-jh":'tab:red'}
    #linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf":'tab:green', "po":'tab:red',\
    #   "srf":"tab:pink", "letkf":"tab:purple", "kf":"tab:cyan", "var":"tab:olive",\
    #   "var4d":"tab:brown"}
    #perts = ["mlef", "grad", "etkf-fh", "etkf-jh"]#, "po", "srf", "letkf"]
    #linestyle = {"mlef":"solid", "grad":"dashed",
    # "etkf-fh":"solid", "etkf-jh":"dashed"}
    #linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf-fh":'tab:green',"etkf-jh":'tab:red'}   
    x = np.arange(na+1)
    #sigma = {"linear": 8.0e-2, "quadratic": 8.0e-2, "cubic": 7.0e-4, "quartic": 7.0e-4,\
    #"quadratic-nodiff": 8.0e-2, "cubic-nodiff": 7.0e-4, "quartic-nodiff": 7.0e-4}
    sigma = {"linear": 8.0e-2, "quadratic": 1.0e-3, "cubic": 1.0e-3, "quartic": 1.0e-2, \
    "quadratic-nodiff": 1.0e-3, "cubic-nodiff": 1.0e-3, "quartic-nodiff": 1.0e-2}
elif model == "z05":
    perts = ["mlef", "etkf", "po", "srf", "letkf", "kf", "var"]
    linecolor = {"mlef":'tab:blue',"etkf":'tab:orange', "po":'tab:green',\
        "srf":"tab:red", "letkf":"tab:pink", "kf":"tab:purple", "var":"tab:cyan",\
        "var4d":"tab:brown"}
    x = np.arange(na)+1
    sigma = {"linear": 0.05, "quadratic": 0.05}
elif model == "l96" or model == "tc87":
    perts = ["mlef", "envar", "etkf", "po", "srf", "letkf", "kf", "var",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
    linecolor = {"mlef":'tab:blue',"envar":'tab:orange',"etkf":'tab:green', "po":'tab:red',\
        "srf":"tab:pink", "letkf":"tab:purple", "kf":"tab:cyan", "var":"tab:olive"}
    marker = {"3d":"o","4d":"x","3ds":"x","4ds":"^"}
    if len(sys.argv) > 4:
        pt = sys.argv[4]
        if pt == "mlef":
            perts = perts + [pt+"be", pt+"bm", pt+"cw",pt+"y",
            "4d"+pt, "4d"+pt+"be", "4d"+pt+"bm", "4d"+pt+"cw", "4d"+pt+"y"]
            linecolor.update({pt:'tab:blue',pt+"be":'tab:red',pt+"bm":'tab:pink',\
                pt+"cw":'tab:green',pt+"y":'tab:orange'})
        else:
            perts = perts + [pt, pt+"be", pt+"bm", "l"+pt]
            linecolor.update({pt:'tab:blue',pt+"be":'tab:orange',pt+"bm":'tab:green',"l"+pt:'tab:red'})
        perts = list(set(perts))
        #perts = [pt, pt+"be", pt+"bm", "l"+pt]
        #linecolor = {pt:'tab:blue',pt+"be":'tab:orange',pt+"bm":'tab:green',"l"+pt:'tab:red'}
        perts = [pt, pt+"be", pt+"bm", "l"+pt+"0", "l"+pt+"1", "l"+pt+"2", 'letkf']
        linecolor = {pt:'tab:blue',pt+"be":'tab:orange',pt+"bm":'tab:green',
        "l"+pt+"0":'tab:cyan', "l"+pt+"1":'tab:pink', "l"+pt+"2":'tab:purple', 'letkf':'tab:red'}
        #perts = ["l"+pt+"1", "l"+pt+"2", "l"+pt+"3", 'letkf']
        #linecolor = {"l"+pt+"1":'tab:blue', "l"+pt+"2":'tab:orange', "l"+pt+"3":'tab:green', 'letkf':'tab:red'}
    #sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    #"quadratic-nodiff": 1.0, "cubic-nodiff": 1.0, "test":1.0}
    sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
    x = np.arange(na) + 1
elif model == "qg":
    perts = ["letkf", "mlef", "envar", "etkf", "po", "srf",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dmlef"]
    linecolor = {"mlef":'tab:blue',"envar":'tab:orange',"etkf":'tab:green', "po":'tab:red',\
        "srf":"tab:pink", "letkf":"tab:purple",\
        "4dmlef":'tab:blue',"4detkf":'tab:green', "4dpo":'tab:red',\
        "4dsrf":"tab:pink", "4dletkf":"tab:purple"}
    sigma = {"linear": 4.0}
    x = np.arange(na) + 1
y = np.ones(x.size) * sigma[op]
if model == "qg":
    fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(12,10),constrained_layout=True)
    figf, axf = plt.subplots(nrows=2,ncols=1,figsize=(12,10),constrained_layout=True)
    fig2, ax2 = plt.subplots(nrows=2,ncols=1,figsize=(12,10),constrained_layout=True)
    figf2, axf2 = plt.subplots(nrows=2,ncols=1,figsize=(12,10),constrained_layout=True)
else:
    fig, ax = plt.subplots(figsize=(10,5),constrained_layout=True)
    figf, axf = plt.subplots(figsize=(10,5),constrained_layout=True)
    fig2, ax2 = plt.subplots(figsize=(10,5),constrained_layout=True)
    figf2, axf2 = plt.subplots(figsize=(10,5),constrained_layout=True)
#ax2 = ax.twinx()
i = 0
f = "enda_{}.txt".format(op)
try:
    e = np.loadtxt(f)
    if np.isnan(e).any():
        print("divergence in NoDA")
    else:
        print("NoDA, mean RMSE = {}".format(np.mean(e[int(na/3):])))
        ax.plot(x, e, linestyle='dotted', color='gray', label='NoDA')
except OSError or FileNotFoundError:
    print("not exist {}".format(f))
for pt in perts:
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
        print("{}, analysis RMSE = {}".format(pt,np.mean(e[int(na/3):,1])))
    else:
        print("{}, analysis RMSE = {}".format(pt,np.mean(e[int(na/3):])))
    #ax.plot(x, e, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    if model == "qg":
        if pt[:2] != "4d":
            for i in range(2):
                ax[i].plot(x, e[:,i], linestyle="solid", color=linecolor[pt], label=pt)
        else:
            for i in range(2):
                ax[i].plot(x, e[:,i], linestyle="dashed", color=linecolor[pt], label=pt)
    else:
        if pt[:2] != "4d":
            ax.plot(x, e, marker=marker["3d"], color=linecolor[pt], label=pt)
        else:
            ax.plot(x, e, marker=marker["4d"], color=linecolor[pt[2:]], label=pt)
#    if pt!="kf" and pt!="var" and pt!="4dvar":
    f = "stda_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    stda = np.loadtxt(f)
    if model == "qg":
        stda = stda.reshape(-1,2)
        for i in range(2):
            ax[i].plot(x, stda[:,i], linestyle="dashed", color=linecolor[pt], label=pt+" stdv.")
    else:
        if pt[:2] != "4d":
            ax.plot(x, stda, linestyle="dashed", marker=marker["3ds"], color=linecolor[pt], label=pt+" stdv.")
        else:
            ax.plot(x, stda, linestyle="dashed", marker=marker["4ds"], color=linecolor[pt[2:]], label=pt+" stdv.")
    if model == "qg":
        if e.shape[1] > na:
            for i in range(2):
                if pt[:2] != "4d":
                    ax2[i].plot(x[1:], stda[:,i]/e[1:,i], marker=marker["3d"], color=linecolor[pt], label=pt)
                else:
                    ax2[i].plot(x[1:], stda[:,i]/e[1:,i], marker=marker["4d"], color=linecolor[pt[2:]], label=pt)
        else:
            for i in range(2):
                if pt[:2] != "4d":
                    ax2[i].plot(x, stda[:,i]/e[:,i], marker=marker["3d"], color=linecolor[pt], label=pt)
                else:
                    ax2[i].plot(x, stda[:,i]/e[:,i], marker=marker["4d"], color=linecolor[pt[2:]], label=pt)
    else:
        if e.size > na:
            if pt[:2] != "4d":
                ax2.plot(x[1:], stda/e[1:], marker=marker["3d"], color=linecolor[pt], label=pt)
            else:
                ax2.plot(x[1:], stda/e[1:], marker=marker["4d"], color=linecolor[pt[2:]], label=pt)
        else:
            if pt[:2] != "4d":
                ax2.plot(x, stda/e, marker=marker["3d"], color=linecolor[pt], label=pt)
            else:
                ax2.plot(x, stda/e, marker=marker["4d"], color=linecolor[pt[2:]], label=pt)
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
        print("{}, forecast RMSE = {}".format(pt,np.mean(ef[int(na/3):,1])))
    else:
        print("{}, forecast RMSE = {}".format(pt,np.mean(ef[int(na/3):])))
    #ax.plot(x, e, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    if model == "qg":
        if pt[:2] != "4d":
            for i in range(2):
                axf[i].plot(x, ef[:,i], linestyle="solid", color=linecolor[pt], label=pt)
        else:
            for i in range(2):
                axf[i].plot(x, ef[:,i], linestyle="dashed", color=linecolor[pt], label=pt)
    else:
        if pt[:2] != "4d":
            axf.plot(x, ef, marker=marker["3d"], color=linecolor[pt], label=pt)
        else:
            axf.plot(x, ef, marker=marker["4d"], color=linecolor[pt[2:]], label=pt)
#    if pt!="kf" and pt!="var" and pt!="4dvar":
    f = "stdf_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    stdf = np.loadtxt(f)
    if model == "qg":
        stdf = stdf.reshape(-1,2)
        for i in range(2):
            axf[i].plot(x, stdf[:,i], linestyle="dashed", color=linecolor[pt], label=pt+" stdv.")
    else:
        if pt[:2] != "4d":
            axf.plot(x, stdf, linestyle="dashed", marker=marker["3ds"], color=linecolor[pt], label=pt+" stdv.")
        else:
            axf.plot(x, stdf, linestyle="dashed", marker=marker["4ds"], color=linecolor[pt[2:]], label=pt+" stdv.")
    if model == "qg":
        if ef.shape[1] > na:
            for i in range(2):
                if pt[:2] != "4d":
                    axf2[i].plot(x[1:], stdf[:,i]/ef[1:,i], marker=marker["3d"], color=linecolor[pt], label=pt)
                else:
                    axf2[i].plot(x[1:], stdf[:,i]/ef[1:,i], marker=marker["4d"], color=linecolor[pt[2:]], label=pt)
        else:
            for i in range(2):
                if pt[:2] != "4d":
                    axf2[i].plot(x, stdf[:,i]/ef[:,i], marker=marker["3d"], color=linecolor[pt], label=pt)
                else:
                    axf2[i].plot(x, stdf[:,i]/ef[:,i], marker=marker["4d"], color=linecolor[pt[2:]], label=pt)
    else:
        if ef.size > na:
            if pt[:2] != "4d":
                axf2.plot(x[1:], stdf/ef[1:], marker=marker["3d"], color=linecolor[pt], label=pt)
            else:
                axf2.plot(x[1:], stdf/ef[1:], marker=marker["4d"], color=linecolor[pt[2:]], label=pt)
        else:
            if pt[:2] != "4d":
                axf2.plot(x, stdf/ef, marker=marker["3d"], color=linecolor[pt], label=pt)
            else:
                axf2.plot(x, stdf/ef, marker=marker["4d"], color=linecolor[pt[2:]], label=pt)
    #f = "{}_e_{}-nodiff_{}.txt".format(model, op, pt)
    #if not os.path.isfile(f):
    #    print("not exist {}".format(f))
    #    continue
    #e = np.loadtxt(f)
    #if np.isnan(e).any():
    #    continue
    #ax.plot(x, e, linestyle="dashed", color=linecolor[pt], label="{}-nodiff".format(pt))
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
    if model=="z08":
        #ax.set_ylim(-0.01,0.2)
        ax.set_yscale("log")
        axf.set_yscale("log")
    if model=="tc87":
        ax.set_ylim(-0.01,2.0)
        axf.set_ylim(-0.01,2.0)
    #elif model=="l96":
    #    ax.set_ylim(-0.01,10.0)
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
    ax.legend()
    ax2.legend()
    axf.legend()
    axf2.legend()
fig.savefig("{}_e_{}.png".format(model, op))
fig2.savefig("{}_e+stda_{}.png".format(model, op))
figf.savefig("{}_ef_{}.png".format(model, op))
figf2.savefig("{}_ef+stdf_{}.png".format(model, op))
#fig.savefig("{}_e_{}+nodiff.pdf".format(model, op))
