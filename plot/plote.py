import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
if model == "z08":
    #perts = ["mlef", "grad", "etkf", "po", "srf", "letkf", "kf", "var"]
    perts = ["mlef-fh", "mlef-jh", "etkf-fh", "etkf-jh", "var"]
    linecolor = {"mlef-fh":'tab:blue',"mlef-jh":'tab:orange',"etkf-fh":'tab:green',"etkf-jh":'tab:red',
     "var":"tab:cyan"}
    #linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf":'tab:green', "po":'tab:red',\
    #    "srf":"tab:pink", "letkf":"tab:purple", "kf":"tab:cyan", "var":"tab:olive",\
    #    "var4d":"tab:brown"}
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
    perts = ["mlef", "etkf", "po", "srf", "letkf", "kf", "var",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf":'tab:green', "po":'tab:red',\
        "srf":"tab:pink", "letkf":"tab:purple", "kf":"tab:cyan", "var":"tab:olive",\
        "4dmlef":'tab:blue',"4detkf":'tab:green', "4dpo":'tab:red',\
        "4dsrf":"tab:pink", "4dletkf":"tab:purple","4dvar":"tab:olive"}
    if len(sys.argv) > 4:
        pt = sys.argv[4]
        perts = [pt, pt+"be", pt+"bm", "l"+pt]
        linecolor = {pt:'tab:blue',pt+"be":'tab:orange',pt+"bm":'tab:green',"l"+pt:'tab:red'}
    #sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    #"quadratic-nodiff": 1.0, "cubic-nodiff": 1.0, "test":1.0}
    sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
    x = np.arange(na) + 1
y = np.ones(x.size) * sigma[op]
fig, ax = plt.subplots(figsize=(10,5))
fig2, ax2 = plt.subplots(figsize=(10,5))
#ax2 = ax.twinx()
i = 0
for pt in perts:
    f = "{}_e_{}_{}.txt".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    e = np.loadtxt(f)
    if np.isnan(e).any():
        print("divergence in {}".format(pt))
        continue
    print("{}, mean RMSE = {}".format(pt,np.mean(e[int(na/3):])))
    #ax.plot(x, e, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    if pt[:2] != "4d":
        ax.plot(x, e, linestyle="solid", color=linecolor[pt], label=pt)
    else:
        ax.plot(x, e, linestyle="dashed", color=linecolor[pt], label=pt)
    f = "{}_pa_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    stda = np.zeros(na)
    pa = np.load(f)
    for i in range(na):
        stda[i] = np.sqrt(np.trace(pa[i]))
    if e.size > na:
        if pt[:2] != "4d":
            ax2.plot(x[1:], stda/e[1:], linestyle="solid", color=linecolor[pt], label=pt)
        else:
            ax2.plot(x[1:], stda/e[1:], linestyle="dashed", color=linecolor[pt], label=pt)
    else:
        ax.plot(x, stda, linestyle="dashed", color=linecolor[pt], label=pt+" stdv.")
        if pt[:2] != "4d":
            ax2.plot(x, stda/e, linestyle="solid", color=linecolor[pt], label=pt)
        else:
            ax2.plot(x, stda/e, linestyle="dashed", color=linecolor[pt], label=pt)
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
ax.plot(x, y, linestyle=(0, (5, 10)), color='black')
ax.set(xlabel="analysis cycle", ylabel="RMSE",
        title=op)
ax2.set(xlabel="analysis cycle", ylabel="Pa/RMSE",
        title=op)
if model=="z08":
    #ax.set_ylim(-0.01,0.2)
    ax.set_yscale("log")
if model=="tc87":
    ax.set_ylim(-0.01,2.0)
#elif model=="l96":
#    ax.set_ylim(-0.01,10.0)
if len(x) > 50:
    ax.set_xticks(x[::len(x)//10])
    ax.set_xticks(x[::len(x)//20], minor=True)
    ax2.set_xticks(x[::len(x)//10])
    ax2.set_xticks(x[::len(x)//20], minor=True)
else:
    ax.set_xticks(x[::5])
    ax.set_xticks(x, minor=True)
    ax2.set_xticks(x[::5])
    ax2.set_xticks(x, minor=True)
ax2.set_yscale("log")
ax.legend()
ax2.legend()
fig.savefig("{}_e_{}.png".format(model, op))
fig2.savefig("{}_e+pa_{}.png".format(model, op))
#fig.savefig("{}_e_{}+nodiff.pdf".format(model, op))
