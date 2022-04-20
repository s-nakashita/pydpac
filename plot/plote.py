import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
if model == "z08" or model == "z05":
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
elif model == "l96" or model == "tc87":
    perts = ["mlef", "mlefw", "etkf", "po", "srf", "letkf", "kf", "var",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
    linecolor = {"mlef":'tab:blue',"mlefw":'tab:orange',"etkf":'tab:green', "po":'tab:red',\
        "srf":"tab:pink", "letkf":"tab:purple", "kf":"tab:cyan", "var":"tab:olive",\
        "4dmlef":'tab:blue',"4detkf":'tab:green', "4dpo":'tab:red',\
        "4dsrf":"tab:pink", "4dletkf":"tab:purple","4dvar":"tab:olive"}
    if len(sys.argv) > 4:
        pt = sys.argv[4]
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
y = np.ones(x.size) * sigma[op]
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
#ax2 = ax.twinx()
i = 0
for pert in perts:
    for ltlm in range(2):
        if ltlm == 0:
            f = "{}_e_{}_{}.txt".format(model, op, pert)
            linestyle="solid"
        else:
            f = "{}_e_{}_{}_t.txt".format(model, op, pert)
            linestyle="dashed"
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        e = np.loadtxt(f)
        if np.isnan(e).any():
            print("divergence in {}".format(pert))
            continue
        print("{}, mean RMSE = {}".format(pert,np.mean(e[int(na/3):])))
    #ax.plot(x, e, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
        if pert[:2] != "4d" or ltlm == 0:
            ax.plot(x, e, linestyle=linestyle, color=linecolor[pert], label=pert)
        else:
            ax.plot(x, e, linestyle=linestyle, color=linecolor[pert], label=pert + ",TLM")
        if ltlm == 0:
            f = "{}_pa_{}_{}.npy".format(model, op, pert)
        else:
            f = "{}_pa_{}_{}_t.npy".format(model, op, pert)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        trpa = np.zeros(na)
        if (len(sys.argv)>4 and pt == "mlef") or pert == "mlef" or pert == "mlefw":
            spa = np.load(f)
            for i in range(na):
                pa = spa[i] @ spa[i].T
                trpa[i] = np.mean(np.diag(pa))
        else:
            pa = np.load(f)
            for i in range(na):
                trpa[i] = np.mean(np.diag(pa[i]))
        if e.size > na:
            if pert[:2] != "4d" or ltlm == 0:
                ax2.plot(x[1:], trpa/e[1:], linestyle=linestyle, color=linecolor[pert], label=pert)
            else:
                ax2.plot(x[1:], trpa/e[1:], linestyle=linestyle, color=linecolor[pert], label=pert + ",TLM")
        else:
            if pert[:2] != "4d" or ltlm == 0:
                ax2.plot(x, trpa/e, linestyle=linestyle, color=linecolor[pert], label=pert)
            else:
                ax2.plot(x, trpa/e, linestyle=linestyle, color=linecolor[pert], label=pert + ",TLM")
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
    ax.set_xticks(x[4::5])
    ax.set_xticks(x, minor=True)
    ax2.set_xticks(x[4::5])
    ax2.set_xticks(x, minor=True)
ax2.set_yscale("log")
ax.legend()
ax2.legend()
fig.savefig("{}_e_{}.png".format(model, op))
fig2.savefig("{}_e+pa_{}.png".format(model, op))
#fig.savefig("{}_e_{}+nodiff.pdf".format(model, op))
