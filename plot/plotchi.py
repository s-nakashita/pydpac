import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
#perts = ["etkf", "po", "srf", "letkf"]
perts = ["mlef", "grad", "etkf", "po", "srf", "letkf"]
if model == "l96":
    perts = ["mlef", "etkf", "po", "srf", "letkf", "kf", "var",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
    linecolor = {"mlef":'blue',"grad":'orange',"etkf":'green', "po":'red',\
        "srf":"pink", "letkf":"purple", "kf":"cyan", "var":"olive"}
    marker = {"3d":"o","4d":"x"}
    #na = 100
elif model == "burgers":
    perts = ["mlef", "grad", "etkf-fh", "etkf-jh"]#, "po", "srf", "letkf"]
    linestyle = {"mlef":"solid", "grad":"dashed",
     "etkf-fh":"solid", "etkf-jh":"dashed"}
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf-fh":'tab:green',"etkf-jh":'tab:red'}
elif model == "kdvb":
    perts = ["mlef", "etkf", "po", "srf", "letkf", "kf", "var"]
    linecolor = {"mlef":'blue',"grad":'orange',"etkf":'green', "po":'red',\
        "srf":"pink", "letkf":"purple", "kf":"cyan", "var":"olive"}
x = np.arange(na) + 1
y = np.ones(x.shape)
fig, ax = plt.subplots(figsize=(10,5))
for pt in perts:
    f = "{}_chi_{}_{}.txt".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    chi = np.loadtxt(f)
    chim = np.zeros_like(chi)
    if pt[:2] == "4d":
        mark=marker["4d"]
        color=linecolor[pt[2:]]
        ls="dashed"
    else:
        mark=marker["3d"]
        color=linecolor[pt]
        ls="solid"
    ax.scatter(x, chi, marker=mark, color="tab:"+color, label=pt)
    for k in range(chi.size):
        chim[k] = np.mean(chi[k:min(k+10,chi.size)])
    ax.plot(x, chim, linestyle=ls, color=color)
ax.plot(x, y, linestyle="dotted", color='black')
ax.set_yscale("log")
ax.set_ylim(1e-1, 10)
#if np.max(chi) > 1000.0:
#    ax.set_ylim(0.1, 1000.0)
#    ax.set_yticks([1,10,100,1000])
#if np.max(chi) > 10000.0:
#    ax.set_ylim(0.1, 10000.0)
#    ax.set_yticks([1,10,100,1000,10000])
ax.set(xlabel="analysis cycle", ylabel="Chi2",
        title=op)
ax.set_xticks(x[::len(x)//10])
ax.set_xticks(x[::len(x)//20], minor=True)
ax.legend()
fig.savefig("{}_chi_{}.png".format(model, op))
