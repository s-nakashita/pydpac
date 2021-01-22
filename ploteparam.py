import sys
import os
import numpy as np
import matplotlib.pyplot as plt

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
ptype = sys.argv[4]
if model == "z08" or model == "z05":
    perts = ["mlef", "grad", "etkf-fh", "etkf-jh"]#, "po", "srf", "letkf"]
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf-fh":'tab:green',"etkf-jh":'tab:red',
     "kf":"tab:cyan"}
    na += 1
    x = np.arange(na)
    #sigma = {"linear": 8.0e-2, "quadratic": 8.0e-2, "cubic": 7.0e-4, "quartic": 7.0e-4,\
    #"quadratic-nodiff": 8.0e-2, "cubic-nodiff": 7.0e-4, "quartic-nodiff": 7.0e-4}
    sigma = {"linear": 8.0e-2, "quadratic": 1.0e-3, "cubic": 1.0e-3, "quartic": 1.0e-2, \
    "quadratic-nodiff": 1.0e-3, "cubic-nodiff": 1.0e-3, "quartic-nodiff": 1.0e-2}
elif model == "l96":
    perts = ["mlef", "etkf", "po", "srf", "letkf"]
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf":'tab:green', "po":'tab:red',\
        "srf":"tab:pink", "letkf":"tab:purple", "kf":"tab:cyan", "var":"tab:olive",\
        "var4d":"tab:brown"    }
    #sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    #"quadratic-nodiff": 1.0, "cubic-nodiff": 1.0, "test":1.0}
    sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0}
    x = np.arange(na) + 1
if ptype == "loc":
    var = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
elif ptype == "infl":
    var = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
#y = np.ones(len(var)) * sigma[op]
#fig, ax = plt.subplots()
for pt in perts:
    fig, ax = plt.subplots()
    i = 0
    el = np.zeros(len(var))
    for ivar in var:
    #f = "{}_e_{}_{}_{}.txt".format(model, op, pt, int(ivar))
        f = "{}_e_{}_{}_{}.txt".format(model, op, pt, ivar)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            i = -1
            continue
        e = np.loadtxt(f)
        if np.isnan(e).any():
            print("divergence in {}".format(pt))
            el[i] = np.nan
            i += 1
            continue
        el[i] = np.mean(e[int(na/3):])
        i += 1
    #ax.plot(x, e, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    if i > 0:
        ax.plot(var, el, linestyle="solid", color=linecolor[pt], label=pt)
        ax.set(xlabel="{} parameter".format(ptype), ylabel="RMSE",
            title=op)
        ax.set_xticks(var)
#ax.legend()
        fig.savefig("{}_e{}_{}_{}.png".format(model, ptype, op, pt))
#fig.savefig("{}_e_{}+nodiff.pdf".format(model, op))
