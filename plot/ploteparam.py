import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
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
else:
    perts = ["mlef", "etkf", "po", "srf", "letkf", "kf", "var",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf":'tab:green', "po":'tab:red',\
        "srf":"tab:pink", "letkf":"tab:purple", "kf":"tab:cyan", "var":"tab:olive",\
        "mlefcw":"tab:green","mlefy":"tab:orange","mlefbe":"tab:red","mlefbm":"tab:pink"}
    marker = {"3d":"o","4d":"x"}
    #sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    #"quadratic-nodiff": 1.0, "cubic-nodiff": 1.0, "test":1.0}
    sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0}
    x = np.arange(na) + 1
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
            if ptype=="infl" or ptype=="sigb" or ptype=="lb":
                var.append(float(tmp))
            else:
                var.append(int(tmp))
except FileNotFoundError:
    print("not found params.txt")
    pass
#y = np.ones(len(var)) * sigma[op]
fig, ax = plt.subplots(constrained_layout=True)
methods = []
nsuccess = []
mean = []
std = []
for pt in perts:
    #fig, ax = plt.subplots()
    i = 0
    j = 0
    success = np.zeros(len(var))
    el = np.zeros(len(var))
    es = np.zeros(len(var))
    for ivar in var:
    #f = "{}_e_{}_{}_{}.txt".format(model, op, pt, int(ivar))
        f = "{}_e_{}_{}_{}.txt".format(model, op, pt, ivar)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            el[i] = np.nan
            es[i] = np.nan
        else:
            data = np.loadtxt(f)
            e = data[:,1:]
            if np.isnan(e).any():
                print("divergence in {}".format(pt))
                el[i] = np.nan
                es[i] = np.nan
            else:
                success[i] = data[0,0]
                el[i] = np.mean(e[0,int(na/3):])
                es[i] = np.mean(e[1,int(na/3):])
                j += 1
        i+=1
    #ax.plot(x, e, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    if j > 0:
        methods.append(pt)
        nsuccess.append(success)
        mean.append(el)
        std.append(es)
xaxis = np.arange(len(var)) - len(methods)*0.025
i=0
for pt in methods:
    ns = nsuccess[i]
    el = mean[i]
    es = std[i]
    if pt[:2] == "4d":
        mark=marker["4d"]; color=linecolor[pt[2:]]
    else:
        mark=marker["3d"]; color=linecolor[pt]
    ax.errorbar(xaxis, el, yerr=es, marker=mark, color=color, label=pt)
    for j in range(ns.size):
        ax.text(xaxis[j], 0.93, f'{int(ns[j]):d}',\
        transform=ax.get_xaxis_transform(),\
        ha='center',fontsize=16,c='r')
    xaxis += 0.05
    i+=1
ax.set(xlabel="{} parameter".format(ptype), ylabel="RMSE",
            title=op)
ax.set_xticks(np.arange(len(var)))
ax.set_xticklabels(var)
if len(methods) > 1:
    ax.legend(loc='upper left')
        #fig.savefig("{}_e{}_{}_{}.png".format(model, ptype, op, pt))
fig.savefig("{}_e{}_{}.png".format(model, ptype, op))
#fig.savefig("{}_e_{}+nodiff.pdf".format(model, op))
