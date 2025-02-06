import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
if model == "burgers" or model == "kdvb":
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
    perts = ["mlef", "envar", "envar_nest", "etkf", "po", "srf", "letkf", "kf", "var", "var_nest",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
    linecolor = {"mlef":'tab:blue',"envar":'tab:orange',"envar_nest":'tab:green',"etkf":'tab:green', "po":'tab:red',\
        "srf":"tab:pink", "letkf":"tab:purple", "kf":"tab:cyan", "var":"tab:olive","var_nest":"tab:brown",\
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
            if ptype=="infl" or ptype=="sigo"\
                or ptype=="sigb" or ptype=="sigv"\
                or ptype=="infl_lrg":
                var.append(float(tmp))
            #elif ptype=="sigb":
            #    tmp2 = tmp.split()
            #    var.append(f"g{tmp2[0]}l{tmp2[1]}")
            else:
                var.append(int(tmp))
except FileNotFoundError:
    print("not found params.txt")
    pass
#y = np.ones(len(var)) * sigma[op]
methods = []
nsuccess_gm = []
mean_gm = []
std_gm = []
nsuccess_lam = []
mean_lam = []
std_lam = []
for pt in perts:
    i = 0
    j = 0
    success_gm = np.zeros(len(var))
    el_gm = np.zeros(len(var))
    es_gm = np.zeros(len(var))
    success_lam = np.zeros(len(var))
    el_lam = np.zeros(len(var))
    es_lam = np.zeros(len(var))
    for ivar in var:
    #f = "{}_e_{}_{}_{}.txt".format(model, op, pt, int(ivar))
        ## GM
        f = "{}_e_gm_{}_{}_{}.txt".format(model, op, pt, ivar)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            el_gm[i] = np.nan
            es_gm[i] = np.nan
        else:
            data = np.loadtxt(f)
            e = data[:,1:]
            if np.isnan(e).any():
                print("divergence in {}".format(pt))
                el_gm[i] = np.nan
                es_gm[i] = np.nan
            else:
                success_gm[i] = data[0,0]
                el_gm[i] = np.mean(e[0,int(na/3):])
                es_gm[i] = np.mean(e[1,int(na/3):])
                j += 1
        ## LAM
        f = "{}_e_lam_{}_{}_{}.txt".format(model, op, pt, ivar)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            el_lam[i] = np.nan
            es_lam[i] = np.nan
        else:
            data = np.loadtxt(f)
            e = data[:,1:]
            if np.isnan(e).any():
                print("divergence in {}".format(pt))
                el_lam[i] = np.nan
                es_lam[i] = np.nan
            else:
                success_lam[i] = data[0,0]
                el_lam[i] = np.mean(e[0,int(na/3):])
                es_lam[i] = np.mean(e[1,int(na/3):])
        i += 1
    #ax.plot(x, e, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    if j > 0:
        methods.append(pt)
        nsuccess_gm.append(success_gm)
        mean_gm.append(el_gm)
        std_gm.append(es_gm)
        nsuccess_lam.append(success_lam)
        mean_lam.append(el_lam)
        std_lam.append(es_lam)
xaxis = np.arange(len(var)) - len(methods)*0.025
fig, axs = plt.subplots(figsize=[8,6],nrows=2,sharex=True,constrained_layout=True)
i=0
for pt in methods:
    ns_gm = nsuccess_gm[i]
    el_gm = mean_gm[i]
    es_gm = std_gm[i]
    ns_lam = nsuccess_lam[i]
    el_lam = mean_lam[i]
    es_lam = std_lam[i]
    print(f"{pt} min:{np.min(el_lam)} {ptype}={var[np.argmin(el_lam)]}")
    if pt[:2] == "4d":
        mark=marker["4d"]; color=linecolor[pt[2:]]
    else:
        mark=marker["3d"]; color=linecolor[pt]
    axs[0].errorbar(xaxis, el_gm, yerr=es_gm, marker=mark, color=color, label=pt)
    for j in range(ns_gm.size):
        axs[0].text(xaxis[j], 0.93, f'{int(ns_gm[j]):d}',\
        transform=axs[0].get_xaxis_transform(),\
        ha='center',fontsize=16,c='r')
    axs[1].errorbar(xaxis, el_lam, yerr=es_lam, marker=mark, color=color, label=pt)
    for j in range(ns_lam.size):
        axs[1].text(xaxis[j], 0.93, f'{int(ns_lam[j]):d}',\
        transform=axs[1].get_xaxis_transform(),\
        ha='center',fontsize=16,c='r')
    xaxis += 0.05
    i+=1
axs[0].set(ylabel="RMSE",title=op+", GM")
axs[1].set(xlabel="{} parameter".format(ptype),ylabel="RMSE",title=op+", LAM")
axs[1].set_xticks(np.arange(len(var)))
axs[1].set_xticklabels(var)
if len(methods) > 1:
    axs[0].legend(loc='upper left')
#plt.show()
fig.savefig("{}_e{}_{}.png".format(model, ptype, op))
