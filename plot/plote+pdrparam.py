import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
from methods import perts, linecolor, iinflist, inflcolor, infltype, inflmarkers

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
nspinup = na//5
pt = sys.argv[4]
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
    marker = {"3d":"o","4d":"x"}
    #sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    #"quadratic-nodiff": 1.0, "cubic-nodiff": 1.0, "test":1.0}
    sigma = {"linear": 1.0, "quadratic": 8.0e-1, "cubic": 7.0e-2, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, "test":1.0}
    x = np.arange(na) + 1
if len(sys.argv)>5:
    ptype = sys.argv[5]
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

#y = np.ones(len(var)) * sigma[op]
methods = []
nsuccess = []
xaxes = []
emean = []
estd = []
pmean = []
pstd = []
#for pt in perts:
#for iinf in [3,4,5]:
iinf=-3
for nit in [2,3,5,10,20]:
    ddir = f"{pt}_infl{iinf}_nit{nit}_nmem"
    try:
        with open(f"{ddir}/params.txt","r") as f:
            ptype=f.readline()[:-1]
            var=[]
            while(True):
                tmp=f.readline()[:-1]
                if tmp=='': break
                if ptype=="infl" or ptype=="sigo" or ptype=="sigb" or ptype=="lb":
                    var.append(float(tmp))
                else:
                    var.append(int(tmp))
    except FileNotFoundError:
        print("not found params.txt")
        continue
    #fig, ax = plt.subplots()
    i = 0
    j = 0
    success = np.zeros(len(var))
    el = np.zeros(len(var))
    es = np.zeros(len(var))
    pl = np.zeros(len(var))
    ps = np.zeros(len(var))
    for ivar in var:
    #f = "{}_e_{}_{}_{}.txt".format(model, op, pt, int(ivar))
        f = f"{ddir}/{model}_e_{op}_{pt}_{ivar}.txt"
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
                el[i] = np.mean(e[0,nspinup:])
                es[i] = np.mean(e[1,nspinup:])
                j += 1
        f = f"{ddir}/{model}_pdr_{op}_{pt}_{ivar}.txt"
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            pl[i] = np.nan
            ps[i] = np.nan
        else:
            data = np.loadtxt(f)
            p = data[:,1:]
            if np.isnan(p).any():
                print("divergence in {}".format(pt))
                pl[i] = np.nan
                ps[i] = np.nan
            else:
                pl[i] = np.mean(p[0,nspinup:])
                ps[i] = np.mean(p[1,nspinup:])
                j += 1
        i+=1
    #ax.plot(x, e, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    if j > 0:
        methods.append(nit)
        nsuccess.append(success)
        xaxes.append(np.arange(len(var)))
        emean.append(el)
        estd.append(es)
        pmean.append(pl)
        pstd.append(ps)
# optimal rtps
i = 0
j = 0
success = np.zeros(len(var))
el = np.zeros(len(var))
es = np.zeros(len(var))
pl = np.zeros(len(var))
ps = np.zeros(len(var))
optparams = {16:0.7,20:0.5,24:0.4,28:0.3,32:0.3,36:0.3,40:0.2}
for ivar in var:
    ddir = f"{pt}_nmem{ivar}_infl3"
    #f = "{}_e_{}_{}_{}.txt".format(model, op, pt, int(ivar))
    f = f"{ddir}/{model}_e_{op}_{pt}_{optparams[ivar]}.txt"
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
                el[i] = np.mean(e[0,nspinup:])
                es[i] = np.mean(e[1,nspinup:])
                j += 1
    f = f"{ddir}/{model}_pdr_{op}_{pt}_{optparams[ivar]}.txt"
    if not os.path.isfile(f):
            print("not exist {}".format(f))
            pl[i] = np.nan
            ps[i] = np.nan
    else:
            data = np.loadtxt(f)
            p = data[:,1:]
            if np.isnan(p).any():
                print("divergence in {}".format(pt))
                pl[i] = np.nan
                ps[i] = np.nan
            else:
                pl[i] = np.mean(p[0,nspinup:])
                ps[i] = np.mean(p[1,nspinup:])
                j += 1
    i+=1
if j > 0:
    methods.append('rtps')
    nsuccess.append(success)
    xaxes.append(np.arange(len(var)))
    emean.append(el)
    estd.append(es)
    pmean.append(pl)
    pstd.append(ps)
#xaxis = np.arange(len(var)) - len(methods)*0.025
i=0
fig, axs = plt.subplots(nrows=2,sharex=True,figsize=[8,8],constrained_layout=True)
colors = {2:'b',3:'r',5:'g',10:'orange',20:'brown'}
marks = {2:'o',3:'s',5:'^',10:'*',20:'v'}
#for iinf in methods:
for nit in methods:
    if nit=='rtps':
        label = nit
        c = 'tab:cyan'
        mark = 'd'
    else:
        label = infltype[iinf]+f' nit={nit}'
        c = colors[nit]
        mark = marks[nit]
    #label=infltype[iinf]
    #c = inflcolor[iinf]
    #mark = inflmarkers[iinf]
    xaxis = xaxes[i] #- len(methods)*0.025 + i*0.05
    ns = nsuccess[i]
    el = emean[i]
    es = estd[i]
    pl = pmean[i]
    ps = pstd[i]
    #axs[0].errorbar(xaxis, el, yerr=es, marker=mark, color=c, label=label)
    #axs[1].errorbar(xaxis, pl, yerr=ps, marker=mark, color=c, label=label)
    axs[0].plot(xaxis, el, marker=mark, color=c, label=label)
    axs[1].plot(xaxis, pl, marker=mark, color=c, label=label)
    #for j in range(ns.size):
    #    axs[0].text(xaxis[j], 0.93, f'{int(ns[j]):d}',\
    #    transform=axs[0].get_xaxis_transform(),\
    #    ha='center',fontsize=9,c='r')
    if nit=='rtps':
        for j in range(ns.size):
            axs[0].text(xaxis[j], el[j], f'{optparams[var[j]]:.1f}',\
            ha='center',va='top',fontsize=12,c=c)
            axs[1].text(xaxis[j], pl[j], f'{optparams[var[j]]:.1f}',\
            ha='center',va='top',fontsize=12,c=c)
    #xaxis += 0.05
    i+=1
axs[1].hlines([1],0,1,colors='k',transform=axs[1].get_yaxis_transform())
#ax.set(xlabel="{} parameter".format(ptype), ylabel="RMSE",
#            title=op)
axs[0].set_ylabel('RMSE')
axs[1].set_ylabel('PDR')
axs[1].set_xlabel('Number of ensemble members')
axs[1].set_xticks(np.arange(0,len(var)))
axs[1].set_xticklabels(var)
axs[0].set_ylim(0.16,0.51)
axs[1].set_ylim(0.85,1.15)
axs[0].grid()
axs[1].grid()
if len(methods) > 1:
    axs[0].legend(loc='upper right')
    #axs[1].legend(loc='upper right')
        #fig.savefig("{}_e{}_{}_{}.png".format(model, ptype, op, pt))
fig.savefig("{}_e+pdr{}_{}_{}.png".format(model, ptype, op, pt))
#fig.savefig("{}_e_{}+nodiff.pdf".format(model, op))
plt.show()
