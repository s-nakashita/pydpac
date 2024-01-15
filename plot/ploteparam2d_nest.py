import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from plot_heatmap import heatmap, annotate_heatmap
import collections
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "etkf", "po", "srf", "letkf", "kf", "var", "var_nest", \
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
try:
    with open("params.txt","r") as f:
        ptype=f.readline()[:-1]
        vargm=[]
        varlam=[]
        while(True):
            tmp=f.readline()[:-1]
            if tmp=='': break
            tmp2 = tmp.split()
            vargm.append(tmp2[0])
            varlam.append(tmp2[1])
#            var.append(f"g{tmp2[0]}l{tmp2[1]}")
except FileNotFoundError:
    print("not found params.txt")
    exit()
## remove overlap
vargm = [k for k, v in collections.Counter(vargm).items() if v > 1]
varlam = [k for k, v in collections.Counter(varlam).items() if v > 1]
print(vargm)
print(varlam)
#y = np.ones(len(var)) * sigma[op]
nvar_gm = len(vargm)
nvar_lam = len(varlam)
methods = []
nsuccess_gm = []
mean_gm = []
std_gm = []
nsuccess_lam = []
mean_lam = []
std_lam = []
for pt in perts:
    i = 0
    ns = 0
    success_gm = np.zeros((nvar_gm,nvar_lam))
    el_gm = np.zeros((nvar_gm,nvar_lam))
    es_gm = np.zeros((nvar_gm,nvar_lam))
    success_lam = np.zeros((nvar_gm,nvar_lam))
    el_lam = np.zeros((nvar_gm,nvar_lam))
    es_lam = np.zeros((nvar_gm,nvar_lam))
    for ivargm in vargm:
        j = 0
        for ivarlam in varlam:
            ivar=f"g{ivargm}l{ivarlam}"
            ## GM
            f = "{}_e_gm_{}_{}_{}.txt".format(model, op, pt, ivar)
            if not os.path.isfile(f):
                print("not exist {}".format(f))
                el_gm[i,j] = np.nan
                es_gm[i,j] = np.nan
            else:
                data = np.loadtxt(f)
                e = data[:,1:]
                if np.isnan(e).any():
                    print("divergence in {}".format(pt))
                    el_gm[i,j] = np.nan
                    es_gm[i,j] = np.nan
                else:
                    success_gm[i,j] = data[0,0]
                    el_gm[i,j] = np.mean(e[0,int(na/3):])
                    es_gm[i,j] = np.mean(e[1,int(na/3):])
                    ns += 1
            ## LAM
            f = "{}_e_lam_{}_{}_{}.txt".format(model, op, pt, ivar)
            if not os.path.isfile(f):
                print("not exist {}".format(f))
                el_lam[i,j] = np.nan
                es_lam[i,j] = np.nan
            else:
                data = np.loadtxt(f)
                e = data[:,1:]
                if np.isnan(e).any():
                    print("divergence in {}".format(pt))
                    el_lam[i,j] = np.nan
                    es_lam[i,j] = np.nan
                else:
                    success_lam[i,j] = data[0,0]
                    el_lam[i,j] = np.mean(e[0,int(na/3):])
                    es_lam[i,j] = np.mean(e[1,int(na/3):])
            j += 1
        i += 1
    #ax.plot(x, e, linestyle=linestyle[pt], color=linecolor[pt], label=pt)
    if ns > 0:
        methods.append(pt)
        nsuccess_gm.append(success_gm)
        mean_gm.append(el_gm)
        std_gm.append(es_gm)
        nsuccess_lam.append(success_lam)
        mean_lam.append(el_lam)
        std_lam.append(es_lam)
i=0
for pt in methods:
    fig, axs = plt.subplots(figsize=[12,6],ncols=2,constrained_layout=True)
    ns_gm = nsuccess_gm[i]
    el_gm = mean_gm[i]
    es_gm = std_gm[i]
    ns_lam = nsuccess_lam[i]
    el_lam = mean_lam[i]
    es_lam = std_lam[i]
    #if pt[:2] == "4d":
    #    mark=marker["4d"]; color=linecolor[pt[2:]]
    #else:
    #    mark=marker["3d"]; color=linecolor[pt]
    im0, cbar0 = heatmap(el_gm, vargm, varlam, ax=axs[0],\
        cmap='PiYG_r',cbarlabel="averaged RMSE in GM")
    txt0 = annotate_heatmap(im0, valfmt="{x:.2f}")
    #axs[0].errorbar(xaxis, el_gm, yerr=es_gm, marker=mark, color=color, label=pt)
    #for j in range(ns_gm.size):
    #    axs[0].text(xaxis[j], 0.93, f'{int(ns_gm[j]):d}',\
    #    transform=axs[0].get_xaxis_transform(),\
    #    ha='center',fontsize=16,c='r')
    im1, cbar1 = heatmap(el_lam, vargm, varlam, ax=axs[1],\
        cmap='PiYG_r',cbarlabel="averaged RMSE in LAM")
    txt1 = annotate_heatmap(im1, valfmt="{x:.2f}")
    #axs[1].errorbar(xaxis, el_lam, yerr=es_lam, marker=mark, color=color, label=pt)
    #for j in range(ns_lam.size):
    #    axs[1].text(xaxis[j], 0.93, f'{int(ns_lam[j]):d}',\
    #    transform=axs[1].get_xaxis_transform(),\
    #    ha='center',fontsize=16,c='r')
    #xaxis += 0.05
    for ax in axs:
        ax.set(ylabel="{} GM".format(ptype),xlabel="{} LAM".format(ptype),title=op+", "+pt)
    fig.savefig("{}_e2d{}_{}_{}.png".format(model, ptype, op, pt))
    plt.show()
    i+=1
