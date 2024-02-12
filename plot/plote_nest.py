import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "envar", "envar_nest",\
    "etkf", "po", "srf", "letkf", "kf", "var","var_nest",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
linecolor = {"mlef":'tab:blue',"envar":'tab:orange',"envar_nest":'tab:green',"etkf":'tab:green', "po":'tab:red',\
        "srf":"tab:pink", "letkf":"tab:purple", "kf":"tab:cyan", "var":"tab:olive","var_nest":"tab:brown",\
        "mlefcw":"tab:green","mlefy":"tab:orange","mlefbe":"tab:red","mlefbm":"tab:pink"}
marker = {"3d":"o","4d":"s","3ds":"x","4ds":"^"}
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
x = np.arange(na) + 1
y = np.ones(x.size) * sigma[op]
fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(12,10),constrained_layout=True)
fig2, ax2 = plt.subplots(nrows=2,ncols=1,figsize=(12,10),constrained_layout=True)
figf, axf = plt.subplots(nrows=2,ncols=1,figsize=(12,10),constrained_layout=True)
figf2, axf2 = plt.subplots(nrows=2,ncols=1,figsize=(12,10),constrained_layout=True)
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
    #GM
    f = "e_gm_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    e_gm = np.loadtxt(f)
    if np.isnan(e_gm).any():
        print("divergence in {}".format(pt))
    #    continue
    print("{}, GM analysis RMSE = {}".format(pt,np.mean(e_gm[int(na/3):])))
    f = "stda_gm_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    stda_gm = np.loadtxt(f)
    #LAM
    f = "e_lam_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    e_lam = np.loadtxt(f)
    if np.isnan(e_lam).any():
        print("divergence in {}".format(pt))
    #    continue
    print("{}, LAM analysis RMSE = {}".format(pt,np.mean(e_lam[int(na/3):])))
    f = "stda_lam_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    stda_lam = np.loadtxt(f)
    if pt[:2] != "4d":
        ax[0].plot(x, e_gm, linestyle="solid", marker=marker["3d"], color=linecolor[pt], label=pt)
        ax[1].plot(x, e_lam, linestyle="solid", marker=marker["3d"], color=linecolor[pt], label=pt)
        #if pt != "kf" and pt != "var" and pt != "var_nest":
        ax[0].plot(x, stda_gm, linestyle="dashed", marker=marker["3ds"], color=linecolor[pt])
        ax[1].plot(x, stda_lam, linestyle="dashed", marker=marker["3ds"], color=linecolor[pt])
        ax2[0].plot(x, stda_gm/e_gm, marker=marker["3d"], color=linecolor[pt], label=pt)
        ax2[1].plot(x, stda_lam/e_lam, marker=marker["3d"], color=linecolor[pt], label=pt)
    else:
        ax[0].plot(x, e_gm, linestyle="solid", marker=marker["4d"], color=linecolor[pt], label=pt)
        ax[1].plot(x, e_lam, linestyle="solid", marker=marker["4d"], color=linecolor[pt], label=pt)
        #if pt != "4dvar":
        ax[0].plot(x, stda_gm, linestyle="dashed", marker=marker["4ds"], color=linecolor[pt])
        ax[1].plot(x, stda_lam, linestyle="dashed", marker=marker["4ds"], color=linecolor[pt])
        ax2[0].plot(x, stda_gm/e_gm, marker=marker["4d"], color=linecolor[pt], label=pt)
        ax2[1].plot(x, stda_lam/e_lam, marker=marker["4d"], color=linecolor[pt], label=pt)
    ## forecast
    #GM
    f = "ef_gm_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    ef_gm = np.loadtxt(f)
    if np.isnan(ef_gm).any():
        print("divergence in {}".format(pt))
    #    continue
    print("{}, GM forecast RMSE = {}".format(pt,np.mean(ef_gm[int(na/3):])))
    f = "stdf_gm_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    stdf_gm = np.loadtxt(f)
    #LAM
    f = "ef_lam_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    ef_lam = np.loadtxt(f)
    if np.isnan(ef_lam).any():
        print("divergence in {}".format(pt))
    #    continue
    print("{}, LAM forecast RMSE = {}".format(pt,np.mean(ef_lam[int(na/3):])))
    f = "stdf_lam_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    stdf_lam = np.loadtxt(f)
    if pt[:2] != "4d":
        axf[0].plot(x, ef_gm, linestyle="solid", marker=marker["3d"], color=linecolor[pt], label=pt)
        axf[1].plot(x, ef_lam, linestyle="solid", marker=marker["3d"], color=linecolor[pt], label=pt)
        #if pt != "kf" and pt != "var" and pt != "var_nest":
        axf[0].plot(x, stdf_gm, linestyle="dashed", marker=marker["3ds"], color=linecolor[pt])
        axf[1].plot(x, stdf_lam, linestyle="dashed", marker=marker["3ds"], color=linecolor[pt])
        axf2[0].plot(x, stdf_gm/ef_gm, marker=marker["3d"], color=linecolor[pt], label=pt)
        axf2[1].plot(x, stdf_lam/ef_lam, marker=marker["3d"], color=linecolor[pt], label=pt)
    else:
        axf[0].plot(x, ef_gm, linestyle="solid", marker=marker["4d"], color=linecolor[pt], label=pt)
        axf[1].plot(x, ef_lam, linestyle="solid", marker=marker["4d"], color=linecolor[pt], label=pt)
        #if pt != "4dvar":
        axf[0].plot(x, stdf_gm, linestyle="dashed", marker=marker["4ds"], color=linecolor[pt])
        axf[1].plot(x, stdf_lam, linestyle="dashed", marker=marker["4ds"], color=linecolor[pt])
        axf2[0].plot(x, stdf_gm/ef_gm, marker=marker["4d"], color=linecolor[pt], label=pt)
        axf2[1].plot(x, stdf_lam/ef_lam, marker=marker["4d"], color=linecolor[pt], label=pt)
# observation error (loosely dashed)
ax[0].plot(x, y, linestyle=(0, (5, 10)), color='black')
ax[1].plot(x, y, linestyle=(0, (5, 10)), color='black')
ax[0].set(xlabel="analysis cycle", ylabel="RMSE or STDV",
        title=op+" GM")
ax[1].set(xlabel="analysis cycle", ylabel="RMSE or STDV",
        title=op+" LAM")
ax2[0].set(xlabel="analysis cycle", ylabel="STDV/RMSE",
        title=op+" GM")
ax2[1].set(xlabel="analysis cycle", ylabel="STDV/RMSE",
        title=op+" LAM")
for i in range(2):
        if len(x) > 50:
            ax[i].set_xticks(x[::len(x)//10])
            ax[i].set_xticks(x[::len(x)//20], minor=True)
            ax2[i].set_xticks(x[::len(x)//10])
            ax2[i].set_xticks(x[::len(x)//20], minor=True)
        else:
            ax[i].set_xticks(x[::5])
            ax[i].set_xticks(x, minor=True)
            ax2[i].set_xticks(x[::5])
            ax2[i].set_xticks(x, minor=True)
        #ax2.set_yscale("log")
        ax[i].legend()
        ax2[i].legend()
fig.savefig("{}_e_{}.png".format(model, op))
fig2.savefig("{}_e+stda_{}.png".format(model, op))
#fig.savefig("{}_e_{}+nodiff.pdf".format(model, op))
# observation error (loosely dashed)
axf[0].plot(x, y, linestyle=(0, (5, 10)), color='black')
axf[1].plot(x, y, linestyle=(0, (5, 10)), color='black')
axf[0].set(xlabel="forecast cycle", ylabel="RMSE or STDV",
        title=op+" GM")
axf[1].set(xlabel="forecast cycle", ylabel="RMSE or STDV",
        title=op+" LAM")
axf2[0].set(xlabel="forecast cycle", ylabel="STDV/RMSE",
        title=op+" GM")
axf2[1].set(xlabel="forecast cycle", ylabel="STDV/RMSE",
        title=op+" LAM")
for i in range(2):
        if len(x) > 50:
            axf[i].set_xticks(x[::len(x)//10])
            axf[i].set_xticks(x[::len(x)//20], minor=True)
            axf2[i].set_xticks(x[::len(x)//10])
            axf2[i].set_xticks(x[::len(x)//20], minor=True)
        else:
            axf[i].set_xticks(x[::5])
            axf[i].set_xticks(x, minor=True)
            axf2[i].set_xticks(x[::5])
            axf2[i].set_xticks(x, minor=True)
        #ax2.set_yscale("log")
        axf[i].legend()
        axf2[i].legend()
figf.savefig("{}_ef_{}.png".format(model, op))
figf2.savefig("{}_ef+stdf_{}.png".format(model, op))