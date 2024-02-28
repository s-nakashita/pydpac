import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
spinup=True
if len(sys.argv) > 4:
    spinup=(sys.argv[4]=='T')
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
if not spinup:
    ns = na // 5
    x = np.arange(ns,na) + 1
else:
    ns = 0
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
        print("NoDA, mean RMSE = {}".format(np.mean(e[ns:])))
        ax.plot(x, e[ns:], linestyle='dotted', color='gray', label='NoDA')
except OSError or FileNotFoundError:
    print("not exist {}".format(f))
for pt in perts:
    ## analysis
    #GM
    f = "e_gm_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        #print("not exist {}".format(f))
        continue
    e_gm = np.loadtxt(f)
    if np.isnan(e_gm).any():
        print("divergence in {}".format(pt))
    #    continue
    print("{}, GM analysis RMSE = {}".format(pt,np.mean(e_gm[ns:])))
    f = "stda_gm_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        #print("not exist {}".format(f))
        continue
    stda_gm = np.loadtxt(f)
    #LAM
    f = "e_lam_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        #print("not exist {}".format(f))
        continue
    e_lam = np.loadtxt(f)
    if np.isnan(e_lam).any():
        print("divergence in {}".format(pt))
    #    continue
    print("{}, LAM analysis RMSE = {}".format(pt,np.mean(e_lam[ns:])))
    f = "stda_lam_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        #print("not exist {}".format(f))
        continue
    stda_lam = np.loadtxt(f)
    if pt[:2] != "4d":
        ax[0].plot(x, e_gm[ns:], \
            linestyle="solid", marker=marker["3d"], \
            color=linecolor[pt], label=pt+f', mean={np.mean(e_gm[ns:]):.4f}')
        ax[1].plot(x, e_lam[ns:], \
            linestyle="solid", marker=marker["3d"], \
            color=linecolor[pt], label=pt+f', mean={np.mean(e_lam[ns:]):.4f}')
        #if pt != "kf" and pt != "var" and pt != "var_nest":
        #ax[0].plot(x, stda_gm[ns:], linestyle="dashed", marker=marker["3ds"], color=linecolor[pt])
        #ax[1].plot(x, stda_lam[ns:], linestyle="dashed", marker=marker["3ds"], color=linecolor[pt])
        r_gm = stda_gm[ns:]/e_gm[ns:]
        r_lam = stda_lam[ns:]/e_lam[ns:]
        ax2[0].plot(x, r_gm, marker=marker["3d"], color=linecolor[pt], \
            label=pt+f', mean={np.mean(r_gm):.4f}')
        ax2[1].plot(x, r_lam, marker=marker["3d"], color=linecolor[pt], \
            label=pt+f', mean={np.mean(r_lam):.4f}')
    else:
        ax[0].plot(x, e_gm[ns:], \
            linestyle="solid", marker=marker["4d"], \
            color=linecolor[pt], label=pt+f', mean={np.mean(e_gm[ns:]):.4f}')
        ax[1].plot(x, e_lam[ns:], \
            linestyle="solid", marker=marker["4d"], \
            color=linecolor[pt], label=pt+f', mean={np.mean(e_lam[ns:]):.4f}')
        #if pt != "4dvar":
        #ax[0].plot(x, stda_gm[ns:], linestyle="dashed", marker=marker["4ds"], color=linecolor[pt])
        #ax[1].plot(x, stda_lam[ns:], linestyle="dashed", marker=marker["4ds"], color=linecolor[pt])
        r_gm = stda_gm[ns:]/e_gm[ns:]
        r_lam = stda_lam[ns:]/e_lam[ns:]
        ax2[0].plot(x, r_gm, marker=marker["4d"], color=linecolor[pt], \
            label=pt+f', mean={np.mean(r_gm):.4f}')
        ax2[1].plot(x, r_lam, marker=marker["4d"], color=linecolor[pt], \
            label=pt+f', mean={np.mean(r_lam):.4f}')
    ## forecast
    #GM
    f = "ef_gm_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        #print("not exist {}".format(f))
        continue
    ef_gm = np.loadtxt(f)
    if np.isnan(ef_gm).any():
        print("divergence in {}".format(pt))
    #    continue
    print("{}, GM forecast RMSE = {}".format(pt,np.mean(ef_gm[ns:])))
    f = "stdf_gm_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        #print("not exist {}".format(f))
        continue
    stdf_gm = np.loadtxt(f)
    #LAM
    f = "ef_lam_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        #print("not exist {}".format(f))
        continue
    ef_lam = np.loadtxt(f)
    if np.isnan(ef_lam).any():
        print("divergence in {}".format(pt))
    #    continue
    print("{}, LAM forecast RMSE = {}".format(pt,np.mean(ef_lam[ns:])))
    f = "stdf_lam_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        #print("not exist {}".format(f))
        continue
    stdf_lam = np.loadtxt(f)
    if pt[:2] != "4d":
        axf[0].plot(x, ef_gm[ns:], \
            linestyle="solid", marker=marker["3d"], \
            color=linecolor[pt], label=pt+f', mean={np.mean(ef_gm[ns:]):.4f}')
        axf[1].plot(x, ef_lam[ns:], \
            linestyle="solid", marker=marker["3d"], \
            color=linecolor[pt], label=pt+f', mean={np.mean(ef_lam[ns:]):.4f}')
        #if pt != "kf" and pt != "var" and pt != "var_nest":
        #axf[0].plot(x, stdf_gm[ns:], linestyle="dashed", marker=marker["3ds"], color=linecolor[pt])
        #axf[1].plot(x, stdf_lam[ns:], linestyle="dashed", marker=marker["3ds"], color=linecolor[pt])
        r_gm = stdf_gm[ns:]/ef_gm[ns:]
        r_lam = stdf_lam[ns:]/ef_lam[ns:]
        axf2[0].plot(x, r_gm, marker=marker["3d"], color=linecolor[pt], \
            label=pt+f', mean={np.mean(r_gm):.4f}')
        axf2[1].plot(x, r_lam, marker=marker["3d"], color=linecolor[pt], \
            label=pt+f', mean={np.mean(r_lam):.4f}')
    else:
        axf[0].plot(x, ef_gm[ns:], \
            linestyle="solid", marker=marker["4d"], \
            color=linecolor[pt], label=pt+f', mean={np.mean(ef_gm[ns:]):.4f}')
        axf[1].plot(x, ef_lam[ns:], \
            linestyle="solid", marker=marker["4d"], \
            color=linecolor[pt], label=pt+f', mean={np.mean(ef_lam[ns:]):.4f}')
        #if pt != "4dvar":
        #axf[0].plot(x, stdf_gm[ns:], linestyle="dashed", marker=marker["4ds"], color=linecolor[pt])
        #axf[1].plot(x, stdf_lam[ns:], linestyle="dashed", marker=marker["4ds"], color=linecolor[pt])
        r_gm = stdf_gm[ns:]/ef_gm[ns:]
        r_lam = stdf_lam[ns:]/ef_lam[ns:]
        axf2[0].plot(x, r_gm, marker=marker["4d"], color=linecolor[pt], \
            label=pt+f', mean={np.mean(r_gm):.4f}')
        axf2[1].plot(x, r_lam, marker=marker["4d"], color=linecolor[pt], \
            label=pt+f', mean={np.mean(r_lam):.4f}')
# observation error (loosely dashed)
ax[0].plot(x, y, linestyle=(0, (5, 10)), color='black')
ax[1].plot(x, y, linestyle=(0, (5, 10)), color='black')
ax2[0].hlines([1],0,1,colors='gray',transform=ax2[0].get_yaxis_transform(),zorder=0)
ax2[1].hlines([1],0,1,colors='gray',transform=ax2[1].get_yaxis_transform(),zorder=0)
ax[0].set(xlabel="analysis cycle", ylabel="RMSE",
        title=op+" GM")
ax[1].set(xlabel="analysis cycle", ylabel="RMSE",
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
if not spinup:
    fig.savefig("{}_e_{}_nospinup.png".format(model, op))
    fig2.savefig("{}_e+stda_{}_nospinup.png".format(model, op))
else:
    fig.savefig("{}_e_{}.png".format(model, op))
    fig2.savefig("{}_e+stda_{}.png".format(model, op))
#fig.savefig("{}_e_{}+nodiff.pdf".format(model, op))
# observation error (loosely dashed)
axf[0].plot(x, y, linestyle=(0, (5, 10)), color='black')
axf[1].plot(x, y, linestyle=(0, (5, 10)), color='black')
axf2[0].hlines([1],0,1,colors='gray',transform=axf2[0].get_yaxis_transform(),zorder=0)
axf2[1].hlines([1],0,1,colors='gray',transform=axf2[1].get_yaxis_transform(),zorder=0)
axf[0].set(xlabel="forecast cycle", ylabel="RMSE",
        title=op+" GM")
axf[1].set(xlabel="forecast cycle", ylabel="RMSE",
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
if not spinup:
    figf.savefig("{}_ef_{}_nospinup.png".format(model, op))
    figf2.savefig("{}_ef+stdf_{}_nospinup.png".format(model, op))
else:
    figf.savefig("{}_ef_{}.png".format(model, op))
    figf2.savefig("{}_ef+stdf_{}.png".format(model, op))