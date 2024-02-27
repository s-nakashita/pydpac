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
marker = {"3d":"o","4d":"x"}
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
ix_gm = np.loadtxt('ix_gm.txt')
y_gm = np.ones(ix_gm.size) * sigma[op]
ix_lam = np.loadtxt('ix_lam.txt')
y_lam = np.ones(ix_lam.size) * sigma[op]
fig, ax = plt.subplots(nrows=2,ncols=1,figsize=(12,10),constrained_layout=True)
fig2, ax2 = plt.subplots(nrows=2,ncols=1,figsize=(12,10),constrained_layout=True)
i = 0
vmax = 0.0
for pt in perts:
    ## analysis
    #GM
    f = "xdmean_gm_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xdmean_gm = np.loadtxt(f)
    if np.isnan(xdmean_gm).any():
        print("divergence in {}".format(pt))
        continue
    print("{}, GM analysis RMSE = {}".format(pt,np.mean(xdmean_gm)))
    f = "xsmean_gm_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xsmean_gm = np.loadtxt(f)
    print("{}, GM analysis SPREAD = {}".format(pt,np.mean(xsmean_gm)))
    #LAM
    f = "xdmean_lam_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xdmean_lam = np.loadtxt(f)
    if np.isnan(xdmean_lam).any():
        print("divergence in {}".format(pt))
        continue
    print("{}, LAM analysis RMSE = {}".format(pt,np.mean(xdmean_lam)))
    f = "xsmean_lam_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xsmean_lam = np.loadtxt(f)
    print("{}, LAM analysis SPREAD = {}".format(pt,np.mean(xsmean_lam)))
    ax[0].plot(ix_gm, xdmean_gm, linestyle="solid", color=linecolor[pt], label=pt)
    ax[1].plot(ix_lam, xdmean_lam, linestyle="solid", color=linecolor[pt], label=pt)
    if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
        ax[0].plot(ix_gm, xsmean_gm, linestyle="dashed", color=linecolor[pt])
        ax[1].plot(ix_lam, xsmean_lam, linestyle="dashed", color=linecolor[pt])
        vmax = max(np.max(xdmean_gm),np.max(xdmean_lam),np.max(xsmean_gm),np.max(xsmean_lam),vmax)
    else:
        vmax = max(np.max(xdmean_gm),np.max(xdmean_lam),vmax)
    ## forecast
    #GM
    f = "xdfmean_gm_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xdfmean_gm = np.loadtxt(f)
    if np.isnan(xdfmean_gm).any():
        print("divergence in {}".format(pt))
        continue
    print("{}, GM forecast RMSE = {}".format(pt,np.mean(xdfmean_gm)))
    f = "xsfmean_gm_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xsfmean_gm = np.loadtxt(f)
    print("{}, GM forecast SPREAD = {}".format(pt,np.mean(xsfmean_gm)))
    #LAM
    f = "xdfmean_lam_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xdfmean_lam = np.loadtxt(f)
    if np.isnan(xdfmean_lam).any():
        print("divergence in {}".format(pt))
        continue
    print("{}, LAM forecast RMSE = {}".format(pt,np.mean(xdfmean_lam)))
    f = "xsfmean_lam_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xsfmean_lam = np.loadtxt(f)
    print("{}, LAM forecast SPREAD = {}".format(pt,np.mean(xsfmean_lam)))
    ax2[0].plot(ix_gm, xdfmean_gm, linestyle="solid", color=linecolor[pt], label=pt)
    ax2[1].plot(ix_lam, xdfmean_lam, linestyle="solid", color=linecolor[pt], label=pt)
    if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
        ax2[0].plot(ix_gm, xsfmean_gm, linestyle="dashed", color=linecolor[pt])
        ax2[1].plot(ix_lam, xsfmean_lam, linestyle="dashed", color=linecolor[pt])
        vmax = max(np.max(xdfmean_gm),np.max(xdfmean_lam),np.max(xsfmean_gm),np.max(xsfmean_lam),vmax)
    else:
        vmax = max(np.max(xdfmean_gm),np.max(xdfmean_lam),vmax)
# observation error (loosely dashed)
ax[0].plot(ix_gm, y_gm, linestyle=(0, (5, 10)), color='black')
ax[1].plot(ix_lam, y_lam, linestyle=(0, (5, 10)), color='black')
ax[0].vlines([ix_lam[0],ix_lam[-1]],0,1,colors='gray',alpha=0.5,transform=ax[0].get_xaxis_transform())
ax[0].set(xlabel="state", ylabel="RMSE or SPREAD",
        title=op+" GM analysis")
ax[1].set(xlabel="state", ylabel="RMSE or SPREAD",
        title=op+" LAM analysis")
vmax = max(vmax,np.max(y_gm))
#ax[0].set_xticks(ix_gm[::(ix_gm.size//8)])
#ax[1].set_xticks(ix_gm[::(ix_lam.size//8)])
for i in range(2):
    ax[i].set_xlim(ix_gm[0],ix_gm[-1])
    ax[i].set_xticks(ix_gm[::(ix_gm.size//8)])
    ax[i].legend()
    ax[i].set_ylim(0.0,vmax)
fig.savefig("{}_xd_{}.png".format(model, op))

# observation error (loosely dashed)
ax2[0].plot(ix_gm, y_gm, linestyle=(0, (5, 10)), color='black')
ax2[1].plot(ix_lam, y_lam, linestyle=(0, (5, 10)), color='black')
ax2[0].vlines([ix_lam[0],ix_lam[-1]],0,1,colors='gray',alpha=0.5,transform=ax2[0].get_xaxis_transform())
ax2[0].set(xlabel="state", ylabel="RMSE or SPREAD",
        title=op+" GM forecast")
ax2[1].set(xlabel="state", ylabel="RMSE or SPREAD",
        title=op+" LAM forecast")
#vmax2 = max(vmax2,np.max(y_gm))
#ax[0].set_xticks(ix_gm[::(ix_gm.size//8)])
#ax[1].set_xticks(ix_gm[::(ix_lam.size//8)])
for i in range(2):
    ax2[i].set_xlim(ix_gm[0],ix_gm[-1])
    ax2[i].set_xticks(ix_gm[::(ix_gm.size//8)])
    ax2[i].legend()
    ax2[i].set_ylim(0.0,vmax)
fig2.savefig("{}_xdf_{}.png".format(model, op))
