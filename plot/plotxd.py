import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "mlefw", "etkf", "po", "srf", "letkf", "kf", "var",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
linecolor = {"mlef":'tab:blue',"mlefw":'tab:orange',"etkf":'tab:green', "po":'tab:red',\
        "srf":"tab:pink", "letkf":"tab:purple", "kf":"tab:cyan", "var":"tab:olive",\
        "mlefcw":"tab:green","mlefy":"tab:orange","mlefbe":"tab:red","mlefbm":"tab:pink"}
marker = {"3d":"o","4d":"x"}
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
ix = np.loadtxt('ix.txt')
y = np.ones(ix.size) * sigma[op]
fig, ax = plt.subplots(figsize=(12,5),constrained_layout=True)
i = 0
vmax = 0.0
for pt in perts:
    f = "xdmean_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xdmean = np.loadtxt(f)
    if np.isnan(xdmean).any():
        print("divergence in {}".format(pt))
        continue
    print("{}, mean RMSE = {}".format(pt,np.mean(xdmean_gm)))
    f = "xsmean_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xsmean = np.loadtxt(f)
    print("{}, mean SPREAD = {}".format(pt,np.mean(xsmean_gm)))
    ax.plot(ix, xdmean, linestyle="solid", color=linecolor[pt], label=pt)
    ax.plot(ix, xsmean, linestyle="dashed", color=linecolor[pt])
    vmax = max(np.max(xdmean),np.max(xsmean),vmax)
# observation error (loosely dashed)
ax.plot(ix, y, linestyle=(0, (5, 10)), color='black')
ax.set(xlabel="state", ylabel="RMSE or SPREAD",
        title=op)
vmax = max(vmax,np.max(y))
ax.set_xlim(ix[0],ix[-1])
ax.set_xticks(ix[::(ix.size//8)])
ax.legend()
ax.set_ylim(0.0,vmax)
fig.savefig("{}_xd_{}.png".format(model, op))
