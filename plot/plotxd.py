import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
from methods import perts, linecolor

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
marker = {"3d":"o","4d":"x"}
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
ix = np.loadtxt('ix.txt')
y = np.ones(ix.size) * sigma[op]
fig, ax = plt.subplots(figsize=(12,5),constrained_layout=True)
figf, axf = plt.subplots(figsize=(12,5),constrained_layout=True)
i = 0
vmax = 0.0
for pt in perts:
    ## analysis
    f = "xdmean_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xdmean = np.loadtxt(f)
    if np.isnan(xdmean).any():
        print("divergence in {}".format(pt))
        continue
    print("{}, analysis RMSE = {}".format(pt,np.mean(xdmean)))
    f = "xsmean_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xsmean = np.loadtxt(f)
    print("{}, analysis SPREAD = {}".format(pt,np.mean(xsmean)))
    ax.plot(ix, xdmean, linestyle="solid", color=linecolor[pt], label=pt)
    if pt != "kf" and pt != "var" and pt != "4dvar":
        ax.plot(ix, xsmean, linestyle="dashed", color=linecolor[pt])
        vmax = max(np.max(xdmean),np.max(xsmean),vmax)
    else:
        vmax = max(np.max(xdmean),vmax)
    ## forecast
    f = "xdfmean_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xdfmean = np.loadtxt(f)
    if np.isnan(xdfmean).any():
        print("divergence in {}".format(pt))
        continue
    print("{}, forecast RMSE = {}".format(pt,np.mean(xdfmean)))
    f = "xsfmean_{}_{}.txt".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xsfmean = np.loadtxt(f)
    print("{}, forecast SPREAD = {}".format(pt,np.mean(xsfmean)))
    axf.plot(ix, xdfmean, linestyle="solid", color=linecolor[pt], label=pt)
    if pt != "kf" and pt != "var" and pt != "4dvar":
        axf.plot(ix, xsfmean, linestyle="dashed", color=linecolor[pt])
        vmax = max(np.max(xdfmean),np.max(xsfmean),vmax)
    else:
        vmax = max(np.max(xdfmean),vmax)
# observation error (loosely dashed)
ax.plot(ix, y, linestyle=(0, (5, 10)), color='black')
ax.set(xlabel="state", ylabel="RMSE or SPREAD",
        title=op+" analysis")
vmax = max(vmax,np.max(y))
ax.set_xlim(ix[0],ix[-1])
ax.set_xticks(ix[::(ix.size//8)])
ax.legend()
ax.set_ylim(0.0,vmax)
fig.savefig("{}_xd_{}.png".format(model, op))

axf.plot(ix, y, linestyle=(0, (5, 10)), color='black')
axf.set(xlabel="state", ylabel="RMSE or SPREAD",
        title=op+" forecast")
axf.set_xlim(ix[0],ix[-1])
axf.set_xticks(ix[::(ix.size//8)])
axf.legend()
axf.set_ylim(0.0,vmax)
figf.savefig("{}_xdf_{}.png".format(model, op))
