import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
from methods import perts, linecolor

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
linfl = False
if len(sys.argv) > 4:
    pt = sys.argv[4]
    #if pt == "mlef":
    #    perts = perts + [pt+"be", pt+"bm", pt+"cw",pt+"y",
    #    "4d"+pt, "4d"+pt+"be", "4d"+pt+"bm", "4d"+pt+"cw", "4d"+pt+"y"]
    #    linecolor.update({pt:'tab:blue',pt+"be":'tab:red',pt+"bm":'tab:pink',\
    #        pt+"cw":'tab:green',pt+"y":'tab:orange'})
    #else:
    #    perts = perts + [pt, pt+"be", pt+"bm", "l"+pt]
    #    linecolor.update({pt:'tab:blue',pt+"be":'tab:orange',pt+"bm":'tab:green',"l"+pt:'tab:red'})
    perts = list(set(perts))
    #perts = [pt, pt+"be", pt+"bm", "l"+pt]
    #linecolor = {pt:'tab:blue',pt+"be":'tab:orange',pt+"bm":'tab:green',"l"+pt:'tab:red'}
    perts = [pt, pt+"be", pt+"bm", "l"+pt+"0", "l"+pt+"1", "l"+pt+"2", 'letkf']
    linecolor = {pt:'tab:blue',pt+"be":'tab:orange',pt+"bm":'tab:green',
    "l"+pt+"0":'tab:cyan', "l"+pt+"1":'tab:pink', "l"+pt+"2":'tab:purple', 'letkf':'tab:red'}
    #perts = ["l"+pt+"1", "l"+pt+"2", "l"+pt+"3", 'letkf']
    #linecolor = {"l"+pt+"1":'tab:blue', "l"+pt+"2":'tab:orange', "l"+pt+"3":'tab:green', 'letkf':'tab:red'}
if len(sys.argv) > 5 and sys.argv[5]=='infl':
    linfl = True
    iinflist = [-1,0,1,2,3]
    infltype = {-1:'pre-mi',0:'post-mi', 1:'add', 2:'rtpp', 3:'rtps'}
    linecolor = {-1:'tab:blue', 0:'tab:orange', 1:'tab:green', 2:'tab:red', 3:'tab:purple'}

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
if not linfl:
    for pt in perts:
        label = pt
        c = linecolor[pt]
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
        ax.plot(ix, xdmean, linestyle="solid", color=c, label=label)
        if pt != "kf" and pt != "var" and pt != "4dvar":
            ax.plot(ix, xsmean, linestyle="dashed", color=c)
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
        axf.plot(ix, xdfmean, linestyle="solid", color=c, label=label)
        if pt != "kf" and pt != "var" and pt != "4dvar":
            axf.plot(ix, xsfmean, linestyle="dashed", color=c)
            vmax = max(np.max(xdfmean),np.max(xsfmean),vmax)
        else:
            vmax = max(np.max(xdfmean),vmax)
else:
    for iinf in iinflist:
        itype = infltype[iinf]
        c = linecolor[iinf]
        label = itype
        ## analysis
        f = "xdmean_{}_{}_{}.txt".format(op, pt, iinf)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        xdmean = np.loadtxt(f)
        if np.isnan(xdmean).any():
            print("divergence in {}".format(pt))
            continue
        print("{}-{}, analysis RMSE = {}".format(pt,itype,np.mean(xdmean)))
        f = "xsmean_{}_{}_{}.txt".format(op, pt, iinf)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        xsmean = np.loadtxt(f)
        print("{}-{}, analysis SPREAD = {}".format(pt,itype,np.mean(xsmean)))
        ax.plot(ix, xdmean, linestyle="solid", color=c, label=label)
        if pt != "kf" and pt != "var" and pt != "4dvar":
            ax.plot(ix, xsmean, linestyle="dashed", color=c)
            vmax = max(np.max(xdmean),np.max(xsmean),vmax)
        else:
            vmax = max(np.max(xdmean),vmax)
        ## forecast
        f = "xdfmean_{}_{}_{}.txt".format(op, pt, iinf)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        xdfmean = np.loadtxt(f)
        if np.isnan(xdfmean).any():
            print("divergence in {}".format(pt))
            continue
        print("{}-{}, forecast RMSE = {}".format(pt,itype,np.mean(xdfmean)))
        f = "xsfmean_{}_{}_{}.txt".format(op, pt, iinf)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        xsfmean = np.loadtxt(f)
        print("{}-{}, forecast SPREAD = {}".format(pt,itype,np.mean(xsfmean)))
        axf.plot(ix, xdfmean, linestyle="solid", color=c, label=label)
        if pt != "kf" and pt != "var" and pt != "4dvar":
            axf.plot(ix, xsfmean, linestyle="dashed", color=c)
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
if not linfl:
    fig.savefig("{}_xd_{}.png".format(model, op))
else:
    fig.savefig("{}_xd_{}_{}.png".format(model, op, pt))

axf.plot(ix, y, linestyle=(0, (5, 10)), color='black')
axf.set(xlabel="state", ylabel="RMSE or SPREAD",
        title=op+" forecast")
axf.set_xlim(ix[0],ix[-1])
axf.set_xticks(ix[::(ix.size//8)])
axf.legend()
axf.set_ylim(0.0,vmax)
if not linfl:
    figf.savefig("{}_xdf_{}.png".format(model, op))
else:
    figf.savefig("{}_xdf_{}_{}.png".format(model, op, pt))
