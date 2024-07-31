import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
from methods import perts, linecolor, iinflist, infltype, inflcolor

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
nspinup = na // 5
linfl = False
if len(sys.argv)>5 and sys.argv[5]=='infl':
    pt = sys.argv[4]
    linfl=True
markers = {"3d":"o","4d":"x","3ds":"x","4ds":"^"}
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
x = np.arange(na) + 1
fig, ax = plt.subplots(figsize=(10,5),constrained_layout=True)
i = 0
lplot = False
if not linfl:
    for pt in perts:
        if pt[:2] != "4d":
            c = linecolor[pt]
            ls = 'solid'
        else:
            c = linecolor[pt[2:]]
            ls = 'dashed'
        label = pt
        #
        f = "infl_{}_{}.txt".format(op, pt)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        infl = np.loadtxt(f)
        if pt=='mlefcw' or pt=='mlefy':
            infl = infl.reshape(na,-1).mean(axis=1)
        inflavg = np.mean(infl[nspinup:])
        print("{}, adaptive inflation = {}".format(label,inflavg))
        ax.plot(x, infl, color=c, ls=ls, label=label+f'={inflavg:.3f}')
        lplot = True
else:
    for iinf in iinflist:
        c = inflcolor[iinf]
        label = infltype[iinf]
        ls = 'solid'
        #
        f = "infl_{}_{}_{}.txt".format(op, pt, iinf)
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        infl = np.loadtxt(f)
        if pt=='mlefcw' or pt=='mlefy':
            infl = infl.reshape(na,-1).mean(axis=1)
        inflavg = np.mean(infl[int(na/3):])
        print("{}, adaptive inflation = {}".format(label,inflavg))
        ax.plot(x, infl, color=c, ls=ls, label=label+f'={inflavg:.3f}')
        lplot = True
if not lplot: exit()
ax.set(xlabel="analysis cycle", ylabel="Inflation",
        title=op)
if len(x) > 50:
    ax.set_xticks(x[::len(x)//10])
    ax.set_xticks(x[::len(x)//20], minor=True)
else:
    ax.set_xticks(x[::5])
    ax.set_xticks(x, minor=True)
ax.legend()
if linfl:
    fig.savefig("{}_infl_{}_{}.png".format(model, op, pt))
else:
    fig.savefig("{}_infl_{}.png".format(model, op))
