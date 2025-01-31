import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
from methods import perts, linecolor, iinflist,infltype, inflcolor

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = sys.argv[4]
markers = {"3d":"o","4d":"x","3ds":"x","4ds":"^"}
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
x = np.arange(na) + 1
fig, ax = plt.subplots(figsize=(10,5),constrained_layout=True)
i = 0
lplot = False
for iinf in iinflist:
    c = inflcolor[iinf]
    label = infltype[iinf]
    #
    f = "pdr_{}_{}_{}.txt".format(op, pt, iinf)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    pdr = np.loadtxt(f)
    if pt=='mlefcw' or pt=='mlefy':
        pdr = pdr.reshape(na,-1).mean(axis=1)
    pdravg = np.mean(pdr[int(na/3):])
    print("{}, PDR = {}".format(pt,pdravg))
    ax.plot(x, pdr, color=c, label=label+f'={pdravg:.3f}')
    lplot = True
if not lplot: exit()
ax.set(xlabel="analysis cycle", ylabel="PDR",
        title=op)
if len(x) > 50:
    ax.set_xticks(x[::len(x)//10])
    ax.set_xticks(x[::len(x)//20], minor=True)
else:
    ax.set_xticks(x[::5])
    ax.set_xticks(x, minor=True)
ax.legend()
fig.savefig("{}_pdr_{}_{}.png".format(model, op, pt))
