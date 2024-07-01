import sys
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from methods import perts, linecolor

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
marker = {"3d":"o","4d":"x"}
sigma = {"linear": 1.0, "quadratic": 1.0, "cubic": 1.0, \
    "quadratic-nodiff": 8.0e-1, "cubic-nodiff": 7.0e-2, \
    "test":1.0, "abs":1.0, "hint":1.0}
i = 0
vmax = 0.0
lplot = False
lines0 = []
labels0 = []
lines1 = []
labels1 = []
lines2 = []
labels2 = []
lines3 = []
labels3 = []
plot_Jk = False
for pt in perts:
    #LAM
    #f = "{}_lam_coef_a_{}_{}.txt".format(model, op, pt)
    #if os.path.isfile(f):
    #    coef_a = np.loadtxt(f)
    #    cycles_lam = np.arange(1,coef_a.shape[0]+1)
    #else:
    cycles_lam = []
    coef_a = []
    for icycle in range(na):
            #LAM
            f = "{}_lam_coef_a_{}_{}_cycle{}.txt".format(model, op, pt, icycle)
            if os.path.isfile(f):
                coef_a_tmp = np.loadtxt(f)
                coef_a.append(coef_a_tmp)
                cycles_lam.append(icycle)
            else:
                f = "data/{2}/{0}_lam_coef_a_{1}_{2}_cycle{3}.txt".format(model, op, pt, icycle)
                if os.path.isfile(f):
                    coef_a_tmp = np.loadtxt(f)
                    coef_a.append(coef_a_tmp)
                    cycles_lam.append(icycle)
            #
            #cycles.append(icycle)
    if len(coef_a) > 0:
            coef_a = np.array(coef_a)
            np.savetxt("{}_lam_coef_a_{}_{}.txt".format(model, op, pt), coef_a)
    if len(cycles_lam) == 0:
        print("not exist {}".format(pt))
        continue
    fig, ax = plt.subplots(figsize=(12,6),constrained_layout=True)
    coef_a_mean = np.mean(coef_a,axis=1)
    coef_a_std  = np.std( coef_a,axis=1)
    ax.plot(cycles_lam, coef_a_mean, linestyle="solid", label=f"mean={np.mean(coef_a_mean):.3f}")
    ax.fill_between(cycles_lam, coef_a_mean-coef_a_std, coef_a_mean+coef_a_std, \
        color="tab:blue", alpha=0.3)
    ax.grid()
    ax.legend()
    ax.set(xlabel="analysis cycles", ylabel="a",
        title=op+" "+pt)
    fig.savefig("{}_coef_a_{}_{}.png".format(model, op, pt))
    plt.show()