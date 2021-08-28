import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
aos = sys.argv[4]
perts = ["mlef", "etkf", "po", "srf", "letkf", "kf", "var", "var4d", "rep", "rep-mb"]
cmap = "Reds"
for pt in perts:
    fig, ax = plt.subplots()
    f = "{}_e_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    e = np.load(f)
    print(e.shape)
    print(e.min(), e.max())
    t = np.arange(e.shape[0])/4
    xs = np.arange(e.shape[1]) + 1
    xlim = 5.0
    #mappable0 = ax.pcolor(xs, t, e, cmap=cmap, norm=Normalize(vmin=0.0, vmax=xlim))
    mappable0 = ax.contourf(xs, t, e, levels=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
                cmap=cmap, norm=Normalize(vmin=0.0, vmax=xlim))
    ax.contour(xs, t, e, levels=[1.0, 2.0, 3.0, 4.0, 5.0], 
                colors='black', norm=Normalize(vmin=0.0, vmax=xlim))
    ax.set_xticks(xs[::10])
    ax.set_yticks(t[::4])
    ax.set_xlabel("site ocean(1-20):land(21-40)")
    ax.set_ylabel("forecast range (days)")
    ax.set_title("forecast error "+pt+" "+op+" "+aos)
    pp = fig.colorbar(mappable0,ax=ax,orientation="vertical")
    fig.savefig("{}_e_{}_{}_{}_2day.png".format(model,op,pt,aos))