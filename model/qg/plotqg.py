import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams["font.size"] = 18

datadir = Path('test')
#datadir = Path('../../data/qg')
figdir = Path('test/fig')
#figdir = Path('../../data/qg/fig')
if not figdir.exists():
    figdir.mkdir()
ts = 3600
te = 10000
dt = 100
for t in range(ts,te+dt,dt):
    q = np.load(datadir/f"q{t:06d}.npy").T
    psi = np.load(datadir/f"p{t:06d}.npy").T
    n = q.shape[0]
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    fig, axs = plt.subplots(1, 2, figsize=[14, 6])
    z = [psi,q]
    zmax = [1.0e1,1.5e5]
    title = [r"stream function $\psi$", r"potential vorticity $q$"]
    for i in range(len(axs)):
        ax = axs[i]
        c = ax.pcolormesh(x, y, z[i])
#    c = ax.pcolormesh(x, y, z[i],
#            cmap="RdYlBu_r", vmin=-zmax[i], vmax=zmax[i])
        ax.set_title(title[i])
        ax.set_aspect("equal")
        fig.colorbar(c, ax=ax, shrink=0.8)
    fig.suptitle(r"$t=$"+f"{t}")
    fig.tight_layout()
    fig.savefig(figdir/f"pq{t:06d}.png")
    plt.close()

