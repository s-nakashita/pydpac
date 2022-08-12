import numpy as np
import matplotlib.pyplot as plt


t = 1000
t = 1000
q = np.load(f"test/q{t:06d}.npy").T
psi = np.load(f"test/p{t:06d}.npy").T
n = q.shape[0]
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
plt.rcParams["font.size"] = 18
fig, axs = plt.subplots(1, 2, figsize=[14, 6])
z = [q, psi]
zmax = [1.5e5, 1.0e1]
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
fig.savefig(f"test/pq{t:06d}.png")
plt.show()

