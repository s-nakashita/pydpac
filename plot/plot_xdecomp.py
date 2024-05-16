import sys
import os
import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
from nmc_tools import scale_decomp

plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])

perts = ["mlef", "envar", "envar_nest", "etkf", "po", "srf", "letkf", "kf", "var", "var_nest",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]

cmap = "coolwarm"
f = "truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[:na,]
print(xt.shape)
nx_t = xt.shape[1]
t = np.arange(na)
ix_t = np.loadtxt('ix_true.txt')
ix_t_rad = ix_t * 2.0 * np.pi / nx_t
xt2x = interp1d(ix_t,xt)

kthres = [6.,30.]
ncols = len(kthres) + 2
fig, axs = plt.subplots(ncols=ncols,figsize=[12,6],sharey=True,constrained_layout=True)
mp0 = axs[0].pcolormesh(ix_t,t,xt,shading='auto',\
    cmap=cmap,norm=Normalize(-15,15))
axs[0].set_xticks(ix_t[::(nx_t//10)])
axs[0].set_yticks(t[::(na//8)])
axs[0].set_xlabel("site")
axs[0].set_ylabel("DA cycle")
axs[0].set_title("full")
p0 = fig.colorbar(mp0, ax=axs[0], orientation='vertical',shrink=0.6,pad=0.01)
xdecomp = scale_decomp(xt,ix=ix_t_rad, kthres=kthres)
for i,xd in enumerate(xdecomp):
    mp1 = axs[i+1].pcolormesh(ix_t,t,xd,shading='auto',\
    cmap=cmap,norm=Normalize(-15,15))
    axs[i+1].set_xticks(ix_t[::(nx_t//10)])
    axs[i+1].set_xlabel("site")
    #axs[i+1].set_yticks(t[::(na//8)])
    #axs[i+1].set_ylabel("DA cycle")
axs[1].set_title(r"$k \leq$"+f"{kthres[0]}")
axs[2].set_title(f"{kthres[0]}"+r"$\leq k \leq$"+f"{kthres[1]}")
axs[3].set_title(f"{kthres[1]}"+r"$\leq k$")
fig.suptitle("nature")
fig.savefig("nature_decomp.png",dpi=300)
plt.show()
