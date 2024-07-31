import numpy as np
from numpy import random
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['font.size'] = 16
from pathlib import Path
import sys
nx_true = 960
nx_lam  = 480
nx_gm   = 240
intgm   = 4
nk_lam  = 32
nk_gm   = 8
ni = 12
b = 10.0
c = 0.6
dt = 0.05 / 36.0
F = 15.0
ist_lam = 240
nsp = 10
po = 1
if len(sys.argv)>1:
    po = int(sys.argv[1])
## ensemble size
nens = 720
outdir=Path(f'lorenz/ng{nx_gm}nl{nx_lam}kg{nk_gm}kl{nk_lam}nsp{nsp}p{po}F{int(F)}b{b:.1f}c{c:.1f}')

ix_true = np.loadtxt(outdir/'ix_true.txt')
ix_gm = np.loadtxt(outdir/'ix_gm.txt')
ix_lam = np.loadtxt(outdir/'ix_lam.txt')
B_lam = np.load(outdir/'B_lam.npy')
B_gmfull = np.load(outdir/'B_gmfull.npy')
E_lg = np.load(outdir/'E_lg.npy')
E_gl = np.load(outdir/'E_gl.npy')
for i0 in [0,nx_lam // 2,nx_lam-1]:
    fig, axs = plt.subplots(nrows=2,figsize=[8,6],sharex=True,constrained_layout=True)
    axs[0].plot(ix_lam,B_lam[i0,:]/B_lam[i0,i0],c='r',label='LAM')
    i0g = np.argmin(np.abs(ix_gm - ix_lam[i0]))
    axs[0].plot(ix_gm,B_gmfull[i0g,:]/B_gmfull[i0g,i0g],c='b',label='GM')
    axs[1].plot(ix_lam,E_gl[i0,:]/E_gl[i0,i0],c='g',label='GM-LAM')
    axs[1].plot(ix_lam,E_lg[i0,:]/E_lg[i0,i0],c='y',label='LAM-GM')
    for ax in axs:
        ax.set_xlim(ix_true[0],ix_true[-1])
        ax.vlines([ix_lam[0],ix_lam[i0],ix_lam[-1]],0,1,colors='k',ls='dashdot',transform=ax.get_xaxis_transform())
        ax.legend()
    fig.suptitle('correlation')
    fig.savefig(outdir/f"crosscorr1d_gm+lam_i{i0}.png",dpi=300)
    plt.show()
    plt.close()