import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200
import sys
sys.path.append('../plot')
from plot_heatmap import heatmap, annotate_heatmap
from pathlib import Path
import argparse

figdir = Path('l96')

lpatchs = [1,3,5,7,9]
lhalos = [1,3,5,7,9]

parser = argparse.ArgumentParser()
parser.add_argument("-vt","--vt",type=int,default=24,\
    help="verification time (hours)")
parser.add_argument("-ne","--nens",type=int,default=8,\
    help="ensemble size")
argsin = parser.parse_args()
vt = argsin.vt
ioffset = vt // 6
nens = argsin.nens

rmse_rho0_dyn = np.loadtxt(figdir/f'rmse_rho0_dyn_vt{vt}.txt')
rmse_rho1_dyn = np.loadtxt(figdir/f'rmse_rho1_dyn_vt{vt}.txt')
rmse_rho0_dyn_mean = np.nanmean(rmse_rho0_dyn,axis=0)
rmse_rho0_dyn_std = np.nanstd(rmse_rho0_dyn,axis=0)
rmse_rho1_dyn_mean = np.nanmean(rmse_rho1_dyn,axis=0)
rmse_rho1_dyn_std = np.nanstd(rmse_rho1_dyn,axis=0)
fig, axs = plt.subplots(nrows=3,sharex=True,figsize=[8,6],constrained_layout=True)
#axs[1].fill_between([0,1],rmse_rho0_dyn_mean-rmse_rho0_dyn_std,rmse_rho0_dyn_mean+rmse_rho0_dyn_std,\
#        color='gray',alpha=0.2,transform=axs[1].get_yaxis_transform())
axs[1].hlines(rmse_rho0_dyn_mean[0],0,1,colors='k',transform=axs[1].get_yaxis_transform(),label='dyn')
axs[1].hlines(rmse_rho0_dyn_mean[1],0,1,colors='k',ls='dashed',transform=axs[1].get_yaxis_transform())
#axs[2].fill_between([0,1],rmse_rho1_dyn_mean-rmse_rho1_dyn_std,rmse_rho1_dyn_mean+rmse_rho1_dyn_std,\
#        color='gray',alpha=0.2,transform=axs[2].get_yaxis_transform())
axs[2].hlines(rmse_rho1_dyn_mean[0],0,1,colors='k',transform=axs[2].get_yaxis_transform(),label='dyn')
axs[2].hlines(rmse_rho1_dyn_mean[1],0,1,colors='k',ls='dashed',transform=axs[2].get_yaxis_transform())
cmap = plt.get_cmap('tab10')
for icol,lh in enumerate(lhalos):
    rmse_dx_mean = []
    rmse_dx_std = []
    rmse_rho0_ens_mean = []
    rmse_rho0_ens_std = []
    rmse_rho1_ens_mean = []
    rmse_rho1_ens_std = []
    for lp in lpatchs:
        rmse_dx = np.loadtxt(figdir/f'lp{lp}lh{lh}/rmse_dx_letlm_vt{vt}ne{nens}.txt')
        rmse_rho0_ens = np.loadtxt(figdir/f'lp{lp}lh{lh}/rmse_rho0_letlm_vt{vt}ne{nens}.txt')
        rmse_rho1_ens = np.loadtxt(figdir/f'lp{lp}lh{lh}/rmse_rho1_letlm_vt{vt}ne{nens}.txt')
        rmse_dx_mean.append(np.nanmean(rmse_dx))
        rmse_dx_std.append(np.nanstd(rmse_dx))
        rmse_rho0_ens_mean.append(np.nanmean(rmse_rho0_ens,axis=0))
        rmse_rho0_ens_std.append(np.nanstd(rmse_rho0_ens,axis=0))
        rmse_rho1_ens_mean.append(np.nanmean(rmse_rho1_ens,axis=0))
        rmse_rho1_ens_std.append(np.nanstd(rmse_rho1_ens,axis=0))
    rmse_dx_mean = np.array(rmse_dx_mean)
    rmse_dx_std = np.array(rmse_dx_std)
    rmse_rho0_ens_mean = np.array(rmse_rho0_ens_mean)
    rmse_rho0_ens_std = np.array(rmse_rho0_ens_std)
    rmse_rho1_ens_mean = np.array(rmse_rho1_ens_mean)
    rmse_rho1_ens_std = np.array(rmse_rho1_ens_std)
    #axs[0].fill_between(lpatchs,rmse_dx_mean-rmse_dx_std,rmse_dx_mean+rmse_dx_std,\
    #    color=cmap(icol),alpha=0.2)
    axs[0].plot(lpatchs,rmse_dx_mean,marker='o',c=cmap(icol),label=f'lhalo={lh}')
    #axs[1].fill_between(lpatchs,rmse_rho0_ens_mean-rmse_rho0_ens_std,rmse_rho0_ens_mean+rmse_rho0_ens_std,\
    #    color=cmap(icol),alpha=0.2)
    axs[1].plot(lpatchs,rmse_rho0_ens_mean[:,0],marker='o',c=cmap(icol),label=f'lhalo={lh}')
    axs[1].plot(lpatchs,rmse_rho0_ens_mean[:,1],marker='^',c=cmap(icol),ls='dashed')
    #axs[2].fill_between(lpatchs,rmse_rho1_ens_mean-rmse_rho1_ens_std,rmse_rho1_ens_mean+rmse_rho1_ens_std,\
    #    color=cmap(icol),alpha=0.2)
    axs[2].plot(lpatchs,rmse_rho1_ens_mean[:,0],marker='o',c=cmap(icol),label=f'lhalo={lh}')
    axs[2].plot(lpatchs,rmse_rho1_ens_mean[:,1],marker='^',c=cmap(icol),ls='dashed')
axs[2].set_xticks(lpatchs)
axs[2].set_xlabel('length of patch')
axs[2].legend(loc='upper left',bbox_to_anchor=(1.01,1.0))
axs[0].set_title(r'$\mathbf{M}^\mathrm{ens}\delta\mathbf{x}-\mathbf{M}^\mathrm{dyn}\delta\mathbf{x}$')
axs[1].set_title('GC5: '+r'$\rho_t\circ[M(\mathbf{x}+\mathbf{y})-M(\mathbf{x})]-[M(\mathbf{x}+\mathbf{P}_0\mathbf{y})-M(\mathbf{x})]$')
axs[2].set_title('Boxcar: '+r'$\rho_t\circ[M(\mathbf{x}+\mathbf{y})-M(\mathbf{x})]-[M(\mathbf{x}+\mathbf{P}_0\mathbf{y})-M(\mathbf{x})]$')
fig.suptitle(f'RMSE FT{vt:02d} K={nens}')
fig.savefig(figdir/f'rmse_letlm_vt{vt}ne{nens}.png')
plt.show()