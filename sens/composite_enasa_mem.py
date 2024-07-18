import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 14
import xarray as xr
from scipy.stats import ttest_1samp
from pathlib import Path
import sys

datadir = Path('data')

cmap = plt.get_cmap('tab10')
enasas = ['minnorm','diag','ridge','pcr','pls']
colors = {'asa':cmap(0),'mem8':cmap(1),'mem16':cmap(2),'mem24':cmap(3),'mem32':cmap(4),'mem40':cmap(5)}
markers = {'asa':'*','mem8':'o','mem16':'v','mem24':'s','mem32':'P','mem40':'X'}
ms = {'asa':8,'mem8':5,'mem16':5,'mem24':5,'mem32':5,'mem40':5}
marker_style=dict(markerfacecolor='none')
nenslist = [8,16,24,32]
nensbase = 8
dJlim = {24:0.2,48:1.0,72:2.0,96:3.6}
dxlim = {24:0.02,48:0.02,72:0.02,96:0.02}

vt = 24
if len(sys.argv)>1:
    vt = int(sys.argv[1])
solver = 'minnorm'
if len(sys.argv)>2:
    solver = sys.argv[2]
metric = ''
if len(sys.argv)>3:
    metric = sys.argv[3]
figdir = Path(f"fig/vt{vt}{solver}{metric}")
if not figdir.exists(): figdir.mkdir()

# load results
ds_dict = {}
ds_asa = xr.open_dataset(datadir/f'asa{metric}_vt{vt}nens{nensbase}.nc')
print(ds_asa)
ds_dict['asa'] = ds_asa
for nens in nenslist:
    ds = xr.open_dataset(datadir/f'{solver}{metric}_vt{vt}nens{nens}.nc')
    ds_dict[f'mem{nens}'] = ds

ncols = 2
nrows = 3
figdJ, axsdJ = plt.subplots(figsize=[8,10],nrows=nrows,ncols=ncols,constrained_layout=True)
figdx, axsdx = plt.subplots(figsize=[8,10],nrows=nrows,ncols=ncols,constrained_layout=True)

for i, key in enumerate(ds_dict.keys()):
    if key=='asa':
        axdJ = axsdJ[0,0]
        axdx = axsdx[0,0]
    else:
        axdJ = axsdJ[1:,:].flatten()[i-1]
        axdx = axsdx[1:,:].flatten()[i-1]
    marker_style = dict(markerfacecolor=colors[key],markeredgecolor='k',ms=ms[key])
    ds = ds_dict[key]
    x = ds.x
    nx = x.size
    nxh = nx // 2
    hwidth = 1
    i0 = nxh - hwidth
    i1 = nxh + hwidth + 1
    
    ics = ds.ic.values
    dJdx0 = ds.dJdx0.values
    dx0opt = ds.dx0opt.values
    print(dJdx0.shape)
    dJdx0mean = np.zeros(dJdx0.shape[1])
    dx0optmean = np.zeros(dx0opt.shape[1])
    #dJdx0std = np.zeros(dJdx0.shape[1])
    #dx0optstd = np.zeros(dx0opt.shape[1])
    dJdx0_comp = []
    dx0opt_comp = []
    for icycle, ic in enumerate(ics):
        dJtmp = np.roll(dJdx0[icycle],nxh-ic)
        dJdx0_comp.append(np.abs(dJtmp))
        dJdx0mean = dJdx0mean + dJtmp
    #    dJdx0std = dJdx0std + dJtmp**2
        dxtmp = np.roll(dx0opt[icycle],nxh-ic)
        dx0opt_comp.append(np.abs(dxtmp))
        dx0optmean = dx0optmean + dxtmp
    #    dx0optstd = dx0optstd + dxtmp**2
    ncycle = ics.size
    dJdx0mean /= ncycle
    #dJdx0std = dJdx0std/ncycle - dJdx0mean**2
    #dJdx0std[dJdx0std<0.0]=0.0
    #dJdx0std = np.sqrt(dJdx0std)
    dx0optmean /= ncycle
    #dx0optstd = dx0optstd/ncycle - dx0optmean**2
    #dx0optstd[dx0optstd<0.0]=0.0
    #dx0optstd = np.sqrt(dx0optstd)
    
    #axdJ.fill_between(x,dJdx0mean-dJdx0std,dJdx0mean+dJdx0std,color=colors[key],alpha=0.5)
    axdJ.plot(x,dJdx0mean,c=colors[key],**marker_style)
    #axdx.fill_between(x,dx0optmean-dx0optstd,dx0optmean+dx0optstd,color=colors[key],alpha=0.5)
    axdx.plot(x,dx0optmean,c=colors[key],**marker_style)
    ## t-test for zero-mean
    alpha=0.01
    dJdx0_comp = np.array(dJdx0_comp)
    dx0opt_comp = np.array(dx0opt_comp)
    _, dJ_p = ttest_1samp(dJdx0_comp,1.0e-4,alternative='greater',axis=0)
    _, dx_p = ttest_1samp(dx0opt_comp,1.0e-4,alternative='greater',axis=0)
    axdJ.plot(x[dJ_p<alpha],dJdx0mean[dJ_p<alpha],\
        lw=0.0,c=colors[key],marker=markers[key],**marker_style)
    axdx.plot(x[dx_p<alpha],dx0optmean[dx_p<alpha],\
        lw=0.0,c=colors[key],marker=markers[key],**marker_style)
    axdJ.set_ylabel(key)
    axdx.set_ylabel(key)
for ax in np.concatenate((axsdJ.flatten(),axsdx.flatten())):
    ax.fill_between(x[i0:i1],0,1,\
        color='gray',alpha=0.5,transform=ax.get_xaxis_transform(),zorder=0)

#for ax in axsdJ.flatten():
#    ax.set_ylim(-dJlim[vt],dJlim[vt])
#    ax.grid()
#for ax in axsdx.flatten():
#    ax.set_ylim(-dxlim[vt],dxlim[vt])
#    ax.grid()
for axs in [axsdJ,axsdx]:
    for i,ax in enumerate(axs.flatten()):
        if i==0:
            ymin,ymax = ax.get_ylim()
        else:
            ax.set_ylim(ymin,ymax)
        ax.grid()
axsdJ[0,1].remove()
axsdx[0,1].remove()
figdJ.suptitle(r'$\frac{\partial J}{\partial \mathbf{x}_0}$'+f" FT{vt} {solver}")
figdx.suptitle(r'$\delta\mathbf{x}_0^\mathrm{opt}$'+f" FT{vt} {solver}")
figdJ.savefig(figdir/f"composite_dJdx0{metric}_vt{vt}{solver}.png",dpi=300)
figdx.savefig(figdir/f"composite_dx0opt{metric}_vt{vt}{solver}.png",dpi=300)
#plt.show()