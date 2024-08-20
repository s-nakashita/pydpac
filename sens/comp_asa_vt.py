import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 14
import xarray as xr
from sklearn.linear_model import LinearRegression
from pathlib import Path
import argparse

datadir = Path('data')

cmap = plt.get_cmap('tab10')
enasas = ['minnorm','diag','ridge','pcr','pls','std']
colors = {'asa':cmap(0),'minnorm':cmap(1),'diag':cmap(2),'pcr':cmap(3),'ridge':cmap(4),'pls':cmap(5),'std':cmap(6)}
markers = {'asa':'*','minnorm':'o','diag':'v','pcr':'s','ridge':'P','pls':'X','std':'^'}
marker_style=dict(markerfacecolor='none')
vtlist = [24,48,72,96]

parser = argparse.ArgumentParser()
parser.add_argument("-ne","--nens",type=int,default=8,\
    help="ensemble size")
parser.add_argument("-m","--metric",type=str,default="",\
    help="forecast metric type")
argsin = parser.parse_args()
nens = argsin.nens
metric = argsin.metric
figdir = Path(f"fig{metric}/ne{nens}")
if not figdir.exists(): figdir.mkdir()

# load results
nensbase = 8
ds_all = {}
for vt in vtlist:
    ds_asa = xr.open_dataset(datadir/f'asa{metric}_vt{vt}nens{nensbase}.nc')
    print(ds_asa)
    ds_dict = {'asa':ds_asa}

    for solver in enasas:
        ds = xr.open_dataset(datadir/f'{solver}{metric}_vt{vt}nens{nens}.nc')
        ds_dict[solver] = ds

    ds_all[vt] = ds_dict

plt.rcParams['boxplot.medianprops.color'] = 'k'
plt.rcParams['boxplot.flierprops.marker'] = '.'
bwidth = 0.75 / len(ds_dict.keys())
xoffset = 0.5 * bwidth * (len(ds_dict.keys()) - 1)
fig0, ax0 = plt.subplots(figsize=[8,6],constrained_layout=True)
fig1, ax1 = plt.subplots(figsize=[8,6],constrained_layout=True)
patchs = []
if metric=='':
    ylims = [-1.1,1.1]
else:
    ylims = [-0.08,0.02]
for j,vt in enumerate(vtlist):
    pos0 = j+1 - xoffset
    ds_dict = ds_all[vt]
    for i,key in enumerate(ds_dict.keys()):
        marker_style['markerfacecolor']=colors[key]
        marker_style['markeredgecolor']='k'
        flierprops={'markeredgecolor':colors[key]}
        data1 = ds_dict[key].res_nl.values
        data2 = ds_dict[key].res_tl.values
        bplot1 = ax0.boxplot(data1,positions=[pos0],widths=bwidth,\
            patch_artist=True,flierprops=flierprops)#,whis=(0,100))
        for patch in bplot1['boxes']:
            patch.set_facecolor(colors[key])
            #patch.set_alpha(0.3)
        nlower = np.sum(data1<ylims[0])
        nupper = np.sum(data1>ylims[1])
        if nlower>0:
            ax0.text(pos0,0.05,f'{nlower}',transform=ax0.get_xaxis_transform(),\
                ha='center',size='small',weight='bold',color=colors[key])
        if nupper>0:
            ax0.text(pos0,0.95,f'{nupper}',transform=ax0.get_xaxis_transform(),\
                ha='center',size='small',weight='bold',color=colors[key])
        #ym = data1.mean()
        #p,=axs[0].plot([pos0],[ym],lw=0.0,marker=markers[key],ms=10,c=colors[key],**marker_style)
        bplot2 = ax1.boxplot(data2,positions=[pos0],widths=bwidth,\
            patch_artist=True,flierprops=flierprops) #,whis=(0,100))
        for patch in bplot2['boxes']:
            patch.set_facecolor(colors[key])
            #patch.set_alpha(0.3)
        nlower = np.sum(data2<ylims[0])
        nupper = np.sum(data2>ylims[1])
        if nlower>0:
            ax1.text(pos0,0.05,f'{nlower}',transform=ax1.get_xaxis_transform(),\
                ha='center',size='small',weight='bold',color=colors[key])
        if nupper>0:
            ax1.text(pos0,0.95,f'{nupper}',transform=ax1.get_xaxis_transform(),\
                ha='center',size='small',weight='bold',color=colors[key])#ym = data2.mean()
        #p,=axs[1].plot([pos0],[ym],lw=0.0,marker=markers[key],ms=10,c=colors[key],**marker_style)
        pos0 = pos0 + bwidth
        if j==0:
            patchs.append(Patch(color=colors[key],label=key))
for ax in [ax0,ax1]:
    ax.set_xticks(np.arange(1,len(vtlist)+1))
    ax.set_xticklabels([f'FT{v}' for v in vtlist])
    ax.grid(axis='y')
    ax.set_ylim(ylims)
    ax.legend(handles=patchs,loc='upper left',bbox_to_anchor=(1.0,1.0))
ax0.set_title(r'Nonlinear forecast response: $\frac{J(M(\mathbf{x}_0+\delta\mathbf{x}_0^\mathrm{opt}))-J(\mathbf{x}_T)}{J(\mathbf{x}_T)}$'+f', {nens} member')
ax1.set_title(r'Tangent linear response: $\frac{J(\mathbf{x}_T+\mathbf{M}\delta\mathbf{x}_0^\mathrm{opt})-J(\mathbf{x}_T)}{J(\mathbf{x}_T)}$'+f', {nens} member')
fig0.savefig(figdir/f"res_nl_nens{nens}.png",dpi=300)
fig1.savefig(figdir/f"res_tl_nens{nens}.png",dpi=300)
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=[8,6],constrained_layout=True)
patchs = []
for j,vt in enumerate(vtlist):
    pos0 = j+1 - xoffset
    ds_dict = ds_all[vt]
    for i,key in enumerate(ds_dict.keys()):
        if key=='asa': continue
        marker_style['markerfacecolor']=colors[key]
        marker_style['markeredgecolor']='k'
        flierprops={'markeredgecolor':colors[key]}
        data = ds_dict[key].corrdJ.values
        bplot = ax.boxplot(data,positions=[pos0],widths=bwidth,\
            patch_artist=True,flierprops=flierprops) #,whis=(0,100))
        for patch in bplot['boxes']:
            patch.set_facecolor(colors[key])
            #patch.set_alpha(0.3)
        #ym = data.mean()
        #p, = ax.plot([pos0],[ym],lw=0.0,marker=markers[key],ms=10,c=colors[key],**marker_style)
        if j==0:
            patchs.append(Patch(color=colors[key],label=key))
        pos0 = pos0 + bwidth
ax.set_xticks(np.arange(1,len(vtlist)+1))
ax.set_xticklabels([f'FT{v}' for v in vtlist])
#ax.set_ylabel('spatial correlation')
ax.grid(axis='y')
ax.set_ylim(-1.0,1.0)
ax.legend(handles=patchs,loc='upper left',bbox_to_anchor=(1.0,1.0))
fig.suptitle(r'Spatial correlation of $\frac{\partial J}{\partial \mathbf{x}_0}$'+f' against ASA, {nens} member')
fig.savefig(figdir/f"corr_nens{nens}.png",dpi=300)
#plt.show()
plt.close()
