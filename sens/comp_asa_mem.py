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
enasas = ['minnorm','diag','ridge','pcr','pls']
colors = {'asa':cmap(0),'minnorm':cmap(1),'diag':cmap(2),'pcr':cmap(3),'ridge':cmap(4),'pls':cmap(5)}
markers = {'asa':'*','minnorm':'o','diag':'v','pcr':'s','ridge':'P','pls':'X'}
marker_style=dict(markerfacecolor='none')
nelist = [8,16,24,32,40]

parser = argparse.ArgumentParser()
parser.add_argument("-vt","--vt",type=int,default=24,\
    help="verification time (hours)")
parser.add_argument("-m","--metric",type=str,default="",\
    help="forecast metric type")
argsin = parser.parse_args()
vt = argsin.vt
metric = argsin.metric
figdir = Path(f"fig/vt{vt}{metric}")
if not figdir.exists(): figdir.mkdir()

# load results
nensbase = 8
ds_all = {}
ds_asa = xr.open_dataset(datadir/f'asa{metric}_vt{vt}nens{nensbase}.nc')
print(ds_asa)
ds_all['asa'] = ds_asa
for nens in nelist:
    ds_dict = {}
    for solver in enasas:
        ds = xr.open_dataset(datadir/f'{solver}{metric}_vt{vt}nens{nens}.nc')
        ds_dict[solver] = ds

    ds_all[nens] = ds_dict

plt.rcParams['boxplot.medianprops.color'] = 'k'
plt.rcParams['boxplot.flierprops.marker'] = '.'
bwidth = 0.75 / (len(ds_dict.keys())+1)
xoffset = 0.5 * bwidth * (len(ds_dict.keys()))
fig0, ax0 = plt.subplots(figsize=[8,6],constrained_layout=True)
fig1, ax1 = plt.subplots(figsize=[8,6],constrained_layout=True)
patchs = []
for j,nens in enumerate(nelist):
    pos0 = j+1 - xoffset
    if j==0:
        key='asa'
        marker_style['markerfacecolor']=colors[key]
        marker_style['markeredgecolor']='k'
        flierprops={'markeredgecolor':colors[key]}
        data1 = ds_all[key].res_nl.values
        data2 = ds_all[key].res_tl.values
        bplot1 = ax0.boxplot(data1,positions=[pos0],widths=bwidth,\
            patch_artist=True,flierprops=flierprops)#,whis=(0,100))
        for patch in bplot1['boxes']:
            patch.set_facecolor(colors[key])
            #patch.set_alpha(0.3)
        nout = np.sum(data1>1.1)
        if nout>0:
            ax0.text(pos0,0.95,f'{nout}',transform=ax0.get_xaxis_transform(),\
                ha='center',size='small',weight='bold',color=colors[key])
        #ym = data1.mean()
        #p,=axs[0].plot([pos0],[ym],lw=0.0,marker=markers[key],ms=10,c=colors[key],**marker_style)
        bplot2 = ax1.boxplot(data2,positions=[pos0],widths=bwidth,\
            patch_artist=True,flierprops=flierprops) #,whis=(0,100))
        for patch in bplot2['boxes']:
            patch.set_facecolor(colors[key])
            #patch.set_alpha(0.3)
        nout = np.sum(data2>1.1)
        if nout>0:
            ax1.text(pos0,0.95,f'{nout}',transform=ax1.get_xaxis_transform(),\
                ha='center',size='x-small',weight='bold',color=colors[key])
        #ym = data2.mean()
        #p,=axs[1].plot([pos0],[ym],lw=0.0,marker=markers[key],ms=10,c=colors[key],**marker_style)
        pos0 = pos0 + bwidth
        if j==0:
            patchs.append(Patch(color=colors[key],label=key))
    ds_dict = ds_all[nens]
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
        nout = np.sum(data1>1.1)
        if nout>0:
            ax0.text(pos0,0.95,f'{nout}',transform=ax0.get_xaxis_transform(),\
                ha='center',size='small',weight='bold',color=colors[key])
        #ym = data1.mean()
        #p,=axs[0].plot([pos0],[ym],lw=0.0,marker=markers[key],ms=10,c=colors[key],**marker_style)
        bplot2 = ax1.boxplot(data2,positions=[pos0],widths=bwidth,\
            patch_artist=True,flierprops=flierprops) #,whis=(0,100))
        for patch in bplot2['boxes']:
            patch.set_facecolor(colors[key])
            #patch.set_alpha(0.3)
        nout = np.sum(data2>1.1)
        if nout>0:
            ax1.text(pos0,0.95,f'{nout}',transform=ax1.get_xaxis_transform(),\
                ha='center',size='x-small',weight='bold',color=colors[key])
        #ym = data2.mean()
        #p,=axs[1].plot([pos0],[ym],lw=0.0,marker=markers[key],ms=10,c=colors[key],**marker_style)
        pos0 = pos0 + bwidth
        if j==0:
            patchs.append(Patch(color=colors[key],label=key))
for ax in [ax0,ax1]:
    ax.set_xticks(np.arange(1,len(nelist)+1))
    ax.set_xticklabels([f'mem{ne}' for ne in nelist])
    ax.grid(axis='y')
    if metric == '':
        ax.set_ylim(-1.1,1.1)
    ax.legend(handles=patchs,loc='upper left',bbox_to_anchor=(1.0,1.0))
ax0.set_title(r'Nonlinear forecast response: $\frac{J(M(\mathbf{x}_0+\delta\mathbf{x}_0^\mathrm{opt}))-J(\mathbf{x}_T)}{J(\mathbf{x}_T)}$'+f', FT{vt}')
ax1.set_title(r'Tangent linear response: $\frac{J(\mathbf{x}_T+\mathbf{M}\delta\mathbf{x}_0^\mathrm{opt})-J(\mathbf{x}_T)}{J(\mathbf{x}_T)}$'+f', FT{vt}')
fig0.savefig(figdir/f"res_nl_vt{vt}.png",dpi=300)
fig1.savefig(figdir/f"res_tl_vt{vt}.png",dpi=300)
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=[8,6],constrained_layout=True)
patchs = []
for j,nens in enumerate(nelist):
    pos0 = j+1 - xoffset
    ds_dict = ds_all[nens]
    for i,key in enumerate(ds_dict.keys()):
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
ax.set_xticks(np.arange(1,len(nelist)+1))
ax.set_xticklabels([f'mem{ne}' for ne in nelist])
#ax.set_ylabel('spatial correlation')
ax.grid(axis='y')
ax.set_ylim(-1.0,1.0)
ax.legend(handles=patchs,loc='upper left',bbox_to_anchor=(1.0,1.0))
fig.suptitle(r'Spatial correlation of $\frac{\partial J}{\partial \mathbf{x}_0}$'+f' against ASA, FT{vt}')
fig.savefig(figdir/f"corr_vt{vt}.png",dpi=300)
#plt.show()
plt.close()
