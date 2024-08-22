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
enasas = ['minnorm','diag','ridge','pcr','pls']#,'std']
colors = {'asa':cmap(0),'minnorm':cmap(1),'diag':cmap(2),'pcr':cmap(3),'ridge':cmap(4),'pls':cmap(5),'std':cmap(6)}
markers = {'asa':'*','minnorm':'o','diag':'v','pcr':'s','ridge':'P','pls':'X','std':'^'}
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
figdir = Path(f"fig{metric}/res_hess")
if not figdir.exists(): figdir.mkdir()

# load results
nensbase = 8
ds_all = {}
for nens in nelist:
    ds_dict = {}
    ds_asa = xr.open_dataset(datadir/f'res_hess_asa{metric}_vt{vt}nens{nens}.nc')
    print(ds_asa)
    ds_dict['asa'] = ds_asa
    for solver in enasas:
        ds = xr.open_dataset(datadir/f'res_hess_{solver}{metric}_vt{vt}nens{nens}.nc')
        ds_dict[solver] = ds
    ds = xr.open_dataset(datadir/f'std{metric}_vt{vt}nens{nens}.nc')
    ds_dict['std'] = ds
    ds_all[nens] = ds_dict

plt.rcParams['boxplot.medianprops.color'] = 'k'
plt.rcParams['boxplot.flierprops.marker'] = '.'
bwidth = 3.0 / (len(ds_dict.keys())+1)
xoffset = 0.5 * bwidth * (len(ds_dict.keys()))
fig0, ax0 = plt.subplots(figsize=[10,6],constrained_layout=True)
patchs = []
if metric=='':
    ylims = [-1.1,1.1]
else:
    ylims = [-0.08,0.02]
xticks = []
for j,nens in enumerate(nelist):
    xticks.append(4*j+1)
    pos0 = xticks[j] - xoffset
    ds_dict = ds_all[nens]
    for i,key in enumerate(ds_dict.keys()):
        marker_style['markerfacecolor']=colors[key]
        marker_style['markeredgecolor']='k'
        flierprops={'markeredgecolor':colors[key]}
        if key=='std':
            data1 = ds_dict[key].res_nl.values
        else:
            data1 = ds_dict[key].rescalcm.values
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
        pos0 = pos0 + bwidth
        if j==0:
            patchs.append(Patch(color=colors[key],label=key))
ax0.set_xticks(xticks)
ax0.set_xticklabels([f'mem{ne}' for ne in nelist])
ax0.grid(axis='y')
ax0.set_ylim(ylims)
ax0.legend(handles=patchs,loc='upper left',bbox_to_anchor=(1.0,1.0))
ax0.set_title(r'Nonlinear forecast response: $\frac{J(M(\mathbf{x}_0+\delta\mathbf{x}_0^*))-J(\mathbf{x}_T)}{J(\mathbf{x}_T)}$'+f', FT{vt}')
fig0.savefig(figdir/f"rescalcm_vt{vt}.png",dpi=300)
plt.show()
plt.close()
