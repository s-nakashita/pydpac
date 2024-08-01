import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 14
import xarray as xr
from sklearn.linear_model import LinearRegression
from pathlib import Path
import sys
import argparse

datadir = Path('data')

cmap = plt.get_cmap('tab10')
enasas = ['minnorm','diag','pcr','ridge','pls']
colors = {'asa':cmap(0)}
markers = {'asa':'*'}
markerlist = ['*','o','v','s','P','X','d']
marker_style=dict(markerfacecolor='none')
nclist = [2,4,8,16,24]
i=1
for nc in nclist:
    colors[f'nc{nc}'] = cmap(i)
    markers[f'nc{nc}'] = markerlist[i]
    i+=1
colors['all'] = cmap(i)
markers['all'] = cmap(i)

parser = argparse.ArgumentParser()
parser.add_argument("-vt","--vt",type=int,default=24,\
    help="verification time (hours)")
parser.add_argument("-ne","--nens",type=int,default=8,\
    help="ensemble size")
parser.add_argument("-s","--solver",type=str,\
    help="EnASA type (minnorm,pcr,pls)")
parser.add_argument("-m","--metric",type=str,default="",\
    help="forecast metric type")
parser.add_argument("-l","--lag",type=int,default=0,\
    help="lag of time-lag ensemble")
argsin = parser.parse_args()
vt = argsin.vt # hours
ioffset = vt // 6
nens = argsin.nens
metric = argsin.metric
solver = argsin.solver
lag = argsin.lag
figdir = Path(f"fig{metric}/vt{vt}ne{nens}/{solver}")
if lag>0:
    figdir = Path(f"fig{metric}/vt{vt}ne{nens}lag{lag}/{solver}")
if not figdir.exists(): figdir.mkdir()

# load results
nensbase = 8
ds_asa = xr.open_dataset(datadir/f'asa{metric}_vt{vt}nens{nensbase}.nc')
print(ds_asa)
ds_dict = {'asa':ds_asa}

ds_enasa = {}
for nc in nclist:
    if nc >= nens*(lag+1):
        if lag>0:
            ds = xr.open_dataset(datadir/f'{solver}{metric}_vt{vt}nens{nens}lag{lag}.nc')
        else:
            ds = xr.open_dataset(datadir/f'{solver}{metric}_vt{vt}nens{nens}.nc')
        ds_enasa[f'all'] = ds
        ds_dict[f'all'] = ds
        break
    else:
        if lag>0:
            ds = xr.open_dataset(datadir/f'{solver}{metric}nc{nc}_vt{vt}nens{nens}lag{lag}.nc')
        else:
            ds = xr.open_dataset(datadir/f'{solver}{metric}nc{nc}_vt{vt}nens{nens}.nc')
        ds_enasa[f'nc{nc}'] = ds
        ds_dict[f'nc{nc}'] = ds

nrows=2
ncols=int(np.ceil(len(ds_dict.keys())/2))
fig, axs = plt.subplots(ncols=ncols,nrows=nrows,figsize=[10,8],constrained_layout=True)
for i,key in enumerate(ds_dict.keys()):
    res_nl = ds_dict[key].res_nl.values
    res_tl = ds_dict[key].res_tl.values
    ax = axs.flatten()[i]
    ax.plot(res_nl,res_tl,marker=markers[key],lw=0.0, c=colors[key],#ms=10,\
        **marker_style)
    ax.set_title(key)
#ymin, ymax = ax.get_ylim()
ymin = -1.1
ymax = 1.1
line = np.linspace(ymin,ymax,100)
for ax in axs.flatten():
    ax.plot(line,line,color='k',zorder=0)
    ax.grid()
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(ymin,ymax)
    ax.set_aspect(1.0)
axs[-1,1].set_xlabel(r'NLM: $\frac{J(M(\mathbf{x}_0+\delta\mathbf{x}_0^\mathrm{opt}))-J(\mathbf{x}_T)}{J(\mathbf{x}_T)}$')
axs[0,0].set_ylabel(r'TLM: $\frac{J(\mathbf{x}_T+\mathbf{M}\delta\mathbf{x}_0^\mathrm{opt})-J(\mathbf{x}_T)}{J(\mathbf{x}_T)}$')
if lag>0:
    fig.suptitle(r'$\delta J/J$'+f', {solver} FT{vt} {nens}*{lag+1} member')
    fig.savefig(figdir/f"res_vt{vt}nens{nens}lag{lag}.png",dpi=300)
else:
    fig.suptitle(r'$\delta J/J$'+f', {solver} FT{vt} {nens} member')
    fig.savefig(figdir/f"res_vt{vt}nens{nens}.png",dpi=300)
#plt.show()

fig, ax = plt.subplots(figsize=[10,8],constrained_layout=True)
for i,key in enumerate(ds_enasa.keys()):
    marker_style['markerfacecolor']=colors[key]
    marker_style['markeredgecolor']='k'
    x = ds_enasa[key].rmsdJ.values
    y = ds_enasa[key].rmsdx.values
    ax.plot(x,y,lw=0.0,marker='.',ms=5,c=colors[key],zorder=0)
    xm = x.mean()
    ym = y.mean()
    ax.plot([xm],[ym],lw=0.0,marker=markers[key],ms=10,c=colors[key],label=f'{key}=({xm:.2e},{ym:.2e})',**marker_style)
ax.set_xlabel('dJdx0')
ax.set_ylabel('dxopt')
ax.grid()
ax.legend()
if lag>0:
    fig.suptitle(f'RMSD against ASA, {solver} FT{vt} {nens}*{lag+1} member')
    fig.savefig(figdir/f"rms_vt{vt}nens{nens}lag{lag}.png",dpi=300)
else:
    fig.suptitle(f'RMSD against ASA, {solver} FT{vt} {nens} member')
    fig.savefig(figdir/f"rms_vt{vt}nens{nens}.png",dpi=300)
#plt.show()
plt.close()

fig, ax = plt.subplots(figsize=[8,8],constrained_layout=True)
for i,key in enumerate(ds_enasa.keys()):
    marker_style['markerfacecolor']=colors[key]
    marker_style['markeredgecolor']='k'
    data = ds_enasa[key].corrdJ.values
    bplot = ax.boxplot(data,positions=[i+1],patch_artist=True,whis=(0,100))
    for patch in bplot['boxes']:
        patch.set_facecolor(colors[key])
        patch.set_alpha(0.3)
    ym = data.mean()
    ax.plot([i+1],[ym],lw=0.0,marker=markers[key],ms=10,c=colors[key],**marker_style)
ax.set_xticks(np.arange(1,len(ds_enasa.keys())+1))
ax.set_xticklabels(ds_enasa.keys())
ax.set_ylabel('spatial correlation')
ax.grid(axis='y')
#ax.legend()
if lag>0:
    fig.suptitle(f'Spatial correlation of dJdx0 against ASA\n{solver} FT{vt} {nens}*{lag+1} member')
    fig.savefig(figdir/f"corr_vt{vt}nens{nens}lag{lag}.png",dpi=300)
else:
    fig.suptitle(f'Spatial correlation of dJdx0 against ASA\n{solver} FT{vt} {nens} member')
    fig.savefig(figdir/f"corr_vt{vt}nens{nens}.png",dpi=300)
#plt.show()
plt.close()
