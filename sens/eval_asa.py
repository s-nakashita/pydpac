import numpy as np 
import matplotlib.pyplot as plt 
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

parser = argparse.ArgumentParser()
parser.add_argument("-vt","--vt",type=int,default=24,\
    help="verification time (hours)")
parser.add_argument("-ne","--nens",type=int,default=8,\
    help="ensemble size")
parser.add_argument("-m","--metric",type=str,default="",\
    help="forecast metric type")
argsin = parser.parse_args()
vt = argsin.vt # hours
ioffset = vt // 6
nens = argsin.nens
metric = argsin.metric
figdir = Path(f"fig{metric}/vt{vt}ne{nens}")
if not figdir.exists(): figdir.mkdir()

# load results
nensbase = 8
ds_asa = xr.open_dataset(datadir/f'asa{metric}_vt{vt}nens{nensbase}.nc')
print(ds_asa)
ds_dict = {'asa':ds_asa}

ds_enasa = {}
for solver in enasas:
    ds = xr.open_dataset(datadir/f'{solver}{metric}_vt{vt}nens{nens}.nc')
    ds_enasa[solver] = ds
    ds_dict[solver] = ds

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
if metric=='_en':
    ymin, ymax = axs[0,0].get_ylim()
    for ax in axs.flatten()[1:]:
        ymintmp, ymaxtmp = ax.get_ylim()
        ymin = min(ymin,ymintmp)
        ymax = max(ymax,ymaxtmp)
else:
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
fig.suptitle(r'$\delta J/J$'+f', FT{vt} {nens} member')
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
fig.suptitle(f'RMSD against ASA, FT{vt} {nens} member')
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
fig.suptitle(f'Spatial correlation of dJdx0 against ASA, FT{vt} {nens} member')
fig.savefig(figdir/f"corr_vt{vt}nens{nens}.png",dpi=300)
#plt.show()
plt.close()

ncols = 2
nrows = 3
figtl, axstl = plt.subplots(figsize=[8,10],nrows=nrows,ncols=ncols,constrained_layout=True)
fignl, axsnl = plt.subplots(figsize=[8,10],nrows=nrows,ncols=ncols,constrained_layout=True)

#ymin = -1.1
#ymax = 1.1
line = np.linspace(ymin,ymax,100)
for axtl, axnl, solver in zip(axstl.flatten(),axsnl.flatten(),enasas):
    marker_style = dict(markerfacecolor=colors[solver],markeredgecolor='k',ms=10)
    x = ds_asa.res_tl.values
    y = ds_enasa[solver].res_tl.values
    axtl.plot(x,y,lw=0.0,c=colors[solver],marker='.',ms=5,zorder=0,label=solver)
    reg = LinearRegression().fit(x.reshape(-1,1),y)
    coef = reg.coef_[0]
    intercept = reg.intercept_
    r2 = reg.score(x.reshape(-1,1),y)
    yest = reg.predict(line.reshape(-1,1))
    axtl.plot(line,yest,c='k')
    axtl.set_title(f'y={coef:.2f}x+{intercept:.2f}, '+r'r$^2$='+f'{r2:.2e}')
    axtl.set_ylabel(solver)
    #axs[0].plot(x.mean(),y.mean(),lw=0.0,c=enasa_colors[solver],marker=enasa_markers[solver],label=f'{solver}=({x.mean():.2e},{y.mean():.2e})',**marker_style)
    x = ds_asa.res_nl.values
    y = ds_enasa[solver].res_nl.values
    axnl.plot(x,y,lw=0.0,c=colors[solver],marker='.',ms=5,zorder=0,label=solver)
    reg = LinearRegression().fit(x.reshape(-1,1),y)
    coef = reg.coef_[0]
    intercept = reg.intercept_
    r2 = reg.score(x.reshape(-1,1),y)
    yest = reg.predict(line.reshape(-1,1))
    axnl.plot(line,yest,c='k')
    axnl.set_title(f'y={coef:.2f}x+{intercept:.2f}, '+r'r$^2$='+f'{r2:.2e}')
    axnl.set_ylabel(solver)
    #axs[1].plot(x.mean(),y.mean(),lw=0.0,c=enasa_colors[solver],marker=enasa_markers[solver],label=f'{solver}=({x.mean():.2e},{y.mean():.2e})',**marker_style)
axstl[-1,0].set_xlabel('ASA')
#axs[0].set(title=r'TLM: $\frac{J(\mathbf{x}_T+\mathbf{M}\delta\mathbf{x}_0^\mathrm{opt})-J(\mathbf{x}_T)}{J(\mathbf{x}_T)}$',xlabel='ASA',ylabel='EnASA')
axsnl[-1,0].set_xlabel('ASA')
#axs[1].set(title=r'NLM: $\frac{J(M(\mathbf{x}_0+\delta\mathbf{x}_0^\mathrm{opt}))-J(\mathbf{x}_T)}{J(\mathbf{x}_T)}$',xlabel='ASA',ylabel='EnASA')
for ax in np.concatenate((axstl.flatten()[:-1],axsnl.flatten()[:-1])):
    ax.plot(line,line,color='gray',zorder=0)
    ax.grid()
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(ymin,ymax)
    ax.set_aspect(1.0)
    #ax.legend(loc='lower right',handletextpad=0.) #,bbox_to_anchor=(1.01,1.0))
axstl[-1,-1].remove()
axsnl[-1,-1].remove()
figtl.suptitle(r'TLM: $\frac{J(\mathbf{x}_T+\mathbf{M}\delta\mathbf{x}_0^\mathrm{opt})-J(\mathbf{x}_T)}{J(\mathbf{x}_T)}$'+f" FT{vt} {nens} member")
fignl.suptitle(r'NLM: $\frac{J(M(\mathbf{x}_0+\delta\mathbf{x}_0^\mathrm{opt}))-J(\mathbf{x}_T)}{J(\mathbf{x}_T)}$'+f" FT{vt} {nens} member")
figtl.savefig(figdir/f"restl_vs_asa_vt{vt}nens{nens}.png",dpi=300)
fignl.savefig(figdir/f"resnl_vs_asa_vt{vt}nens{nens}.png",dpi=300)
#plt.show()