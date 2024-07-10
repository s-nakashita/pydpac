import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 14
import xarray as xr
from sklearn.linear_model import LinearRegression
from pathlib import Path
import sys

datadir = Path('data')
figdir = Path('fig')

cmap = plt.get_cmap('tab10')
enasas = ['minnorm','diag','pcr','ridge','pls']
enasa_colors = {'minnorm':cmap(1),'diag':cmap(2),'pcr':cmap(3),'ridge':cmap(4),'pls':cmap(5)}
enasa_markers = {'minnorm':'o','diag':'v','pcr':'s','ridge':'P','pls':'X'}

vt = 24
if len(sys.argv)>1:
    vt = int(sys.argv[1])
nens = 8
if len(sys.argv)>2:
    nens = int(sys.argv[2])
# load results
ds_asa = xr.open_dataset(datadir/f'asa_vt{vt}nens{nens}.nc')
print(ds_asa)

ds_enasa = {}
for solver in enasas:
    ds = xr.open_dataset(datadir/f'{solver}_vt{vt}nens{nens}.nc')
    ds_enasa[solver] = ds

ncols = 2
nrows = 3
figtl, axstl = plt.subplots(figsize=[8,10],nrows=nrows,ncols=ncols,constrained_layout=True)
fignl, axsnl = plt.subplots(figsize=[8,10],nrows=nrows,ncols=ncols,constrained_layout=True)

ymin = -1.1
ymax = 1.1
line = np.linspace(ymin,ymax,100)
for axtl, axnl, solver in zip(axstl.flatten(),axsnl.flatten(),enasas):
    marker_style = dict(markerfacecolor=enasa_colors[solver],markeredgecolor='k',ms=10)
    x = ds_asa.res_tl.values
    y = ds_enasa[solver].res_tl.values
    axtl.plot(x,y,lw=0.0,c=enasa_colors[solver],marker='.',ms=5,zorder=0,label=solver)
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
    axnl.plot(x,y,lw=0.0,c=enasa_colors[solver],marker='.',ms=5,zorder=0,label=solver)
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
plt.show()