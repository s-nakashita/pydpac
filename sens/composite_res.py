import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 14
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
import xarray as xr
from sklearn.linear_model import LinearRegression
from pathlib import Path
import argparse

datadir = Path('data')

cmap = plt.get_cmap('tab10')
enasas = ['minnorm','diag','ridge','pcr','pls']
colors = {'asa':cmap(0),'minnorm':cmap(1),'diag':cmap(2),'pcr':cmap(3),'ridge':cmap(4),'pls':cmap(5)}
markers = {'asa':'*','minnorm':'o','diag':'v','pcr':'s','ridge':'P','pls':'X'}
ms = {'asa':8,'minnorm':5,'diag':5,'pcr':5,'ridge':5,'pls':5}
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
nensbase = 8

figdir = Path(f"fig{metric}/res")
if not figdir.exists(): figdir.mkdir()

# load results
ds_calc = xr.open_dataset(datadir/f'res_calc{metric}_vt{vt}nens{nens}.nc')
ds_dict = {}
ds_dict['calc'] = ds_calc
ds_asa = xr.open_dataset(datadir/f'res_asa{metric}_vt{vt}nens{nens}.nc')
print(ds_asa)
ds_dict['asa'] = ds_asa
for solver in enasas:
    ds = xr.open_dataset(datadir/f'res_{solver}{metric}_vt{vt}nens{nens}.nc')
    ds_dict[solver] = ds

# x0i
ix0s = ds_calc.ix0.values
x0i = ds_calc.x0i
nx = x0i.shape[1]
nxh = nx // 2
x0imean = np.zeros(nx)
x0istd = np.zeros(nx)
nsample = 0
for isample, ix in enumerate(ix0s):
    x0itmp = np.roll(x0i[isample],nxh-ix)
    x0imean = x0imean + x0itmp
    x0istd = x0istd + x0itmp**2
    nsample += 1
x0imean = x0imean / nsample
x0istd = x0istd/nsample - x0imean**2
x0istd = np.where(x0istd>0.0,np.sqrt(x0istd),0.0)
fig, ax = plt.subplots(figsize=[6,4],constrained_layout=True)
ax.plot(np.arange(nx),x0imean,marker='^',lw=2.0)
ax.fill_between(np.arange(nx),x0imean-x0istd,x0imean+x0istd,alpha=0.5)
ax.set_title(r'$\Delta \mathbf{x}_{0i}$'+f' FT{vt} {nens} member')
ax.grid()
fig.savefig(figdir/f'x0i_vt{vt}nens{nens}.png')
#plt.show()
plt.close()

## ASA center points
ds_asa = xr.open_dataset(datadir/f'asa{metric}_vt{vt}nens{nensbase}.nc')
ics = ds_asa.ic.values
ncycle = 100
ngrid = nsample // ncycle
print(f"nsample={nsample} ncycle={ncycle} ngrid={ngrid}")
isample = 0
far = []
near = []
for icycle in range(ncycle):
    ic = ics[icycle]
    for igrid in range(ngrid):
        ix0 = ix0s[isample]
        if np.abs(ic - ix0)<=10:
            far.append(False)
            near.append(True)
        else:
            far.append(True)
            near.append(False)
        isample += 1
print(f"far={np.sum(far)} near={np.sum(near)}")

resmul_far = dict()
resmul_near = dict()
resuni_far = dict()
resuni_near = dict()

for key in ds_dict.keys():
    resmul_far[key] = ds_dict[key].resmul[far]
    resmul_near[key] = ds_dict[key].resmul[near]
    resuni_far[key] = ds_dict[key].resuni[far]
    resuni_near[key] = ds_dict[key].resuni[near]

for i in range(2):
    if i==0:
        figname = 'far'
        resmul_dict = resmul_far
        resuni_dict = resuni_far
    else:
        figname = 'near'
        resmul_dict = resmul_near
        resuni_dict = resuni_near
    fig, axs = plt.subplots(figsize=[8,6],ncols=2,nrows=2,constrained_layout=True)
    resmul_calc = resmul_dict['calc']
    resuni_calc = resuni_dict['calc']
    for key in resmul_dict.keys():
        if key=='calc': continue
        figdir1 = figdir/key
        axs[0,0].plot(resmul_dict[key],resmul_calc,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
        axs[0,1].plot(resuni_dict[key],resmul_calc,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
        axs[1,0].plot(resmul_dict[key],resuni_calc,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
        axs[1,1].plot(resuni_dict[key],resuni_calc,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
        fig1, axs1 = plt.subplots(figsize=[6,6],ncols=2,nrows=2,constrained_layout=True)
        x = resmul_dict[key].values.reshape(-1,1)
        y = resmul_calc.values
        reg = LinearRegression().fit(x,y)
        r2 = reg.score(x,y)
        axs1[0,0].plot(resmul_dict[key],resmul_calc,lw=0.0,marker=markers[key],c=colors[key],**marker_style)
        axs1[0,0].plot(x,reg.predict(x),c=colors[key],label=f'y={reg.coef_[0]:.2f}x+{reg.intercept_:.2f}\nr^2={r2:.2e}')
        axs1[0,1].plot(resuni_dict[key],resmul_calc,lw=0.0,marker=markers[key],c=colors[key],**marker_style)
        axs1[1,0].plot(resmul_dict[key],resuni_calc,lw=0.0,marker=markers[key],c=colors[key],**marker_style)
        x = resuni_dict[key].values.reshape(-1,1)
        y = resuni_calc.values
        reg = LinearRegression().fit(x,y)
        r2 = reg.score(x,y)
        axs1[1,1].plot(resuni_dict[key],resuni_calc,lw=0.0,marker=markers[key],c=colors[key],**marker_style)
        axs1[1,1].plot(x,reg.predict(x),c=colors[key],label=f'y={reg.coef_[0]:.2f}x+{reg.intercept_:.2f}\nr^2={r2:.2e}')
        for ax in axs1.flatten():
            ymin, ymax = ax.get_ylim()
            ylim = max(-ymin,ymax)
            ax.set_xlim(-ylim,ylim)
            ax.set_ylim(-ylim,ylim)
            line = np.linspace(-ylim,ylim,100)
            ax.plot(line,line,color='k',zorder=0)
            #ax.legend()
            #ax.set_title(key)
            ax.grid()
            ax.set_aspect(1.0)
        axs1[0,0].legend()
        axs1[1,1].legend()
        #axs1[0,0].set_xlabel('predicted (multivariate)')
        axs1[0,0].set_ylabel('measured (multivatiate)')
        #axs1[0,1].set_xlabel('predicted (univariate)')
        #axs1[0,1].set_ylabel('measured (multivariate)')
        axs1[1,0].set_xlabel('predicted (multivariate)')
        axs1[1,0].set_ylabel('measured (univariate)')
        axs1[1,1].set_xlabel('predicted (univariate)')
        #axs1[1,1].set_ylabel('measured (univariate)')
        nsample = resmul_dict[key].size
        fig1.suptitle(r'$\Delta J(\Delta x_{0i})$'+f' {figname} {key} vt={vt}h, Nens={nens} #{nsample}')
        fig1.savefig(figdir1/f'resunimul_{figname}_{key}_vt{vt}ne{nens}.png')
        #plt.show()
        plt.close()
    for ax in axs.flatten():
        ymin, ymax = ax.get_ylim()
        ylim = max(-ymin,ymax)
        ax.set_xlim(-ylim,ylim)
        ax.set_ylim(-ylim,ylim)
        line = np.linspace(-ylim,ylim,100)
        ax.plot(line,line,color='k',zorder=0)
        #ax.legend()
        #ax.set_title(key)
        ax.grid()
        ax.set_aspect(1.0)
    axs[0,1].legend(loc='upper left',bbox_to_anchor=(1.01,1.0))
    #axs[0,0].set_xlabel('predicted (multivariate)')
    axs[0,0].set_ylabel('measured (multivatiate)')
    #axs[0,1].set_xlabel('predicted (univariate)')
    #axs[0,1].set_ylabel('measured (multivariate)')
    axs[1,0].set_xlabel('predicted (multivariate)')
    axs[1,0].set_ylabel('measured (univariate)')
    axs[1,1].set_xlabel('predicted (univariate)')
    #axs[1,1].set_ylabel('measured (univariate)')
    nsample = resmul_dict['calc'].size
    fig.suptitle(r'$\Delta J(\Delta x_{0i})$'+f' {figname} vt={vt}h, Nens={nens} #{nsample}')
    fig.savefig(figdir/f'resunimul_{figname}_vt{vt}ne{nens}.png')
    #plt.show()
    plt.close()
