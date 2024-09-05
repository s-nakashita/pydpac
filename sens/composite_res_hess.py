import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 14
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from confidence_ellipse import confidence_ellipse
from matplotlib.patches import Patch
import xarray as xr
from sklearn.linear_model import LinearRegression
from pathlib import Path
import argparse

datadir = Path('data')

cmap = plt.get_cmap('tab10')
enasas = ['minnorm','diag','ridge','pcr','pls']
colors = {'asa':cmap(0),'minnorm':cmap(1),'diag':cmap(2),'pcr':cmap(3),'ridge':cmap(4),'pls':cmap(5),'std':cmap(6)}
markers = {'asa':'*','minnorm':'o','diag':'v','pcr':'s','ridge':'P','pls':'X','std':'^'}
ms = {'asa':8,'minnorm':5,'diag':5,'pcr':5,'ridge':5,'pls':5,'std':5}
marker_style=dict(markerfacecolor='none')

parser = argparse.ArgumentParser()
parser.add_argument("-vt","--vt",type=int,default=24,\
    help="verification time (hours)")
parser.add_argument("-ne","--nens",type=int,default=8,\
    help="ensemble size")
parser.add_argument("-m","--metric",type=str,default="",\
    help="forecast metric type")
parser.add_argument("-e","--ens",action='store_true',\
    help="ensemble estimated A")
argsin = parser.parse_args()
vt = argsin.vt # hours
ioffset = vt // 6
nens = argsin.nens
metric = argsin.metric
lens = argsin.ens
nensbase = 8

figdir = Path(f"fig{metric}/res_hess")
if not figdir.exists(): figdir.mkdir()

# load results
ds_dict = {}
if lens:
    ds_asa = xr.open_dataset(datadir/f'res_hessens_asa{metric}_vt{vt}nens{nens}.nc')
else:
    ds_asa = xr.open_dataset(datadir/f'res_hess_asa{metric}_vt{vt}nens{nensbase}.nc')
print(ds_asa)
ds_dict['asa'] = ds_asa
for solver in enasas:
    if lens:
        ds = xr.open_dataset(datadir/f'res_hessens_{solver}{metric}_vt{vt}nens{nens}.nc')
    else:
        ds = xr.open_dataset(datadir/f'res_hess_{solver}{metric}_vt{vt}nens{nens}.nc')
    ds_dict[solver] = ds
ds = xr.open_dataset(datadir/f'std{metric}_vt{vt}nens{nens}.nc')
#ds_dict['std'] = ds
ics = ds.ic.values
## x0i
#ncols = 3
#nrows = 3
#figsize = [10,8]
figsize = [8,6]
figdx = plt.figure(figsize=figsize,constrained_layout=True)
#axsdx = dict()
#iplt = 4
ax = figdx.add_subplot(111)
for key in ds_dict.keys():
#    if key=='asa':
#        axsdx[key] = figdx.add_subplot(nrows,ncols,1)
#    else:
#        axsdx[key] = figdx.add_subplot(nrows,ncols,iplt)
#        iplt += 1
    ds = ds_dict[key]
    x = ds.x
    nx = x.size
    nxh = nx // 2
    hwidth = 1
    i0 = nxh - hwidth
    i1 = nxh + hwidth + 1
    x0mean = np.zeros(nx)
    x0std  = np.zeros(nx)
    for icycle, ic in enumerate(ics):
        if key=='std':
            x0tmp = np.roll(ds.dx0opt[icycle],nxh-ic)
        else:
            x0tmp = np.roll(-ds.x0[icycle],nxh-ic)
        x0mean = x0mean + x0tmp
        x0std  = x0std  + x0tmp**2
    x0mean /= ics.size
    x0std = x0std/ics.size - x0mean**2
    x0std = np.where(x0std>0.0,np.sqrt(x0std),0.0)
    ax.plot(x,x0mean,c=colors[key],marker=markers[key],label=key)
    #axsdx[key].plot(x,x0mean,c=colors[key],marker=markers[key])
    ##axsdx[key].fill_between(x,x0mean-x0std,x0mean+x0std,alpha=0.5,color=colors[key])
    #axsdx[key].fill_between(x[i0:i1],0,1,\
    #    color='gray',alpha=0.5,transform=axsdx[key].get_xaxis_transform(),zorder=0)
    #axsdx[key].set_title(key)
    #axsdx[key].grid()
ax.fill_between(x[i0:i1],0,1,\
    color='gray',alpha=0.5,transform=ax.get_xaxis_transform(),zorder=0)
ax.legend(ncol=2)
ax.grid()
if lens:
    figdx.suptitle(r'$\Delta \mathbf{x}_{0}^{*}$ with $\mathbf{A}_\mathrm{ens}$'+f' FT{vt} {nens} member')
    figdx.savefig(figdir/f'x0i_Aens_vt{vt}nens{nens}.png')
else:
    figdx.suptitle(r'$\Delta \mathbf{x}_{0}^{*}$'+f' FT{vt} {nens} member')
    figdx.savefig(figdir/f'x0i_vt{vt}nens{nens}.png')
#plt.show()
plt.close()
#exit()

figh, axh = plt.subplots(figsize=[8,6],constrained_layout=True)
for key in ds_dict.keys():
    ds = ds_dict[key]
    if key=='std':
        r = ds.res_nl.values
    else:
        r = ds.rescalcm.values
    rmean = r.mean()
    rstd = r.std()
    label=key#+'\n'+r'$\mu$='+f'{rmean:.2f}, '+r'$\sigma$='+f'{rstd:.2f}'
    #if metric=='':
    #    bins = np.linspace(-1.0,1.0,51)
    #else:
    bins = 50
    axh.hist(r,bins=bins,histtype='step',lw=2.0,density=True,color=colors[key],label=label)
    if key=='std': continue
    figdir1 = figdir/key
    if not figdir1.exists(): figdir1.mkdir()
    fig = plt.figure(figsize=[6,6.5],constrained_layout=True)
    gs = GridSpec(4,4,figure=fig)
    ax = fig.add_subplot(gs[1:4,1:4])
    ax_histx = fig.add_subplot(gs[0,1:4],sharex=ax)
    ax_histy = fig.add_subplot(gs[1:4,0],sharey=ax)
    marker_style.update(markerfacecolor=colors[key],markeredgecolor='k',markeredgewidth=0.5)
    ax.plot(ds.resestp,ds.rescalcp,lw=0.0,marker='$+$',c=colors[key],**marker_style)
    marker_style.update(markerfacecolor=colors[key],markeredgecolor='gray',markeredgewidth=0.3)
    ax.plot(ds.resestm,ds.rescalcm,lw=0.0,marker='$-$',c=colors[key],**marker_style)
    #if metric=='':
    #    ylim = 5.0
    #else:
    xmin1,xmax1 = np.percentile(ds.resestm,[1.,99.])
    xmin2,xmax2 = np.percentile(ds.resestp,[1.,99.])
    xmin = min(xmin1,xmin2)
    xmax = max(xmax1,xmax2)
    #xmin, xmax = ax.get_xlim()
    ymin1,ymax1 = np.percentile(ds.rescalcm,[1.,99.])
    ymin2,ymax2 = np.percentile(ds.rescalcp,[1.,99.])
    ymin = min(ymin1,ymin2)
    ymax = max(ymax1,ymax2)
    #ymin, ymax = ax.get_ylim()
    ymax = max(ymax,xmax)
    ymin = min(ymin,xmin)
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(ymin,ymax)
    line = np.linspace(ymin,ymax,100)
    ax.plot(line,line,color='k',zorder=0)
    ax.grid()
    #ax.set_aspect(1.0)
    # histgram
    ax_histx.grid()
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    xmin, xmax = ax.get_xlim()
    bins = np.linspace(xmin,xmax,51)
    ax_histx.hist(ds.resestp,bins=bins,density=True,color=colors[key])
    ax_histx.hist(ds.resestm,bins=bins,density=True,color=colors[key])
    ax_histy.grid()
    ax_histy.yaxis.set_tick_params(labelleft=False)
    xmin, xmax = ax.get_ylim()
    bins = np.linspace(xmin,xmax,51)
    ax_histy.hist(ds.rescalcp,bins=bins,density=True,color=colors[key],orientation='horizontal')
    ax_histy.hist(ds.rescalcm,bins=bins,density=True,color=colors[key],orientation='horizontal')
    ax_histx.set_title('predicted',fontsize=14)
    ax_histy.set_ylabel('measured')
    if lens:
        fig.suptitle(r'$\Delta J(\Delta x_{0}^{*})$ with $\mathbf{A}_\mathrm{ens}$'+f' {key} vt={vt}h, Nens={nens}')
        fig.savefig(figdir1/f'res_hessens_calc_vs_est_{key}_vt{vt}ne{nens}.png')
    else:
        if key=='asa':
            fig.suptitle(r'$\Delta J(\Delta x_{0}^{*})$'+f' {key} vt={vt}h')
            fig.savefig(figdir1/f'res_calc_vs_est_{key}_vt{vt}ne{nensbase}.png')
        else:
            fig.suptitle(r'$\Delta J(\Delta x_{0}^{*})$'+f' {key} vt={vt}h, Nens={nens}')
            fig.savefig(figdir1/f'res_calc_vs_est_{key}_vt{vt}ne{nens}.png')
    #plt.show()
    plt.close(fig=fig)
axh.set_xlabel(r'nonlinear forecast response $\Delta J(\Delta x_{0}^{*})$')
axh.legend(loc='upper left',bbox_to_anchor=(1.01,1.0))
if lens:
    figh.suptitle(f'FT{vt}, {nens} member with '+r'$\mathbf{A}_\mathrm{ens}$')
    figh.savefig(figdir/f'hist_hessens_rescalcm_vt{vt}ne{nens}.png',dpi=300)
else:
    figh.suptitle(f'FT{vt}, {nens} member')
    figh.savefig(figdir/f'hist_rescalcm_vt{vt}ne{nens}.png',dpi=300)
#plt.show()
plt.close()