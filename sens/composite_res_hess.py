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
import pandas as pd
from sklearn.linear_model import LinearRegression
from pathlib import Path
import argparse

datadir = Path('/Volumes/FF520/pyesa/adata/l96')

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
parser.add_argument("-l","--loc",type=int,default=None,\
    help="localization radius for ensemble A")
parser.add_argument("-x","--x0",action='store_true',help="plot x0 composite")
argsin = parser.parse_args()
vt = argsin.vt # hours
ioffset = vt // 6
nens = argsin.nens
metric = argsin.metric
lens = argsin.ens
rloc = argsin.loc
lx0 = argsin.x0
nensbase = 8

figdir = Path(f"fig{metric}/res_hess")
if not figdir.exists(): figdir.mkdir()

# load results
ds_dict = {}
if lens and rloc is None:
    ds_asa = xr.open_dataset(datadir/f'res_hessens_asa{metric}_vt{vt}nens{nens}.nc')
else:
    ds_asa = xr.open_dataset(datadir/f'res_hess_asa{metric}_vt{vt}nens{nensbase}.nc')
print(ds_asa)
ds_dict['asa'] = ds_asa
for solver in enasas:
    if lens:
        if rloc is not None:
            ds = xr.open_dataset(datadir/f'res_hessens_loc{rloc}_{solver}{metric}_vt{vt}nens{nens}.nc')
        else:
            ds = xr.open_dataset(datadir/f'res_hessens_{solver}{metric}_vt{vt}nens{nens}.nc')
    else:
        ds = xr.open_dataset(datadir/f'res_hess_{solver}{metric}_vt{vt}nens{nens}.nc')
    ds_dict[solver] = ds
ds = xr.open_dataset(datadir/f'std{metric}_vt{vt}nens{nens}.nc')
#ds_dict['std'] = ds
ics = ds.ic.values
## x0i
if lx0:
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
        if rloc is not None:
            figdx.suptitle(r'$\Delta \mathbf{x}_{0}^{*}$ with $\mathbf{A}_\mathrm{ens}$ $r_\mathrm{loc}$='+f'{rloc} FT{vt} {nens} member')
            figdx.savefig(figdir/f'x0i_Aens_loc{rloc}_vt{vt}nens{nens}.png')
        else:
            figdx.suptitle(r'$\Delta \mathbf{x}_{0}^{*}$ with $\mathbf{A}_\mathrm{ens}$'+f' FT{vt} {nens} member')
            figdx.savefig(figdir/f'x0i_Aens_vt{vt}nens{nens}.png')
    else:
        figdx.suptitle(r'$\Delta \mathbf{x}_{0}^{*}$'+f' FT{vt} {nens} member')
        figdx.savefig(figdir/f'x0i_vt{vt}nens{nens}.png')
    #plt.show()
    plt.close()
    #exit()

# histgrams
figh, axh = plt.subplots(figsize=[8,6],constrained_layout=True)
# mean + stdv
figm, axm = plt.subplots(figsize=[6,6],constrained_layout=True)
yminall = 999.9
ymaxall = -999.9
estp_mean_dict = dict(FT=vt,member=nens)
calcp_mean_dict = dict(FT=vt,member=nens)
estm_mean_dict = dict(FT=vt,member=nens)
calcm_mean_dict = dict(FT=vt,member=nens)
rmsd_p_dict = dict(FT=vt,member=nens)
rmsd_m_dict= dict(FT=vt,member=nens)
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
    rmsd_p = np.sqrt(np.mean((ds.resestp-ds.rescalcp)**2))
    ax.plot(ds.resestp,ds.rescalcp,lw=0.0,marker='$+$',c=colors[key],label=f'RMSD={rmsd_p:.2f}',**marker_style)
    rmsd_p_dict[key] = rmsd_p.values
    if key!='diag':
        xm_p = np.mean(ds.resestp)
        xs_p = np.std(ds.resestp)
        ym_p = np.mean(ds.rescalcp)
        ys_p = np.std(ds.rescalcp)
        estp_mean_dict[key] = xm_p.values
        calcp_mean_dict[key] = ym_p.values
        axm.plot(xm_p,ym_p,c=colors[key],marker='o',ms=10,lw=0.0,label=f'{key},$+$',**marker_style)
        #axm.errorbar(xm_p,ym_p,xerr=xs_p,yerr=ys_p,c=colors[key],marker='$p$',ms=10,label=f'{key},$+$',**marker_style)
        xm_m = np.mean(ds.resestm)
        xs_m = np.std(ds.resestm)
        ym_m = np.mean(ds.rescalcm)
        ys_m = np.std(ds.rescalcm)
        estm_mean_dict[key] = xm_m.values
        calcm_mean_dict[key] = ym_m.values
        axm.plot(xm_m,ym_m,c=colors[key],marker='^',ms=10,lw=0.0,label=f'{key},$-$',**marker_style)
        #axm.errorbar(xm_m,ym_m,xerr=xs_m,yerr=ys_m,c=colors[key],marker='$m$',ms=10,label=f'{key},$-$',**marker_style)
        xmin = min(xm_p-xs_p,xm_m-xs_m)
        xmax = max(xm_p+xs_p,xm_m+xs_m)
        ymin = min(ym_p-ys_p,ym_m-ys_m)
        ymax = max(ym_p+ys_p,ym_m+ys_m)
        ymin = min(xmin,ymin)
        ymax = max(xmax,ymax)
        yminall = min(yminall,ymin)
        ymaxall = max(ymaxall,ymax)
    marker_style.update(markerfacecolor=colors[key],markeredgecolor='gray',markeredgewidth=0.3)
    rmsd_m = np.sqrt(np.mean((ds.resestm-ds.rescalcm)**2))
    ax.plot(ds.resestm,ds.rescalcm,lw=0.0,marker='$-$',c=colors[key],label=f'RMSD={rmsd_m:.2f}',**marker_style)
    rmsd_m_dict[key] = rmsd_m.values
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
    ylim = max(ymax,-1.0*ymin)
    ymax = ylim
    ymin = -1.0*ylim
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(ymin,ymax)
    line = np.linspace(ymin,ymax,100)
    ax.plot(line,line,color='k',zorder=0)
    ax.grid()
    ax.legend()
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
    ax_histx.set_title('predicted (linear)',fontsize=14)
    ax_histy.set_ylabel('measured (nonlinear)')
    if lens:
        if rloc is not None:
            fig.suptitle(r'$\Delta J(\Delta x_{0}^{*})$ with $\mathbf{A}_\mathrm{ens}$ $r_\mathrm{loc}=$'+f'{rloc} {key} vt={vt}h, Nens={nens}')
            fig.savefig(figdir1/f'res_hessens_loc{rloc}_calc_vs_est_{key}_vt{vt}ne{nens}.png')
        else:
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
    if rloc is not None:
        figh.suptitle(f'FT{vt}, {nens} member with '+r'$\mathbf{A}_\mathrm{ens}$ $r_\mathrm{loc}$='+f'{rloc}')
        figh.savefig(figdir/f'hist_hessens_loc{rloc}_rescalcm_vt{vt}ne{nens}.png',dpi=300)
    else:
        figh.suptitle(f'FT{vt}, {nens} member with '+r'$\mathbf{A}_\mathrm{ens}$')
        figh.savefig(figdir/f'hist_hessens_rescalcm_vt{vt}ne{nens}.png',dpi=300)
else:
    figh.suptitle(f'FT{vt}, {nens} member')
    figh.savefig(figdir/f'hist_rescalcm_vt{vt}ne{nens}.png',dpi=300)
#plt.show()
plt.close(fig=figh)

ylim = max(ymaxall,-1.0*yminall)
ymaxall = ylim
yminall = -1.0*ylim
axm.set_ylim(yminall,ymaxall)
axm.set_xlim(yminall,ymaxall)
line = np.linspace(yminall,ymaxall,100)
axm.plot(line,line,color='k',zorder=0)
axm.grid()
axm.legend(fontsize=12,ncol=2)
axm.set_ylabel('measured (nonlinear)')
axm.set_xlabel('predicted (linear)')
if lens:
    if rloc is not None:
        figm.suptitle(r'$\Delta J(\Delta x_{0}^{*})$ with $\mathbf{A}_\mathrm{ens}$ $r_\mathrm{loc}$='+f'{rloc} vt={vt}h, Nens={nens}')
        figm.savefig(figdir/f'res_hessens_loc{rloc}_calc_vs_est_vt{vt}ne{nens}.png')
    else:
        figm.suptitle(r'$\Delta J(\Delta x_{0}^{*})$ with $\mathbf{A}_\mathrm{ens}$'+f' vt={vt}h, Nens={nens}')
        figm.savefig(figdir/f'res_hessens_calc_vs_est_vt{vt}ne{nens}.png')
else:
    figm.suptitle(r'$\Delta J(\Delta x_{0}^{*})$'+f' vt={vt}h, Nens={nens}')
    figm.savefig(figdir/f'res_calc_vs_est_vt{vt}ne{nens}.png')
#plt.show()
plt.close(fig=figm)

# save csv data
if lens:
    if rloc is not None:
        csvname=f'res_hessens_loc{rloc}'
    else:
        csvname='res_hessens'
else:
    csvname='res_hess'
ds = pd.DataFrame(estp_mean_dict,columns=estp_mean_dict.keys(),index=[0])
ds.to_csv(datadir/f'{csvname}_estp_mean_vt{vt}ne{nens}.csv')
ds = pd.DataFrame(estm_mean_dict,columns=estm_mean_dict.keys(),index=[0])
ds.to_csv(datadir/f'{csvname}_estm_mean_vt{vt}ne{nens}.csv')
ds = pd.DataFrame(calcp_mean_dict,columns=calcp_mean_dict.keys(),index=[0])
ds.to_csv(datadir/f'{csvname}_calcp_mean_vt{vt}ne{nens}.csv')
ds = pd.DataFrame(calcm_mean_dict,columns=calcm_mean_dict.keys(),index=[0])
ds.to_csv(datadir/f'{csvname}_calcm_mean_vt{vt}ne{nens}.csv')
ds = pd.DataFrame(rmsd_p_dict,columns=rmsd_p_dict.keys(),index=[0])
ds.to_csv(datadir/f'{csvname}_rmsd_p_vt{vt}ne{nens}.csv')
ds = pd.DataFrame(rmsd_m_dict,columns=rmsd_m_dict.keys(),index=[0])
ds.to_csv(datadir/f'{csvname}_rmsd_m_vt{vt}ne{nens}.csv')
