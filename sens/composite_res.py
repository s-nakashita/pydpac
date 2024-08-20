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

resmul_dict = dict()
resuni_dict = dict()
for key in ds_dict.keys():
    if key=='calc':
        resmul_calc = ds_dict[key].resmul
        resuni_calc = ds_dict[key].resuni
    else:
        resmul_dict[key] = ds_dict[key].resmul
        resuni_dict[key] = ds_dict[key].resuni

#figall, axsall = plt.subplots(figsize=[8,6],ncols=2,nrows=2,constrained_layout=True)
#axsall[0,1].yaxis.set_tick_params(labelleft=False)
#axsall[1,1].yaxis.set_tick_params(labelleft=False)
figall = plt.figure(figsize=[8,6],constrained_layout=True)
axs00 = figall.add_subplot(221)
axs01 = figall.add_subplot(222,sharey=axs00)
axs10 = figall.add_subplot(223)
axs11 = figall.add_subplot(224,sharey=axs10)
axs01.yaxis.set_tick_params(labelleft=False)
axs11.yaxis.set_tick_params(labelleft=False)
axsall = np.array([[axs00,axs01],[axs10,axs11]])
figmulsize = [7,8]
figmul = plt.figure(figsize=figmulsize,constrained_layout=True)
gsmul = GridSpec(8,7,figure=figmul)
axsmul = dict()
axsmul['asa'] = figmul.add_subplot(gsmul[1:4,1:4])
axsmul['diag'] = figmul.add_subplot(gsmul[1:4,4:7],sharey=axsmul['asa'])
axsmul['minnorm'] = figmul.add_subplot(gsmul[5:8,1:4],sharex=axsmul['asa'])
axsmul['pls'] = figmul.add_subplot(gsmul[5:8,4:7],sharey=axsmul['minnorm'],sharex=axsmul['diag'])
axsmul_histx = dict()
axsmul_histx['asa'] = figmul.add_subplot(gsmul[0,1:4])
axsmul_histx['diag'] = figmul.add_subplot(gsmul[0,4:7])
axsmul_histx['minnorm'] = figmul.add_subplot(gsmul[4,1:4])
axsmul_histx['pls'] = figmul.add_subplot(gsmul[4,4:7])
axsmul_histy = dict()
axsmul_histy['asa'] = figmul.add_subplot(gsmul[1:4,0])
#axsmul_histy['diag'] = figmul.add_subplot(gsmul[1:4,4])
axsmul_histy['minnorm'] = figmul.add_subplot(gsmul[5:8,0])
#axsmul_histy['pls'] = figmul.add_subplot(gsmul[5:8,4])

for i,key in enumerate(resmul_dict.keys()):
    figdir1 = figdir/key
    if not figdir1.exists(): figdir1.mkdir()
    #fig, axs = plt.subplots(figsize=[6,6],ncols=2,nrows=2,constrained_layout=True)
    fig = plt.figure(figsize=[8,8],constrained_layout=True)
    gs = GridSpec(7,7,figure=fig)
    axs_00 = fig.add_subplot(gs[1:4,1:4])
    axs_01 = fig.add_subplot(gs[1:4,4:7],sharey=axs_00)
    axs_10 = fig.add_subplot(gs[4:7,1:4],sharex=axs_00)
    axs_11 = fig.add_subplot(gs[4:7,4:7],sharey=axs_10,sharex=axs_01)
    axs = np.array([
        [axs_00, axs_01],
        [axs_10, axs_11]
        ])
    axs[0,0].plot(resmul_dict[key],resmul_calc,lw=0.0,marker=markers[key],c=colors[key],**marker_style)
    _, aspectr, theta = confidence_ellipse(resmul_dict[key],resmul_calc,axs[0,0],\
        n_std=3,edgecolor='firebrick')
    handles=[Patch(facecolor='none',edgecolor='firebrick',\
        label=f'aspect={aspectr:.2f}, '+r'$\theta$='+f'{theta:.0f}')]
    axs[0,0].legend(handles=handles)
    if key in axsmul.keys():
        axsmul[key].plot(resmul_dict[key],resmul_calc,lw=0.0,marker=markers[key],c=colors[key],**marker_style)
        _,_,_ = confidence_ellipse(resmul_dict[key],resmul_calc,axsmul[key],\
        n_std=3,edgecolor='firebrick')
        axsmul[key].legend(handles=handles,loc='lower center',fontsize=12)
    axs[0,1].plot(resuni_dict[key],resmul_calc,lw=0.0,marker=markers[key],c=colors[key],**marker_style)
    _, aspectr, theta = confidence_ellipse(resuni_dict[key],resmul_calc,axs[0,1],\
        n_std=3,edgecolor='firebrick')
    axs[0,1].legend(handles=[Patch(facecolor='none',edgecolor='firebrick',\
        label=f'aspect={aspectr:.2f}, '+r'$\theta$='+f'{theta:.0f}')])
    axs[1,0].plot(resmul_dict[key],resuni_calc,lw=0.0,marker=markers[key],c=colors[key],**marker_style)
    _, aspectr, theta = confidence_ellipse(resmul_dict[key],resuni_calc,axs[1,0],\
        n_std=3,edgecolor='firebrick')
    axs[1,0].legend(handles=[Patch(facecolor='none',edgecolor='firebrick',\
        label=f'aspect={aspectr:.2f}, '+r'$\theta$='+f'{theta:.0f}')])
    axs[1,1].plot(resuni_dict[key],resuni_calc,lw=0.0,marker=markers[key],c=colors[key],**marker_style)
    _, aspectr, theta = confidence_ellipse(resuni_dict[key],resuni_calc,axs[1,1],\
        n_std=3,edgecolor='firebrick')
    axs[1,1].legend(handles=[Patch(facecolor='none',edgecolor='firebrick',\
        label=f'aspect={aspectr:.2f}, '+r'$\theta$='+f'{theta:.0f}')])
    #x = resmul_dict[key].values.reshape(-1,1)
    #y = resmul_calc.values
    #reg = LinearRegression().fit(x,y)
    #r2 = reg.score(x,y)
    #axs[0,0].plot(x,reg.predict(x),c=colors[key],label=f'y={reg.coef_[0]:.2f}x+{reg.intercept_:.2f}\nr^2={r2:.2e}')
    #x = resuni_dict[key].values.reshape(-1,1)
    #y = resuni_calc.values
    #reg = LinearRegression().fit(x,y)
    #r2 = reg.score(x,y)
    #axs[1,1].plot(x,reg.predict(x),c=colors[key],label=f'y={reg.coef_[0]:.2f}x+{reg.intercept_:.2f}\nr^2={r2:.2e}')
    #axs[0,0].legend()
    #axs[1,1].legend()
    
    axsall[0,0].plot(resmul_dict[key],resmul_calc,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
    axsall[0,1].plot(resuni_dict[key],resmul_calc,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
    axsall[1,0].plot(resmul_dict[key],resuni_calc,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
    axsall[1,1].plot(resuni_dict[key],resuni_calc,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
    for j,ax in enumerate(axs.flatten()):
        if j<2:
            ylims = np.percentile(resmul_calc,[.5,99.5])
        else:
            ylims = np.percentile(resuni_calc,[.5,99.5])
        #ymin, ymax = ax.get_ylim()
        ymin, ymax = ylims
        ylim = max(-ymin,ymax)
        if j==0 or j==3:
            ax.set_xlim(-ylim,ylim)
            #ax.set_aspect(1.0)
        else:
            if j==1:
                xmin, xmax = axs[1,1].get_xlim()
            else:
                xmin, xmax = axs[0,0].get_xlim()
            xlim = max(-xmin,xmax)
            ax.set_xlim(-xlim,xlim)
        ax.set_ylim(-ylim,ylim)
        line = np.linspace(-ylim,ylim,100)
        ax.plot(line,line,color='k',zorder=0)
        #ax.legend()
        #ax.set_title(key)
        ax.grid()
        #ax.set_aspect(1.0)
    if key in axsmul.keys():
        ax = axsmul[key]
        ymin, ymax = np.percentile(resmul_calc,[.5,99.5])
        ylim = max(-ymin,ymax)
        ax.set_ylim(-ylim,ylim)
        ax.set_xlim(-ylim,ylim)
        line = np.linspace(-ylim,ylim,100)
        ax.plot(line,line,color='k',zorder=0)
        #ax.legend()
        #ax.set_title(key)
        ax.grid()
        ax.set_aspect(1.0)
    # histgram
    ax00_histx = fig.add_subplot(gs[0,1:4],sharex=axs[0,0])
    ax00_histx.grid()
    ax00_histx.xaxis.set_tick_params(labelbottom=False)
    xmin, xmax = axs[0,0].get_xlim()
    bins = np.linspace(xmin,xmax,51)
    ax00_histx.hist(resmul_dict[key],bins=bins,density=True,color=colors[key])
    if key in axsmul_histx.keys():
        axsmul_histx[key].sharex(axsmul[key])
        axsmul_histx[key].grid()
        axsmul_histx[key].xaxis.set_tick_params(labelbottom=False)
        xmin, xmax = axsmul[key].get_xlim()
        bins = np.linspace(xmin,xmax,51)
        axsmul_histx[key].hist(resmul_dict[key],bins=bins,density=True,color=colors[key])
    ax00_histy = fig.add_subplot(gs[1:4,0],sharey=axs[0,0])
    ax00_histy.grid()
    ax00_histy.yaxis.set_tick_params(labelleft=False)
    xmin, xmax = axs[0,0].get_ylim()
    bins = np.linspace(xmin,xmax,51)
    ax00_histy.hist(resmul_calc,bins=bins,density=True,color='k',orientation='horizontal')
    if key in axsmul_histy.keys():
        axsmul_histy[key].sharey(axsmul[key])
        axsmul_histy[key].grid()
        axsmul_histy[key].yaxis.set_tick_params(labelleft=False)
        xmin, xmax = axsmul[key].get_ylim()
        bins = np.linspace(xmin,xmax,51)
        axsmul_histy[key].hist(resmul_calc,bins=bins,density=True,color='k',orientation='horizontal')
    ax01_histx = fig.add_subplot(gs[0,4:7],sharex=axs[0,1])
    ax01_histx.grid()
    ax01_histx.xaxis.set_tick_params(labelbottom=False)
    xmin, xmax = axs[0,1].get_xlim()
    bins = np.linspace(xmin,xmax,51)
    ax01_histx.hist(resuni_dict[key],bins=bins,density=True,color=colors[key])
    ax10_histy = fig.add_subplot(gs[4:7,0],sharey=axs[1,0]) #,axes_class=floating_axes.FloatingAxes,grid_helper=grid_helper)
    ax10_histy.grid()
    ax10_histy.yaxis.set_tick_params(labelleft=False)
    xmin, xmax = axs[1,0].get_ylim()
    bins = np.linspace(xmin,xmax,51)
    ax10_histy.hist(resuni_calc,bins=bins,density=True,orientation='horizontal',color='k')
    #axs[0,1].legend(loc='upper left',bbox_to_anchor=(1.01,1.0))
    #axs[0,0].set_xlabel('predicted (multivariate)')
    #axs[0,0].set_ylabel('measured (multivariate)')
    #axs[0,1].set_xlabel('predicted (univariate)')
    #axs[0,1].set_ylabel('measured (multivariate)')
    #axs[1,0].set_xlabel('predicted (multivariate)')
    #axs[1,0].set_ylabel('measured (univariate)')
    #axs[1,1].set_xlabel('predicted (univariate)')
    #axs[1,1].set_ylabel('measured (univariate)')
    ax00_histx.set_title('predicted (multivariate)',fontsize=14)
    ax00_histy.set_ylabel('measured (multivariate)')
    ax01_histx.set_title('predicted (univariate)',fontsize=14)
    ax10_histy.set_ylabel('measured (univariate)')
    fig.suptitle(r'$\Delta J(\Delta x_{0i})$'+f' {key} vt={vt}h, Nens={nens} #{nsample}')
    fig.savefig(figdir1/f'resunimul_{key}_vt{vt}ne{nens}.png')
    #plt.show()
    plt.close(fig=fig)
for j,ax in enumerate(axsall.flatten()):
    if j<2:
        ylims = np.percentile(resmul_calc,[.5,99.5])
    else:
        ylims = np.percentile(resuni_calc,[.5,99.5])
    #ymin, ymax = ax.get_ylim()
    ymin, ymax = ylims
    ylim = max(-ymin,ymax)
    ax.set_ylim(-ylim,ylim)
    ax.set_xlim(-ylim,ylim)
    line = np.linspace(-ylim,ylim,100)
    ax.plot(line,line,color='k',zorder=0)
    #ax.legend()
    #ax.set_title(key)
    ax.grid()
    ax.set_aspect(1.0)
axsall[0,1].legend(loc='upper left',bbox_to_anchor=(1.01,1.0))
#axsall[0,0].set_xlabel('predicted (multivariate)')
axsall[0,0].set_ylabel('measured (multivariate)')
#axsall[0,1].set_xlabel('predicted (univariate)')
#axsall[0,1].set_ylabel('measured (multivariate)')
axsall[1,0].set_xlabel('predicted (multivariate)')
axsall[1,0].set_ylabel('measured (univariate)')
axsall[1,1].set_xlabel('predicted (univariate)')
#axsall[1,1].set_ylabel('measured (univariate)')
figall.suptitle(r'$\Delta J(\Delta x_{0i})$'+f' vt={vt}h, Nens={nens} #{nsample}')
figall.savefig(figdir/f'resunimul_vt{vt}ne{nens}.png')

for key in axsmul_histx.keys():
    axsmul_histx[key].set_title(f'{key}',fontsize=14)
axsmul_histy['asa'].set_ylabel('measured')
axsmul_histy['minnorm'].set_ylabel('measured')
figmul.suptitle(r'$\Delta J(\Delta x_{0i})$'+f' vt={vt}h, Nens={nens} #{nsample}')
figmul.savefig(figdir/f'resmul_vt{vt}ne{nens}.png')
#plt.show()
plt.close()
#exit()
#fig, axs = plt.subplots(figsize=[8,6],ncols=2,nrows=2,constrained_layout=True)
#axs[0,1].yaxis.set_tick_params(labelleft=False)
#axs[1,1].yaxis.set_tick_params(labelleft=False)
fig = plt.figure(figsize=[8,6],constrained_layout=True)
axs00 = fig.add_subplot(221)
axs01 = fig.add_subplot(222,sharey=axs00)
axs10 = fig.add_subplot(223)
axs11 = fig.add_subplot(224,sharey=axs10)
axs01.yaxis.set_tick_params(labelleft=False)
axs11.yaxis.set_tick_params(labelleft=False)
axs = np.array([[axs00,axs01],[axs10,axs11]])
resmul_asa = resmul_dict['asa']
resuni_asa = resuni_dict['asa']
for i,key in enumerate(resmul_dict.keys()):
    if key=='asa': continue
    axs[0,0].plot(resmul_dict[key],resmul_asa,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
    axs[0,1].plot(resuni_dict[key],resmul_asa,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
    axs[1,0].plot(resmul_dict[key],resuni_asa,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
    axs[1,1].plot(resuni_dict[key],resuni_asa,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
for j,ax in enumerate(axs.flatten()):
    if j<2:
        ylims = np.percentile(resmul_asa,[.5,99.5])
    else:
        ylims = np.percentile(resuni_asa,[.5,99.5])
    #ymin, ymax = ax.get_ylim()
    ymin, ymax = ylims
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
axs[0,0].set_ylabel('ASA (multivariate)')
#axs[0,0].set_xlabel('EnASA (multivatiate)')
#axs[0,1].set_ylabel('ASA (multivariate)')
#axs[0,1].set_xlabel('EnASA (univariate)')
axs[1,0].set_ylabel('ASA (univariate)')
axs[1,0].set_xlabel('EnASA (multivariate)')
#axs[1,1].set_ylabel('ASA (univariate)')
axs[1,1].set_xlabel('EnASA (univariate)')
fig.suptitle(r'$\Delta J(\Delta x_{0i})$'+f' vt={vt}h, Nens={nens} #{nsample}')
fig.savefig(figdir/f'resunimul_vsasa_vt{vt}ne{nens}.png')
#plt.show()
plt.close()
#exit()

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
    #fig, axs = plt.subplots(figsize=[8,6],ncols=2,nrows=2,constrained_layout=True)
    fig = plt.figure(figsize=[8,6],constrained_layout=True)
    axs00 = fig.add_subplot(221)
    axs01 = fig.add_subplot(222,sharey=axs00)
    axs10 = fig.add_subplot(223)
    axs11 = fig.add_subplot(224,sharey=axs10)
    axs01.yaxis.set_tick_params(labelleft=False)
    axs11.yaxis.set_tick_params(labelleft=False)
    axs = np.array([[axs00,axs01],[axs10,axs11]])
    resmul_calc = resmul_dict['calc']
    resuni_calc = resuni_dict['calc']
    print(f"resmul_calc={resmul_calc.size}")
    print(f"resuni_calc={resuni_calc.size}")
    figmul = plt.figure(figsize=figmulsize,constrained_layout=True)
    gsmul = GridSpec(8,7,figure=figmul)
    axsmul = dict()
    axsmul['asa'] = figmul.add_subplot(gsmul[1:4,1:4])
    axsmul['diag'] = figmul.add_subplot(gsmul[1:4,4:7],sharey=axsmul['asa'])
    axsmul['minnorm'] = figmul.add_subplot(gsmul[5:8,1:4],sharex=axsmul['asa'])
    axsmul['pls'] = figmul.add_subplot(gsmul[5:8,4:7],sharey=axsmul['minnorm'],sharex=axsmul['diag'])
    axsmul_histx = dict()
    axsmul_histx['asa'] = figmul.add_subplot(gsmul[0,1:4])
    axsmul_histx['diag'] = figmul.add_subplot(gsmul[0,4:7])
    axsmul_histx['minnorm'] = figmul.add_subplot(gsmul[4,1:4])
    axsmul_histx['pls'] = figmul.add_subplot(gsmul[4,4:7])
    axsmul_histy = dict()
    axsmul_histy['asa'] = figmul.add_subplot(gsmul[1:4,0])
    #axsmul_histy['diag'] = figmul.add_subplot(gsmul[1:4,4])
    axsmul_histy['minnorm'] = figmul.add_subplot(gsmul[5:8,0])
    #axsmul_histy['pls'] = figmul.add_subplot(gsmul[5:8,4])
    for key in resmul_dict.keys():
        if key=='calc': continue
        figdir1 = figdir/key
        axs[0,0].plot(resmul_dict[key],resmul_calc,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
        axs[0,1].plot(resuni_dict[key],resmul_calc,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
        axs[1,0].plot(resmul_dict[key],resuni_calc,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
        axs[1,1].plot(resuni_dict[key],resuni_calc,lw=0.0,marker=markers[key],c=colors[key],label=key,**marker_style)
        #fig1, axs1 = plt.subplots(figsize=[6,6],ncols=2,nrows=2,constrained_layout=True)
        fig1 = plt.figure(figsize=[8,8],constrained_layout=True)
        gs = GridSpec(7,7,figure=fig1)
        axs1_00 = fig1.add_subplot(gs[1:4,1:4])
        axs1_01 = fig1.add_subplot(gs[1:4,4:7],sharey=axs1_00)
        axs1_10 = fig1.add_subplot(gs[4:7,1:4],sharex=axs1_00)
        axs1_11 = fig1.add_subplot(gs[4:7,4:7],sharey=axs1_10,sharex=axs1_01)
        axs1 = np.array([
            [axs1_00, axs1_01],
            [axs1_10, axs1_11]
            ])
        axs1[0,0].plot(resmul_dict[key],resmul_calc,lw=0.0,marker=markers[key],c=colors[key],**marker_style)
        _, aspectr, theta = confidence_ellipse(resmul_dict[key],resmul_calc,axs1[0,0],\
            n_std=3,edgecolor='firebrick')
        handles = [Patch(facecolor='none',edgecolor='firebrick',\
            label=f'aspect={aspectr:.2f}, '+r'$\theta$='+f'{theta:.0f}')]
        axs1[0,0].legend(handles=handles)
        if key in axsmul.keys():
            axsmul[key].plot(resmul_dict[key],resmul_calc,lw=0.0,marker=markers[key],c=colors[key],**marker_style)
            _,_,_ = confidence_ellipse(resmul_dict[key],resmul_calc,axsmul[key],\
            n_std=3,edgecolor='firebrick')
            axsmul[key].legend(handles=handles,loc='lower center',fontsize=12)
        axs1[0,1].plot(resuni_dict[key],resmul_calc,lw=0.0,marker=markers[key],c=colors[key],**marker_style)
        _, aspectr, theta = confidence_ellipse(resuni_dict[key],resmul_calc,axs1[0,1],\
            n_std=3,edgecolor='firebrick')
        axs1[0,1].legend(handles=[Patch(facecolor='none',edgecolor='firebrick',\
            label=f'aspect={aspectr:.2f}, '+r'$\theta$='+f'{theta:.0f}')])
        axs1[1,0].plot(resmul_dict[key],resuni_calc,lw=0.0,marker=markers[key],c=colors[key],**marker_style)
        _, aspectr, theta = confidence_ellipse(resmul_dict[key],resuni_calc,axs1[1,0],\
            n_std=3,edgecolor='firebrick')
        axs1[1,0].legend(handles=[Patch(facecolor='none',edgecolor='firebrick',\
            label=f'aspect={aspectr:.2f}, '+r'$\theta$='+f'{theta:.0f}')])
        axs1[1,1].plot(resuni_dict[key],resuni_calc,lw=0.0,marker=markers[key],c=colors[key],**marker_style)
        _, aspectr, theta = confidence_ellipse(resuni_dict[key],resuni_calc,axs1[1,1],\
            n_std=3,edgecolor='firebrick')
        axs1[1,1].legend(handles=[Patch(facecolor='none',edgecolor='firebrick',\
            label=f'aspect={aspectr:.2f}, '+r'$\theta$='+f'{theta:.0f}')])
        #x = resmul_dict[key].values.reshape(-1,1)
        #y = resmul_calc.values
        #reg = LinearRegression().fit(x,y)
        #r2 = reg.score(x,y)
        #axs1[0,0].plot(x,reg.predict(x),c=colors[key],label=f'y={reg.coef_[0]:.2f}x+{reg.intercept_:.2f}\nr^2={r2:.2e}')
        #x = resuni_dict[key].values.reshape(-1,1)
        #y = resuni_calc.values
        #reg = LinearRegression().fit(x,y)
        #r2 = reg.score(x,y)
        #axs1[1,1].plot(x,reg.predict(x),c=colors[key],label=f'y={reg.coef_[0]:.2f}x+{reg.intercept_:.2f}\nr^2={r2:.2e}')
        #axs1[0,0].legend()
        #axs1[1,1].legend()
        for j,ax in enumerate(axs1.flatten()):
            if j<2:
                ylims = np.percentile(resmul_calc,[.5,99.5])
            else:
                ylims = np.percentile(resuni_calc,[.5,99.5])
            ymin, ymax = ylims
            #ymin, ymax = ax.get_ylim()
            ylim = max(-ymin,ymax)
            ax.set_ylim(-ylim,ylim)
            if j==0 or j==3:
                ax.set_xlim(-ylim,ylim)
                #ax.set_aspect(1.0)
            else:
                if j==1:
                    xmin, xmax = axs1[1,1].get_xlim()
                else:
                    xmin, xmax = axs1[0,0].get_xlim()
                xlim = max(-xmin,xmax)
                ax.set_xlim(-xlim,xlim)
            line = np.linspace(-ylim,ylim,100)
            ax.plot(line,line,color='k',zorder=0)
            #ax.legend()
            #ax.set_title(key)
            ax.grid()
        if key in axsmul.keys():
            ax = axsmul[key]
            ymin, ymax = np.percentile(resmul_calc,[.5,99.5])
            ylim = max(-ymin,ymax)
            ax.set_ylim(-ylim,ylim)
            ax.set_xlim(-ylim,ylim)
            line = np.linspace(-ylim,ylim,100)
            ax.plot(line,line,color='k',zorder=0)
            #ax.legend()
            #ax.set_title(key)
            ax.grid()
            ax.set_aspect(1.0)
        # histgram
        ax00_histx = fig1.add_subplot(gs[0,1:4],sharex=axs1[0,0])
        ax00_histx.grid()
        #divider00 = make_axes_locatable(axs1[0,0])
        #ax00_histx = divider00.append_axes("top",.5,pad=0.1,sharex=axs1[0,0])
        ax00_histx.xaxis.set_tick_params(labelbottom=False)
        xmin, xmax = axs1[0,0].get_xlim()
        bins = np.linspace(xmin,xmax,51)
        ax00_histx.hist(resmul_dict[key],bins=bins,density=True,color=colors[key])
        #ax00_histx.set_xlim(axs1[0,0].get_xlim())
        #tr00 = Affine2D().scale().rotate_deg(90)
        #ymin00,ymax00 = axs1[0,0].get_ylim()
        #grid_helper = floating_axes.GridHelperCurveLinear(
        #    tr00,extremes=(ymin00,ymax00,0,1),
        #    grid_locator1=MaxNLocator(nbins=4),
        #    grid_locator2=MaxNLocator(nbins=2)
        #)
        if key in axsmul_histx.keys():
            axsmul_histx[key].sharex(axsmul[key])
            axsmul_histx[key].grid()
            axsmul_histx[key].xaxis.set_tick_params(labelbottom=False)
            xmin, xmax = axsmul[key].get_xlim()
            bins = np.linspace(xmin,xmax,51)
            axsmul_histx[key].hist(resmul_dict[key],bins=bins,density=True,color=colors[key])
        ax00_histy = fig1.add_subplot(gs[1:4,0],sharey=axs1[0,0])#,axes_class=floating_axes.FloatingAxes,grid_helper=grid_helper)
        ax00_histy.grid()
        ax00_histy.yaxis.set_tick_params(labelleft=False)
        xmin, xmax = axs1[0,0].get_ylim()
        bins = np.linspace(xmin,xmax,51)
        ax00_histy.hist(resmul_calc,bins=bins,density=True,color='k',orientation='horizontal')
        #ax00_histy.set_ylim(axs1[0,0].get_ylim())
        #divider01 = make_axes_locatable(axs1[0,1])
        #ax01_histx = divider01.append_axes("top",.5,pad=0.1,sharex=axs1[0,1])
        #ax01_histx.xaxis.set_tick_params(labelbottom=False)
        #ax01_histx.hist(resuni_dict[key],bins=50,density=True,color=colors[key])
        #ax01_histy = divider01.append_axes("right",.5,pad=0.1,sharey=axs1[0,1])
        #ax01_histy.yaxis.set_tick_params(labelleft=False)
        #ax01_histy.hist(resmul_calc,bins=50,density=True,orientation='horizontal',color=colors[key])
        #tr11 = Affine2D().rotate_deg(180)
        #xmin11,xmax11 = axs1[1,1].get_xlim()
        #grid_helper = floating_axes.GridHelperCurveLinear(
        #    tr11,extremes=(xmin11,xmax11,0,1),
        #    grid_locator1=MaxNLocator(nbins=4),
        #    grid_locator2=MaxNLocator(nbins=2)
        #)
        if key in axsmul_histy.keys():
            axsmul_histy[key].sharey(axsmul[key])
            axsmul_histy[key].grid()
            axsmul_histy[key].yaxis.set_tick_params(labelleft=False)
            xmin, xmax = axsmul[key].get_ylim()
            bins = np.linspace(xmin,xmax,51)
            axsmul_histy[key].hist(resmul_calc,bins=bins,density=True,color='k',orientation='horizontal')
        ax01_histx = fig1.add_subplot(gs[0,4:7],sharex=axs1[0,1])#,axes_class=floating_axes.FloatingAxes,grid_helper=grid_helper)
        ax01_histx.grid()
        ax01_histx.xaxis.set_tick_params(labelbottom=False)
        #aux_ax11_histx = ax11_histx.get_aux_axes(tr11)
        xmin, xmax = axs1[0,1].get_xlim()
        bins = np.linspace(xmin,xmax,51)
        ax01_histx.hist(resuni_dict[key],bins=bins,density=True,color=colors[key])
        #ax01_histx.set_xlim(axs1[0,1].get_xlim())
        #divider11 = make_axes_locatable(axs1[1,1])
        #ax11_histy = divider11.append_axes("right",.5,pad=0.1,sharey=axs1[1,1])
        #tr11 = Affine2D().rotate_deg(270)
        #ymin11,ymax11 = axs1[1,1].get_ylim()
        #grid_helper = floating_axes.GridHelperCurveLinear(
        #    tr11,extremes=(ymin11,ymax11,0,1),
        #    grid_locator1=MaxNLocator(nbins=4),
        #    grid_locator2=MaxNLocator(nbins=2)
        #)
        ax10_histy = fig1.add_subplot(gs[4:7,0],sharey=axs1[1,0]) #,axes_class=floating_axes.FloatingAxes,grid_helper=grid_helper)
        ax10_histy.grid()
        ax10_histy.yaxis.set_tick_params(labelleft=False)
        #aux_ax11_histy = ax11_histy.get_aux_axes(tr11)
        xmin, xmax = axs1[1,0].get_ylim()
        bins = np.linspace(xmin,xmax,51)
        ax10_histy.hist(resuni_calc,bins=bins,density=True,orientation='horizontal',color='k')
        #ax10_histy.set_ylim(axs1[1,0].get_ylim())
        #axs1[0,0].set_xlabel('predicted (multivariate)')
        ax00_histx.set_title('predicted (multivariate)')
        #axs1[0,0].set_ylabel('measured (multivariate)')
        ax00_histy.set_ylabel('measured (multivariate)')
        #axs1[0,1].set_xlabel('predicted (univariate)')
        ax01_histx.set_title('predicted (univariate)')
        #axs1[0,1].set_ylabel('measured (multivariate)')
        #axs1[1,0].set_xlabel('predicted (multivariate)')
        #axs1[1,0].set_ylabel('measured (univariate)')
        ax10_histy.set_ylabel('measured (univariate)')
        #axs1[1,1].set_xlabel('predicted (univariate)')
        #axs1[1,1].set_ylabel('measured (univariate)')
        nsample = resmul_dict[key].size
        fig1.suptitle(r'$\Delta J(\Delta x_{0i})$'+f' {figname} {key} vt={vt}h, Nens={nens} #{nsample}')
        fig1.savefig(figdir1/f'resunimul_{figname}_{key}_vt{vt}ne{nens}.png')
        #plt.show(block=False)
        plt.close(fig=fig1)
        #exit()
    for j,ax in enumerate(axs.flatten()):
        if j<2:
            ylims = np.percentile(resmul_calc,[.5,99.5])
        else:
            ylims = np.percentile(resuni_calc,[.5,99.5])
        ymin, ymax = ylims
        #ymin, ymax = ax.get_ylim()
        ylim = max(-ymin,ymax)
        ax.set_ylim(-ylim,ylim)
        #if j==0 or j==3:
        ax.set_xlim(-ylim,ylim)
        ax.set_aspect(1.0)
        #else:
        #    if j==1:
        #        xmin, xmax = axs[1,1].get_xlim()
        #    else:
        #        xmin, xmax = axs[0,0].get_xlim()
        #    xlim = max(-xmin,xmax)
        #    ax.set_xlim(-xlim,xlim)
        line = np.linspace(-ylim,ylim,100)
        ax.plot(line,line,color='k',zorder=0)
        #ax.legend()
        #ax.set_title(key)
        ax.grid()
    axs[0,1].legend(loc='upper left',bbox_to_anchor=(1.01,1.0))
    #axs[0,0].set_xlabel('predicted (multivariate)')
    axs[0,0].set_ylabel('measured (multivariate)')
    #axs[0,1].set_xlabel('predicted (univariate)')
    #axs[0,1].set_ylabel('measured (multivariate)')
    axs[1,0].set_xlabel('predicted (multivariate)')
    axs[1,0].set_ylabel('measured (univariate)')
    axs[1,1].set_xlabel('predicted (univariate)')
    #axs[1,1].set_ylabel('measured (univariate)')
    nsample = resmul_dict['calc'].size
    fig.suptitle(r'$\Delta J(\Delta x_{0i})$'+f' {figname} vt={vt}h, Nens={nens} #{nsample}')
    fig.savefig(figdir/f'resunimul_{figname}_vt{vt}ne{nens}.png')
    for key in axsmul_histx.keys():
        axsmul_histx[key].set_title(f'{key}',fontsize=14)
    axsmul_histy['asa'].set_ylabel('measured')
    axsmul_histy['minnorm'].set_ylabel('measured')
    figmul.suptitle(r'$\Delta J(\Delta x_{0i})$'+f' {figname} vt={vt}h, Nens={nens} #{nsample}')
    figmul.savefig(figdir/f'resmul_{figname}_vt{vt}ne{nens}.png')
    #plt.show()
    plt.close()
