import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 14
import xarray as xr
from sklearn.linear_model import LinearRegression
from pathlib import Path
import argparse

datadir = Path('/Volumes/FF520/pyesa/adata/l96')

cmap = plt.get_cmap('tab10')
enasas = ['minnorm','diag','ridge','pcr','pls']#,'std']
colors = {'asa':cmap(0),'minnorm':cmap(1),'diag':cmap(2),'pcr':cmap(3),'ridge':cmap(4),'pls':cmap(5),'std':cmap(6)}
markers = {'asa':'*','minnorm':'o','diag':'v','pcr':'s','ridge':'P','pls':'X','std':'^'}
marker_style=dict(markerfacecolor='none')
nelist = [8,16,24,32] #,40]
vtlist = [24,48,72,96]

parser = argparse.ArgumentParser()
parser.add_argument("-vt","--vt",type=int,default=None,\
    help="verification time (hours)")
parser.add_argument("-ne","--nens",type=int,default=None,\
    help="ensemble size")
parser.add_argument("-m","--metric",type=str,default="",\
    help="forecast metric type")
parser.add_argument("-e","--ens",action='store_true',\
    help="ensemble estimated A")
parser.add_argument("-l","--loc",type=int,default=None,\
    help="localization radius for ensemble A")
argsin = parser.parse_args()
vt = argsin.vt
nens = argsin.nens
metric = argsin.metric
lens = argsin.ens
if lens:
    rloc = argsin.loc
else:
    rloc = None
if vt is None and nens is None:
    print("must be specified either vt or nens")
    exit()
elif vt is not None:
    vtype = f'vt{vt}'
    vlist = nelist
else:
    vtype = f'ne{nens}'
    vlist = vtlist

figdir = Path(f"fig{metric}/res_hess")
if not figdir.exists(): figdir.mkdir()

# load results
nensbase = 8
if metric=='':
    success = dict()
    failure = dict()
    success['asa'] = []
    failure['asa'] = []
    rsuccess = -0.75
    rfailure = 0.0
    for key in enasas:
        success[key] = []
        failure[key] = []
    if lens:
        if rloc is not None:
            f = open(figdir/f'nsuccess_failure_hessens_loc{rloc}_{vtype}.txt',mode='w')
        else:
            f = open(figdir/f'nsuccess_failure_hessens_{vtype}.txt',mode='w')
    else:
        f = open(figdir/f'nsuccess_failure_{vtype}.txt',mode='w')
ds_all = {}
Jb_dict = {}
Jb_base = {}
for v in vlist:
    ds_dict = {}
    if vtype[:2]=='vt':
        nens = v
        dstmp = xr.open_dataset(datadir/f'Jb_vt{vt}nens{nens}.nc')
        Jb = dstmp.Jb.values
        Jb_dict[nens] = Jb
        #if lens and (rloc is None):
        #    ds_asa = xr.open_dataset(datadir/f'res_hessens_asa{metric}_vt{vt}nens{nens}.nc')
        #    ds_dict['asa'] = ds_asa
        #else:
        if nens==nensbase:
                ds_asa = xr.open_dataset(datadir/f'res_hess_asa{metric}_vt{vt}nens{nensbase}.nc')
                ds_dict['asa'] = ds_asa
        print(ds_asa)
        if metric=='':
            nsuccess = np.sum(ds_asa.rescalcm.values/Jb<=rsuccess)
            nfailure = np.sum(ds_asa.rescalcm.values/Jb>rfailure)
            if lens:
                f.write(f"asa ne{nens}: #succcess={nsuccess} #failure={nfailure}\n")
            else:
                if nens==nensbase:
                    f.write(f"asa: #succcess={nsuccess} #failure={nfailure}\n")
            success['asa'].append(nsuccess / ds_asa.rescalcm.size)
            failure['asa'].append(nfailure / ds_asa.rescalcm.size)
        for solver in enasas:
            if lens:
                if rloc is not None:
                    ds = xr.open_dataset(datadir/f'res_hessens_loc{rloc}_{solver}{metric}_vt{vt}nens{nens}.nc')
                else:
                    ds = xr.open_dataset(datadir/f'res_hessens_{solver}{metric}_vt{vt}nens{nens}.nc')
            else:
                ds = xr.open_dataset(datadir/f'res_hess_{solver}{metric}_vt{vt}nens{nens}.nc')
            ds_dict[solver] = ds
            if metric=='':
                nsuccess = np.sum(ds.rescalcm.values/Jb<=rsuccess)
                nfailure = np.sum(ds.rescalcm.values/Jb>rfailure)
                f.write(f"{solver} ne{v}: #success={nsuccess} #failure={nfailure}\n")
                success[solver].append(nsuccess/ds.rescalcm.size)
                failure[solver].append(nfailure/ds.rescalcm.size)
        #ds = xr.open_dataset(datadir/f'std{metric}_vt{vt}nens{nens}.nc')
        #ds_dict['std'] = ds
        ds_all[nens] = ds_dict
    else:
        vt = v
        dstmp = xr.open_dataset(datadir/f'Jb_vt{vt}nens{nens}.nc')
        Jb = dstmp.Jb.values
        Jb_dict[vt] = Jb
        dstmp = xr.open_dataset(datadir/f'Jb_vt{vt}nens{nensbase}.nc')
        Jb = dstmp.Jb.values
        Jb_base[vt] = Jb
        #if lens and (rloc is None):
        #    ds_asa = xr.open_dataset(datadir/f'res_hessens_asa{metric}_vt{vt}nens{nensbase}.nc')
        #else:
        ds_asa = xr.open_dataset(datadir/f'res_hess_asa{metric}_vt{vt}nens{nensbase}.nc')
        ds_dict['asa'] = ds_asa
        if metric=='':
            nsuccess = np.sum(ds_asa.rescalcm.values/Jb_base[vt]<=rsuccess)
            nfailure = np.sum(ds_asa.rescalcm.values/Jb_base[vt]>rfailure)
            f.write(f"asa vt{v}: #succcess={nsuccess} #failure={nfailure}\n")
            success['asa'].append(nsuccess / ds_asa.rescalcm.size)
            failure['asa'].append(nfailure / ds_asa.rescalcm.size)
        for solver in enasas:
            if lens:
                if rloc is not None:
                    ds = xr.open_dataset(datadir/f'res_hessens_loc{rloc}_{solver}{metric}_vt{vt}nens{nens}.nc')
                else:
                    ds = xr.open_dataset(datadir/f'res_hessens_{solver}{metric}_vt{vt}nens{nens}.nc')
            else:
                ds = xr.open_dataset(datadir/f'res_hess_{solver}{metric}_vt{vt}nens{nens}.nc')
            ds_dict[solver] = ds
            if metric=='':
                nsuccess = np.sum(ds.rescalcm.values/Jb_dict[vt]<=rsuccess)
                nfailure = np.sum(ds.rescalcm.values/Jb_dict[vt]>rfailure)
                f.write(f"{solver} vt{v}: #success={nsuccess} #failure={nfailure}\n")
                success[solver].append(nsuccess/ds.rescalcm.size)
                failure[solver].append(nfailure/ds.rescalcm.size)
        #ds = xr.open_dataset(datadir/f'std{metric}_vt{vt}nens{nens}.nc')
        #ds_dict['std'] = ds
        ds_all[vt] = ds_dict
plt.rcParams['boxplot.medianprops.color'] = 'k'
plt.rcParams['boxplot.flierprops.marker'] = '.'
#if lens and rloc is None:
#    bwidth = 4.0 / (len(ds_dict.keys()))
#else:
bwidth = 4.0 / (len(ds_dict.keys())+1)
xoffset = 0.5 * bwidth * (len(ds_dict.keys()))
fig0, ax0 = plt.subplots(figsize=[10,6],constrained_layout=True)
figc, axc = plt.subplots(figsize=[10,6],constrained_layout=True)
patchs = []
if metric=='':
    ylims = [-1.1,1.1]
else:
    ylims = []
xticks = []
for j,v in enumerate(vlist):
    xticks.append(5*j+1)
    pos0 = xticks[j] - xoffset
    ds_dict = ds_all[v]
    if 'asa' in ds_dict.keys():
        xref = ds_dict['asa'].x0
        print(xref.shape)
        yi = xref - xref.mean(axis=1)
        sy = np.sqrt(np.sum(yi*yi,axis=1))
    for i,key in enumerate(ds_dict.keys()):
        marker_style['markerfacecolor']=colors[key]
        marker_style['markeredgecolor']='k'
        flierprops={'markeredgecolor':colors[key]}
        if key=='std':
            data1 = ds_dict[key].res_nl.values
        else:
            data1 = ds_dict[key].rescalcm.values
            if metric=='':
                if key=='asa' and vtype[:2]=='ne':
                    data1 = data1 / Jb_base[v]
                else:
                    data1 = data1 / Jb_dict[v]
        bplot1 = ax0.boxplot(data1,positions=[pos0],widths=bwidth,\
            patch_artist=True,flierprops=flierprops)#,whis=(0,100))
        for patch in bplot1['boxes']:
            patch.set_facecolor(colors[key])
            #patch.set_alpha(0.3)
        if len(ylims)>0:
            nlower = np.sum(data1<ylims[0])
            nupper = np.sum(data1>ylims[1])
        else:
            nlower = 0
            nupper = 0
        if nlower>0:
            ax0.text(pos0,0.05,f'{nlower}',transform=ax0.get_xaxis_transform(),\
                ha='center',size='x-small',weight='bold',color=colors[key],\
                bbox=dict(facecolor='yellow', pad=1, alpha=0.7))
        if nupper>0:
            ax0.text(pos0,0.95,f'{nupper}',transform=ax0.get_xaxis_transform(),\
                ha='center',size='x-small',weight='bold',color=colors[key],\
                bbox=dict(facecolor='yellow', pad=1, alpha=0.7))
        #ym = data1.mean()
        #p,=axs[0].plot([pos0],[ym],lw=0.0,marker=markers[key],ms=10,c=colors[key],**marker_style)
        if key != 'asa' and key != 'std':
            x = ds_dict[key].x0
            xi = x - x.mean(axis=1)
            sx = np.sqrt(np.sum(xi*xi,axis=1))
            sxy = np.sum(xi*yi,axis=1)
            c = sxy / sx / sy
            bplot2 = axc.boxplot(c,positions=[pos0],widths=bwidth,\
            patch_artist=True,flierprops=flierprops)#,whis=(0,100))
            for patch in bplot2['boxes']:
                patch.set_facecolor(colors[key])
        pos0 = pos0 + bwidth
        if j==0:
            if key == 'asa' and lens:
                patchs.append(Patch(color=colors[key],label=key+r' ($\mathbf{A}_\mathrm{est}$)'))
            else:
                patchs.append(Patch(color=colors[key],label=key))
ax0.set_xticks(xticks)
axc.set_xticks(xticks)
if vtype[:2]=='vt':
    ax0.set_xticklabels([f'mem{ne}' for ne in nelist])
    axc.set_xticklabels([f'mem{ne}' for ne in nelist])
else:
    ax0.set_xticklabels([f'FT{f}' for f in vtlist])
    axc.set_xticklabels([f'FT{f}' for f in vtlist])
ax0.grid(axis='y')
if len(ylims)>0:
    ax0.set_ylim(ylims)
axc.grid(axis='y')
axc.set_ylim(-1,1)
ax0.legend(handles=patchs,loc='upper left',bbox_to_anchor=(1.0,1.0))
axc.legend(handles=patchs[1:],loc='upper left',bbox_to_anchor=(1.0,1.0))
if metric=='':
    title=r'Nonlinear forecast response: $\frac{J(M(\mathbf{x}_0+\delta\mathbf{x}_0^*))-J(\mathbf{x}_T)}{J(\mathbf{x}_T)}$'
else:
    title=r'Nonlinear forecast response: $J(M(\mathbf{x}_0+\delta\mathbf{x}_0^*))-J(\mathbf{x}_T)$'
titlec=r'Spatial correlation of $\Delta \mathbf{x}_0^{*}$'
if vtype[:2]=='vt':
    title2=f', FT{vt}'
else:
    title2=f', {nens} member'
if lens:
    title3=r' with $\mathbf{A}_\mathrm{ens}$'
    figname='_hessens'
    if rloc is not None:
        title3=title3+r' $r_\mathrm{loc}$='+f'{rloc}'
        figname=figname+f'_loc{rloc}'
else:
    title3=''
    figname=''
ax0.set_title(title+title2+title3)
fig0.savefig(figdir/f"rescalcm{figname}_{vtype}.png",dpi=300)
axc.set_title(titlec+title2+title3)
figc.savefig(figdir/f"corr{figname}_{vtype}.png",dpi=300)
#plt.show()
plt.close()

if metric=='':
    fig, (ax1,ax2) = plt.subplots(nrows=2,sharex=True,figsize=[8,6],constrained_layout=True)
    for key in success.keys():
        if key == 'asa' and lens:
            label=key+r' ($\mathbf{A}_\mathrm{est}$)'
        else:
            label=key
        ax1.plot(vlist,np.array(success[key])*100,c=colors[key],marker=markers[key],label=label)
        ax2.plot(vlist,np.array(failure[key])*100,c=colors[key],marker=markers[key],label=label)
    ax1.legend(loc='upper left',bbox_to_anchor=(1.0,1.0))
    ax1.set_ylabel('success ratio (%)')
    ax2.set_ylabel('failure ratio (%)')
    ax1.set_ylim(20.0,80.0)
    ax2.set_ylim(0.0,20.0)
    for ax in [ax1,ax2]:
        ax.set_xticks(vlist)
        if vtype[:2]=='vt':
            ax.set_xticklabels([f'mem{ne}' if ne==vlist[0] else f'{ne}' for ne in vlist])
        else:
            ax.set_xticklabels([f'FT{f}' for f in vlist])
        ax.grid()
    fig.suptitle(title2[2:]+title3)
    fig.savefig(figdir/f'nsuccess_failure{figname}_{vtype}.png',dpi=300)
    #plt.show()
    plt.close()