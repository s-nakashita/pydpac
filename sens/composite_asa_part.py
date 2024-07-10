import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 14
import xarray as xr
from scipy.stats import ttest_1samp
from pathlib import Path
import sys

datadir = Path('data')
figdir = Path('fig')

cmap = plt.get_cmap('tab10')
enasas = ['minnorm','diag','pcr','ridge','pls']
colors = {'asa':cmap(0),'minnorm':cmap(1),'diag':cmap(2),'pcr':cmap(3),'ridge':cmap(4),'pls':cmap(5)}
markers = {'asa':'*','minnorm':'o','diag':'v','pcr':'s','ridge':'P','pls':'X'}
ms = {'asa':8,'minnorm':5,'diag':5,'pcr':5,'ridge':5,'pls':5}
dJlim = {24:0.2,48:1.0,72:2.0,96:3.6}
dxlim = {24:0.02,48:0.02,72:0.02,96:0.02}

vt = 24
if len(sys.argv)>1:
    vt = int(sys.argv[1])
nens = 8
if len(sys.argv)>2:
    nens = int(sys.argv[2])
# load results
ds_dict = {}
ds_asa = xr.open_dataset(datadir/f'asa_vt{vt}nens{nens}.nc')
print(ds_asa)
ds_dict['asa'] = ds_asa
for solver in enasas:
    ds = xr.open_dataset(datadir/f'{solver}_vt{vt}nens{nens}.nc')
    ds_dict[solver] = ds

# determine best and worst 10% cases based on nonlinear forecast responses to ASA
res_nl_ref = ds_asa.res_nl.values
thres_10 = np.percentile(res_nl_ref, 10)
thres_90 = np.percentile(res_nl_ref, 90)

ncols = 2
nrows = 3
figdJb, axsdJb = plt.subplots(figsize=[8,10],nrows=nrows,ncols=ncols,constrained_layout=True)
figdxb, axsdxb = plt.subplots(figsize=[8,10],nrows=nrows,ncols=ncols,constrained_layout=True)
figdJw, axsdJw = plt.subplots(figsize=[8,10],nrows=nrows,ncols=ncols,constrained_layout=True)
figdxw, axsdxw = plt.subplots(figsize=[8,10],nrows=nrows,ncols=ncols,constrained_layout=True)

for axdJb, axdxb, axdJw, axdxw, solver in \
    zip(axsdJb.flatten(),axsdxb.flatten(),\
        axsdJw.flatten(),axsdxw.flatten(),ds_dict.keys()):
    marker_style = dict(markerfacecolor=colors[solver],markeredgecolor='k',ms=ms[solver])
    ds = ds_dict[solver]
    x = ds.x
    nx = x.size
    nxh = nx // 2
    hwidth = 1
    i0 = nxh - hwidth
    i1 = nxh + hwidth + 1
    
    res_nl = ds.res_nl.values
    ics = ds.ic.values
    dJdx0 = ds.dJdx0.values
    dx0opt = ds.dx0opt.values
    print(dJdx0.shape)
    dJdx0best = np.zeros(dJdx0.shape[1])
    dx0optbest = np.zeros(dx0opt.shape[1])
    dJdx0worst = np.zeros(dJdx0.shape[1])
    dx0optworst = np.zeros(dx0opt.shape[1])
    #dJdx0std = np.zeros(dJdx0.shape[1])
    #dx0optstd = np.zeros(dx0opt.shape[1])
    nbest=0
    nworst=0
    resbest=0.0
    resworst=0.0
    for icycle, ic in enumerate(ics):
        dJtmp = np.roll(dJdx0[icycle],nxh-ic)
        dxtmp = np.roll(dx0opt[icycle],nxh-ic)
        res1 = res_nl_ref[icycle]
        if res1 <= thres_10:
            dJdx0best = dJdx0best + dJtmp
            dx0optbest = dx0optbest + dxtmp
            resbest = resbest + res_nl[icycle]
            nbest+=1
        elif res1 >= thres_90:
            dJdx0worst = dJdx0worst + dJtmp
            dx0optworst = dx0optworst + dxtmp
            resworst = resworst + res_nl[icycle]
            nworst+=1
    #    dJdx0std = dJdx0std + dJtmp**2
    #    dx0optstd = dx0optstd + dxtmp**2
    ncycle = ics.size
    dJdx0best /= nbest
    dx0optbest /= nbest
    resbest /= nbest
    dJdx0worst /= nworst
    dx0optworst /= nworst
    resworst /= nworst
    #dJdx0std = dJdx0std/ncycle - dJdx0mean**2
    #dJdx0std[dJdx0std<0.0]=0.0
    #dJdx0std = np.sqrt(dJdx0std)
    #dx0optstd = dx0optstd/ncycle - dx0optmean**2
    #dx0optstd[dx0optstd<0.0]=0.0
    #dx0optstd = np.sqrt(dx0optstd)
    
    axdJb.plot(x,dJdx0best,c=colors[solver],**marker_style)
    axdxb.plot(x,dx0optbest,c=colors[solver],**marker_style)
    axdJb.set_title(f'#{nbest} '+r'$\delta J/J=$'+f'{resbest:.2f}')
    axdxb.set_title(f'#{nbest} '+r'$\delta J/J=$'+f'{resbest:.2f}')
    axdJb.set_ylabel(solver)
    axdxb.set_ylabel(solver)
    axdJw.plot(x,dJdx0worst,c=colors[solver],**marker_style)
    axdxw.plot(x,dx0optworst,c=colors[solver],**marker_style)
    axdJw.set_title(f'#{nworst} '+r'$\delta J/J=$'+f'{resworst:.2f}')
    axdxw.set_title(f'#{nworst} '+r'$\delta J/J=$'+f'{resworst:.2f}')
    axdJw.set_ylabel(solver)
    axdxw.set_ylabel(solver)
    ### t-test for zero-mean
    #alpha=0.01
    #dJdx0_comp = np.array(dJdx0_comp)
    #dx0opt_comp = np.array(dx0opt_comp)
    #_, dJ_p = ttest_1samp(dJdx0_comp,1.0e-4,alternative='greater',axis=0)
    #_, dx_p = ttest_1samp(dx0opt_comp,1.0e-4,alternative='greater',axis=0)
    #axdJ.plot(x[dJ_p<alpha],dJdx0mean[dJ_p<alpha],\
    #    lw=0.0,c=colors[solver],marker=markers[solver],**marker_style)
    #axdx.plot(x[dx_p<alpha],dx0optmean[dx_p<alpha],\
    #    lw=0.0,c=colors[solver],marker=markers[solver],**marker_style)
for ax in np.concatenate(\
    (
        axsdJb.flatten(),axsdxb.flatten(),\
        axsdJw.flatten(),axsdxw.flatten(),\
        )):
    ax.fill_between(x[i0:i1],0,1,\
        color='gray',alpha=0.5,transform=ax.get_xaxis_transform(),zorder=0)

for axs in [axsdJb,axsdxb,axsdJw,axsdxw]:
    for i,ax in enumerate(axs.flatten()):
        if i==0:
            ymin,ymax = ax.get_ylim()
        else:
            ax.set_ylim(ymin,ymax)
        ax.grid()
figdJb.suptitle(r'$\frac{\partial J}{\partial \mathbf{x}_0}$'+f" best 10% for ASA FT{vt} {nens} member")
figdxb.suptitle(r'$\delta\mathbf{x}_0^\mathrm{opt}$'+f" best 10% for ASA FT{vt} {nens} member")
figdJb.savefig(figdir/f"composite_dJdx0_best10%_vt{vt}nens{nens}.png",dpi=300)
figdxb.savefig(figdir/f"composite_dx0opt_best10%_vt{vt}nens{nens}.png",dpi=300)
figdJw.suptitle(r'$\frac{\partial J}{\partial \mathbf{x}_0}$'+f" worst 10% for ASA FT{vt} {nens} member")
figdxw.suptitle(r'$\delta\mathbf{x}_0^\mathrm{opt}$'+f" worst 10% for ASA FT{vt} {nens} member")
figdJw.savefig(figdir/f"composite_dJdx0_worst10%_vt{vt}nens{nens}.png",dpi=300)
figdxw.savefig(figdir/f"composite_dx0opt_worst10%_vt{vt}nens{nens}.png",dpi=300)
#plt.show()