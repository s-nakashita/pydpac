import sys
import os
import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from matplotlib import colormaps
from nmc_tools import NMC_tools, wnum2wlen, wlen2wnum
from pathlib import Path

import matplotlib as mpl
from scipy import stats
from plot_heatmap import heatmap, annotate_heatmap

plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = 'var' #var or envar
if len(sys.argv)>4:
    pt = sys.argv[4]

preGM = True
ldscl=True

t = np.arange(na)+1
ns = 40 # spinup

ldble_gm  = 1.76 * 0.05 / 6 # LII Lyapunov exponent (1/hour)
ldble_lam = 3.46 * 0.05 / 6 # LIII Lyapunov exponent (1/hour)
print(f"GM Lyapunov exponents = {ldble_gm:.3e} [h^-1^]")
print(f"LAM Lyapunov exponents = {ldble_lam:.3e} [h^-1^]")
print(f"GM doubling time = {np.log(2.0)/ldble_gm:.2f} [hours]")
print(f"LAM doubling time = {np.log(2.0)/ldble_lam:.2f} [hours]")

wdir = Path(f'/Volumes/FF520/nested_envar/data/{model}')
wdir = Path(f'../work/{model}')
obsloc = ''
if len(sys.argv)>5:
    obsloc = sys.argv[5]
#obsloc = '_partiall'
#obsloc = '_partialc'
#obsloc = '_partialr'
#obsloc = '_partialm'
dscldir = wdir / 'var_vs_envar_dscl_m80obs30'
preGMpt = "envar"
lsbdir  = wdir / f'var_vs_envar_lsb_preGM{obsloc}_m80obs30'
lamdir  = wdir / f'var_vs_envar_shrink_dct_preGM{obsloc}_m80obs30'
#figdir = lsbdir / 'fcst'
#if not figdir.exists(): figdir.mkdir(parents=True)
figpngdir = Path(os.environ['HOME']+'/Writing/nested_envar/figpng')
figpdfdir = Path(os.environ['HOME']+'/Writing/nested_envar/figpdf')
#figpngdir = Path(os.environ['HOME']+'/Writing/dissertation/nested_envar/fig')
#figpdfdir = Path(os.environ['HOME']+'/Writing/dissertation/nested_envar/fig')
if obsloc == '':
    figpngdir = figpngdir / 'uniform'
    figpdfdir = figpdfdir / 'uniform'
else:
    figpngdir = figpngdir / obsloc[1:]
    figpdfdir = figpdfdir / obsloc[1:]
if not figpngdir.exists():
    figpngdir.mkdir(parents=True)
if not figpdfdir.exists():
    figpdfdir.mkdir(parents=True)
figdir = figpngdir / 'fcst'
if not figdir.exists(): figdir.mkdir(parents=True)

ptlong = {"envar":"EnVar","var":"3DVar"}
labels = {"conv":ptlong[pt], "lsb":"BLSB+"+ptlong[pt], "nest":"Nested "+ptlong[pt], "dscl":"Dscl"}
linecolor = {"conv":"tab:blue","lsb":'tab:orange',"nest":'tab:green',"dscl":'k'}
captions = {"envar":"(b)","var":"(a)"}

ix_t = np.loadtxt(dscldir/"ix_true.txt")
ix_gm = np.loadtxt(dscldir/"ix_gm.txt")
ix_lam = np.loadtxt(dscldir/"ix_lam.txt")
nx_t = ix_t.size
nx_gm = ix_gm.size
nx_lam = ix_lam.size
i0 = np.argmin(np.abs(ix_gm - ix_lam[0]))
i1 = np.argmin(np.abs(ix_gm - ix_lam[-1]))
xlim = 15.0
nghost = 0 # ghost region for periodicity in LAM
ix_t_rad = ix_t * 2.0 * np.pi / nx_t
ix_gm_rad = ix_gm * 2.0 * np.pi / nx_t
ix_lam_rad = ix_lam * 2.0 * np.pi / nx_t
Lx_gm = 2.0 * np.pi
#dwindow = (1.0 + np.cos(np.pi*np.arange(1,nghost+1)/nghost))*0.5
Lx_lam = 2.0 * np.pi * nx_lam / nx_t
nmc_t = NMC_tools(ix_t_rad,cyclic=True,ttype='c')
nmc_gm = NMC_tools(ix_gm_rad,cyclic=True,ttype='c')
nmc_lam = NMC_tools(ix_lam_rad,cyclic=False,ttype='c')

fig = plt.figure(figsize=[8,6],constrained_layout=True)
axl = fig.add_subplot(111)
figl = plt.figure(figsize=[8,6],constrained_layout=True)
axll = figl.add_subplot(111) # log-plot
figb = plt.figure(figsize=[10,6],constrained_layout=True)
axb = figb.add_subplot(111)

# nature 
f = dscldir/"truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)
print(xt.shape)
xt2x = interp1d(ix_t,xt)
#wnum_t, psd_bg = nmc_t.psd(xt,axis=1,average=False)
psd_t = []
ift = 0
while ift < 9:
    wnum_t, psd_bg = nmc_lam.psd(xt2x(ix_lam)[ns+ift:na+ift,:],axis=1,average=False)
    psd_t.append(np.mean(psd_bg,axis=0))
    ift += 1
psd_t = np.array(psd_t)
#axsp.loglog(wnum_t,psd_bg,c='b',lw=1.0,label='Nature bg')

etgm_dict = {}
xdgm_dict = {}
psdbgm_dict = {}
psdgm_dict = {}
egrowth0_24h_dict = {}
egrowth24_48h_dict = {}
if preGM:
    f = dscldir/"{}_gm_ufext_{}_{}.npy".format(model,op,preGMpt)
    if not f.exists():
        print("not exist {}".format(f))
        exit()
    xfgm = np.load(f)
    print(xfgm.shape)
    xg2x = interp1d(ix_gm,xfgm,axis=2)
    ft = []
    et_gm = []
    xd_gm = []
    psdb_gm = []
    psd_gm = []
    ift = 0
    while ift < 9:
        #xd1 = xfgm[ns:na,ift,i0:i1+1] - xt2x(ix_gm[i0:i1+1])[ns+ift:na+ift,:]
        x1 = xg2x(ix_lam)[ns:na,ift,:]
        xd1 = x1 - xt2x(ix_lam)[ns+ift:na+ift,:]
        ettmp = np.sqrt(np.mean(xd1**2,axis=1))
        xdtmp = np.sqrt(np.mean(xd1**2,axis=0))
        #wnum_gm, psdtmp = nmc_gm.psd(xd1,axis=1)
        wnum_gm, psdtmp = nmc_lam.psd(x1,axis=1)
        psdb_gm.append(psdtmp)
        wnum_gm, psdtmp = nmc_lam.psd(xd1,axis=1)
        ft.append(ift*6) # hours
        et_gm.append(ettmp)
        xd_gm.append(xdtmp)
        psd_gm.append(psdtmp)
        ift += 1
    et_gm = np.array(et_gm)
    xd_gm = np.array(xd_gm)
    psdb_gm = np.array(psdb_gm)
    psd_gm = np.array(psd_gm)
    nft = et_gm.shape[0]
    axl.plot(np.arange(1,nft+1),et_gm.mean(axis=1),c='gray',lw=4.0,alpha=0.7,label='GM')
    axll.semilogy(np.arange(1,nft+1),et_gm.mean(axis=1),c='gray',lw=4.0,alpha=0.7,label='GM')
    egrowth_24h = et_gm[4,:] / et_gm[0,:]
    egrowth_48h = et_gm[8,:] / et_gm[4,:]
    egrowth0_24h_dict['GM'] = egrowth_24h
    egrowth24_48h_dict['GM'] = egrowth_48h
    #print(f"GM error growth in  0-24 hours = {egrowth_24h.mean()}+-{egrowth_24h.std()}")
    #print(f"GM error growth in 24-48 hours = {egrowth_48h.mean()}+-{egrowth_48h.std()}")
    etgm_dict[preGMpt] = et_gm
    xdgm_dict[preGMpt] = xd_gm
    psdbgm_dict[preGMpt] = psdb_gm
    psdgm_dict[preGMpt] = psd_gm

etlam_dict = {}
xdlam_dict = {}
psdblam_dict = {}
psdlam_dict = {}
if ldscl:
    keys = ['dscl','conv','lsb','nest']
else:
    keys = ['conv','lsb','nest']
for key in keys:
    # LAM
    if key=='dscl':
        # downscaling
        f = dscldir/"{}_lam_ufext_preGM_{}_{}.npy".format(model,op,preGMpt)
    elif key=='conv':
        f = lamdir/"{}_lam_ufext_preGM_{}_{}.npy".format(model,op,pt)
    elif key=='nest':
        f = lamdir/"{}_lam_ufext_preGM_{}_{}_nest.npy".format(model,op,pt)
    else:
        f = lsbdir/"{}_lam_ufext_preGM_{}_{}.npy".format(model,op,pt)
    if not f.exists():
        print("not exist {}".format(f))
        continue
    xflam = np.load(f)
    print(xflam.shape)
    ft = []
    et_lam = []
    xd_lam = []
    psdb_lam = []
    psd_lam = []
    ift = 0
    while ift < 9:
        x1 = xflam[ns:na,ift,:]
        xd1 = x1 - xt2x(ix_lam)[ns+ift:na+ift,:]
        ettmp = np.sqrt(np.mean(xd1**2,axis=1))
        xdtmp = np.sqrt(np.mean(xd1**2,axis=0))
        wnum, psdtmp = nmc_lam.psd(x1,axis=1)
        psdb_lam.append(psdtmp)
        wnum, psdtmp = nmc_lam.psd(xd1,axis=1)
        ft.append(ift*6) # hours
        et_lam.append(ettmp)
        xd_lam.append(xdtmp)
        psd_lam.append(psdtmp)
        ift += 1
    et_lam = np.array(et_lam)
    xd_lam = np.array(xd_lam)
    psdb_lam = np.array(psdb_lam)
    psd_lam = np.array(psd_lam)
    nft = et_lam.shape[0]
    print(et_lam.shape)
    print(xd_lam.shape)
    print(psdb_lam.shape)
    print(psd_lam.shape)
    #axl.errorbar(ft,et_lam.mean(axis=1),yerr=et_lam.std(axis=1),c=linecolor[key],label=labels[key])
    axl.plot(np.arange(1,nft+1),et_lam.mean(axis=1),c=linecolor[key],label=labels[key])
    axll.semilogy(np.arange(1,nft+1),et_lam.mean(axis=1),c=linecolor[key],label=labels[key])
    egrowth_24h = et_lam[4,:] / et_lam[0,:]
    egrowth_48h = et_lam[8,:] / et_lam[4,:]
    egrowth0_24h_dict[key] = egrowth_24h
    egrowth24_48h_dict[key] = egrowth_48h
    #print(f"{key} error growth in  0-24 hours = {egrowth_24h.mean()}+-{egrowth_24h.std()}")
    #print(f"{key} error growth in 24-48 hours = {egrowth_48h.mean()}+-{egrowth_48h.std()}")
    etlam_dict[key] = et_lam
    xdlam_dict[key] = xd_lam
    psdblam_dict[key] = psdb_lam
    psdlam_dict[key] = psd_lam
    # adjusted t-test
    for ift in range(nft):
        e1 = etlam_dict['dscl'][ift,]
        e2 = et_lam[ift,]
        d = e1 - e2
        dmean = d.mean()
        vard = np.sum((d-dmean)**2)
        dm = d[:-1] - d[:-1].mean()
        dp = d[1:] - d[1:].mean()
        r1 = np.sum(dm*dp)/np.sqrt(np.sum(dm*dm))/np.sqrt(np.sum(dp*dp))
        ne = d.size * (1-r1)/(1+r1)
        std = np.sqrt(vard/(d.size-1))
        tval = dmean / (std / np.sqrt(ne))
        pval = 1.0 - stats.t.cdf(tval,ne)
        print(f"{key} FT{ift*6} p-value={pval:.3e}")
        if pval < 5.0e-2:
            axl.plot([ift+1],e2.mean(),c=linecolor[key],lw=0.0,marker='o',ms=6)

nmethods = len(etlam_dict)
width = 0.75 / nmethods
xoffset = 0.5 * width * (nmethods - 1)
xaxis = np.arange(1,nft+1) - xoffset
for key in etlam_dict.keys():
    et_lam = etlam_dict[key]
    bplot = axb.boxplot(et_lam.T,positions=xaxis,widths=width,patch_artist=True,\
        medianprops={"color":linecolor[key]},\
        whiskerprops={"color":linecolor[key]},#whis=0.0,\
        flierprops={"color":linecolor[key]},#showfliers=False)
        label=labels[key])
    for patch in bplot['boxes']:
        patch.set_facecolor(linecolor[key])
        patch.set_alpha(0.3)
    xaxis = xaxis + width
#y0 = 0.5
#y = y0 * np.exp(np.array(ft)*ldble_lam)
#axl.plot(np.arange(1,nft+1),y,c='gray',lw=2.0,alpha=0.5,zorder=0)
for ax in [axl,axll,axb]:
    ax.set_ylabel(f'forecast RMSE')
    ax.set_xticks(np.arange(1,nft+1,2))
    ax.set_xticks(np.arange(1,nft+1),minor=True)
    ax.set_xticklabels(ft[::2])
    ax.grid(which='both')
    ax.set_xlabel('forecast hours')
    #axl.hlines([1.0],0,1,colors='gray',ls='dotted',transform=axl.get_yaxis_transform())
if preGM:
    #ax.set_ylim(0.0, 6.0)
    axb.set_ylim(0.0, 2.0)
    xmin, _ = axb.get_xlim()
    axb.set_xlim(xmin, 9.5)
axl.legend(loc='upper left')
axll.legend(loc='upper left')
axb.legend(loc='upper left',bbox_to_anchor=(1.01,0.95))

fig.suptitle(captions[pt],ha='left',x=0.05,fontsize=24)
fig.savefig(figpngdir/f"{model}_efcst48_{op}_{pt}.png",dpi=300)
fig.savefig(figpdfdir/f"{model}_efcst48_{op}_{pt}.pdf")
figl.suptitle(captions[pt],ha='left',x=0.05,fontsize=24)
figl.savefig(figpngdir/f"{model}_efcst48_log_{op}_{pt}.png",dpi=300)
figl.savefig(figpdfdir/f"{model}_efcst48_log_{op}_{pt}.pdf")
figb.savefig(figpngdir/f"{model}_efcst48_box_{op}_{pt}.png",dpi=300)
figb.savefig(figpdfdir/f"{model}_efcst48_box_{op}_{pt}.pdf")
#fig.savefig(figdir/f"{model}_efcst_{op}.pdf")
plt.show()
plt.close()

fig, axs = plt.subplots(ncols=2,figsize=[10,6],sharey=True,constrained_layout=True)
xtick0labels = []
xtick1labels = []
for i,key in enumerate(egrowth0_24h_dict.keys()):
    egrowth24h = egrowth0_24h_dict[key]
    egrowth48h = egrowth24_48h_dict[key]
    axs[0].boxplot(egrowth24h,positions=[i+1],meanline=True,showmeans=True)
    axs[1].boxplot(egrowth48h,positions=[i+1],meanline=True,showmeans=True)
    xtick0labels.append(
        f'{key}\n{egrowth24h.mean():.2f}'+r'$\pm$'+f'{egrowth24h.std():.2f}'
    )
    xtick1labels.append(
        f'{key}\n{egrowth48h.mean():.2f}'+r'$\pm$'+f'{egrowth48h.std():.2f}'
    )
n = len(egrowth0_24h_dict)
axs[0].set_xticks(np.arange(1,n+1))
axs[0].set_xticklabels(xtick0labels,fontsize=12)
axs[0].set_title('FT00-FT24')
axs[1].set_xticks(np.arange(1,n+1))
axs[1].set_xticklabels(xtick1labels,fontsize=12)
axs[1].set_title('FT24-FT48')
fig.suptitle(ptlong[pt])
fig.savefig(figpngdir/f'{model}_egrowth_{op}_{pt}.png',dpi=300)
plt.show()
plt.close()

#import pandas as pd
#for ift in range(9):
#    datadict = {}
#    for key in etlam_dict.keys():
#        data = etlam_dict[key]
#        datadict[key] = data[ift,]
#    df = pd.DataFrame(datadict)
#    df.to_csv(figdir/f"efcst_ft{ift*6}_{op}_{pt}.csv")
#exit()

"""
cmap=plt.get_cmap("PiYG_r")
cl0 = cmap(cmap.N//2)
cl1 = cmap(cmap.N//4*3)
cl2a = cmap(cmap.N-1)
cl2b = cmap(0)
cl3 = cmap(cmap.N//4)
cl4 = cmap(cmap.N//2-1)
cdict = {
    'red':(
        (0.0, cl0[0], cl0[0]),
        (0.25, cl1[0], cl1[0]),
        (0.5, cl2a[0], cl2b[0]),
        (0.75, cl3[0], cl3[0]),
        (1.0, cl4[0], cl4[0]),
    ),
    'green':(
        (0.0, cl0[1], cl0[1]),
        (0.25, cl1[1], cl1[1]),
        (0.5, cl2a[1], cl2b[1]),
        (0.75, cl3[1], cl3[1]),
        (1.0, cl4[1], cl4[1]),
    ),
    'blue':(
        (0.0, cl0[2], cl0[2]),
        (0.25, cl1[2], cl1[2]),
        (0.5, cl2a[2], cl2b[2]),
        (0.75, cl3[2], cl3[2]),
        (1.0, cl4[2], cl4[2]),
    ),
    'alpha':(
        (0.0, cl0[3], cl0[3]),
        (0.25, cl1[3], cl1[3]),
        (0.5, cl2a[3], cl2b[3]),
        (0.75, cl3[3], cl3[3]),
        (1.0, cl4[3], cl4[3]),
    ),
}
mycmap = mpl.colors.LinearSegmentedColormap('GrPi',cdict)
norm = mpl.colors.BoundaryNorm([-0.1,-0.05,-0.01,0.0,0.01,0.05,0.1],8,extend="both")
sigrates = ['<90%','>90%','>95%','>99%','>99%','>95%','>90%','<90%']
fmt = mpl.ticker.FuncFormatter(lambda x, pos: sigrates[norm(x)])

methods_gm = etgm_dict.keys()
methods_lam = etlam_dict.keys()
"""
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
#figgr, axsgr = plt.subplots(nrows=2,ncols=2,figsize=[12,8],constrained_layout=True,sharey=True)
#figsp, axssp = plt.subplots(nrows=2,ncols=2,figsize=[12,8],constrained_layout=True,sharey=True)
nrows = 2
ncols = 2 #3
figheight = nrows * 3
figwidth = ncols * 4
figgr = plt.figure(figsize=[figwidth,figheight],constrained_layout=True)
figsp = plt.figure(figsize=[figwidth,figheight],constrained_layout=True)
figsat = plt.figure(figsize=[figwidth,figheight],constrained_layout=True)
axs = dict()
for figname, fig in zip(['gr','sp','sat'],[figgr,figsp,figsat]):
    ax00 = fig.add_subplot(nrows,ncols,1)
    ax01 = fig.add_subplot(nrows,ncols,2,sharey=ax00)
    #ax02 = fig.add_subplot(233,sharey=ax00)
    ax11 = fig.add_subplot(nrows,ncols,3)
    ax12 = fig.add_subplot(nrows,ncols,4,sharey=ax11)
    #axs[figname] = [ax00,ax01,ax02,ax11,ax12]
    axs[figname] = [ax00,ax01,ax11,ax12]
handle_list = []
label_list = []
cmap = plt.get_cmap('tab10')
colors = colormaps['jet'](np.linspace(0,1,nft))
captions = ['(a)','(b)','(c)','(d)','(e)']
for ift in range(nft):
    ft1 = ft[ift]
    #c = cmap(ift)
    c = colors[ift]
    ##l1, = axsgr[0,0].plot(ix_gm,xdgm_dict[preGMpt][ift,],label=f'FT{ft1}')
    #l1, = axs['gr'][0].plot(ix_lam,xdgm_dict[preGMpt][ift,],c=c,label=f'FT{ft1}')
    #axs['sp'][0].loglog(wnum_gm,psdbgm_dict[preGMpt][ift,],c=c,alpha=0.5)
    #axs['sp'][0].loglog(wnum_gm,psdgm_dict[preGMpt][ift,],c=c,ls='dotted',label=f'FT{ft1}')
    #r = psdgm_dict[preGMpt][ift,] / psdbgm_dict[preGMpt][ift,]
    #axs['sat'][0].semilogx(wnum_gm,r*100.0,c=c,label=f'FT{ft1}')
    #if ift==0:
    #    for figname in axs.keys():
    #        axs[figname][0].set_title('interpolated GM')
    irow=0
    icol=0
    iplot=0 #1
    for key in xdlam_dict.keys():
        #axsgr[irow,icol].plot(ix_lam,xdgm_dict[preGMpt][ift,],c=cmap(ift),alpha=0.5,ls='dashed',)
        #axssp[irow,icol].loglog(wnum_gm,psdgm_dict[preGMpt][ift,],c=cmap(ift),alpha=0.5,ls='dashed')
        l1,=axs['gr'][iplot].plot(ix_lam,xdlam_dict[key][ift,],c=c,label=f'FT{ft1}')
        axs['sp'][iplot].loglog(wnum,psdblam_dict[key][ift,],c=c,alpha=0.5)
        axs['sp'][iplot].loglog(wnum,psdlam_dict[key][ift,],c=c,ls='dotted',label=f'FT{ft1}')
        r = psdlam_dict[key][ift,]/psdblam_dict[key][ift,]
        axs['sat'][iplot].semilogx(wnum,r*100.0,c=c,label=f'FT{ft1}')
        if ift==0:
            for figname in axs.keys():
                annotate_dict = {
                    'xy':(0,1),'xycoords':'axes fraction',
                    'xytext':(0.5,-0.5),'textcoords':'offset fontsize',
                    'fontsize':'medium','va':'top','ha':'left',
                    'bbox':dict(facecolor='0.7',edgecolor='none',pad=3.0)
                }
                if figname=='sp':
                    annotate_dict.update(
                        xy=(1,1),
                        xytext=(-0.5,-0.5),
                        ha='right'
                        )
                axs[figname][iplot].annotate(f"{labels[key]}",**annotate_dict)
                #axs[figname][iplot].set_title(labels[key])
        iplot += 1
        icol+=1
        if icol>=2: icol=0; irow=1
    handle_list.append(l1)
    label_list.append(f'FT{ft1}')
for axgr in axs['gr']:
    axgr.set_xlabel('grid')
    axgr.grid()
#    axgr.legend()
for i,axssp in enumerate([axs['sp'],axs['sat']]):
    for j,axsp in enumerate(axssp):
        axsp.grid()
#    axsp.legend()
        if i==0:
            axsp.set_ylim(1.0e-7,1.0e1)
        if i==1:
            axsp.set_ylim(0.0,200.0)
        #axsp.xaxis.set_major_locator(FixedLocator([180./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi,0.5/np.pi]))
        #axsp.xaxis.set_major_formatter(FixedFormatter([r'$\frac{180}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$',r'$\frac{1}{2\pi}$']))
        axsp.xaxis.set_major_locator(FixedLocator([480,240,120,60,30,12,2]))
        secax = axsp.secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
        secax.xaxis.set_major_locator(FixedLocator([np.pi,np.pi/6.,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
        if j<2:
            axsp.xaxis.set_major_formatter(NullFormatter())
            secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
            secax.xaxis.set_major_formatter(FixedFormatter([r'$\pi$',r'$\frac{\pi}{6}$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
        else:
            axsp.set_xlabel(r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)")
            axsp.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','12','2']))
            secax.xaxis.set_major_formatter(NullFormatter())
axs['gr'][0].set_ylabel('RMSE')
axs['sp'][0].set_ylabel('Variance (solid), Error (dotted)')
axs['sat'][0].set_ylabel(r'$V_e/V_b$ [%]')
#axs['gr'][ncols].set_ylabel('RMSE')
#axs['sp'][ncols].set_ylabel('Variance (solid), Error (dotted)')
#axs['sat'][ncols].set_ylabel(r'$V_e/V_b$ [%]')
#axsgr[1,0].set_ylabel('RMSE')
#axssp[1,0].set_ylabel('Error')
#axsgr[1,0].remove()
#axssp[1,0].remove()
legend_dict = dict(loc='upper left', bbox_to_anchor=(1.01,1.1),facecolor='0.7',fontsize=12)
axs['gr'][ncols-1].legend(handle_list, label_list, **legend_dict)
axs['sp'][ncols-1].legend(handle_list, label_list, **legend_dict)
axs['sat'][ncols-1].legend(handle_list, label_list, **legend_dict)
#figgr.suptitle(f'{ptlong[pt]}')
#figsp.suptitle(f'{ptlong[pt]}')
#figsat.suptitle(f'{ptlong[pt]}, variance power spectra')
figgr.savefig(figdir/f"{model}_xd_comp_{op}_{pt}.png",dpi=300)
figsp.savefig(figdir/f"{model}_spectra_comp_{op}_{pt}.png",dpi=300)
figsat.savefig(figdir/f"{model}_errsat_comp_{op}_{pt}.png",dpi=300)
plt.show()
plt.close(fig=figgr)
plt.close(fig=figsp)
plt.close(fig=figsat)
#exit()
for ift in range(nft):
    ft1 = ft[ift]
    # RMSE in state space
    figgr, axgr = plt.subplots(figsize=[8,5],constrained_layout=True)
    # Error in spectral space
    figsp, axsp = plt.subplots(figsize=[8,7],constrained_layout=True)
    # variance power spectra
    figvp, axvp = plt.subplots(figsize=[8,7],constrained_layout=True)

    # nature run
    axvp.loglog(wnum_t,psd_t[ift,],c='blue',lw=2.0,label='Nature')
    if ft1 > 0:
        if preGM:
            #axgr.plot(ix_gm, xdgm_dict[preGMpt][ift,],\
            axgr.plot(ix_lam, xdgm_dict[preGMpt][ift,],\
                c='gray',lw=4.0,label='interpolated GM')
            axsp.loglog(wnum_gm,psdgm_dict[preGMpt][ift,],c='gray',lw=4.0,label='interpolated GM')
            axvp.loglog(wnum_gm,psdbgm_dict[preGMpt][ift,],c='gray',lw=4.0,label='interpolated GM')
        else:
            for key in xdgm_dict.keys():
                axgr.plot(ix_gm,xdgm_dict[key][ift,],\
                    c=linecolor[key],alpha=0.5,lw=4.0)#,label=labels[key])
                axsp.loglog(wnum_gm,psdgm_dict[key][ift,],\
                    c=linecolor[key],alpha=0.5,lw=4.0)#,label=labels[key])
                axvp.loglog(wnum_gm,psdbgm_dict[key][ift,],\
                    c=linecolor[key],alpha=0.5,lw=4.0)

    for key in xdlam_dict.keys():
        axgr.plot(ix_lam,xdlam_dict[key][ift,],\
            c=linecolor[key],label=labels[key])
        axsp.loglog(wnum,psdlam_dict[key][ift,],\
            c=linecolor[key],label=labels[key])
        axvp.loglog(wnum,psdblam_dict[key][ift,],\
            c=linecolor[key],label=labels[key])

    #axgr.set_ylabel('RMSE')
    axgr.set_xlabel('grid')
    axgr.set_xlim(ix_t[0],ix_t[-1])
    #axgr.hlines([1],0,1,colors='gray',ls='dotted',transform=ax.get_yaxis_transform())
    axgr.legend(loc='upper right')
    #ymin, ymax = axgr.get_ylim()
    #ymax = 1.0
    #ymin = 0.15
    #axgr.set_ylim(ymin,ymax)

    for ax1 in [axsp, axvp]:
        ax1.grid()
        ax1.legend()
        if ift==0:
            ymin, ymax = ax1.get_ylim()
            ax1.set_ylim(4.0e-7,ymax)
        ax1.set_xlabel(r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)")
        #ax1.xaxis.set_major_locator(FixedLocator([180./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi,0.5/np.pi]))
        #ax1.xaxis.set_major_formatter(FixedFormatter([r'$\frac{180}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$',r'$\frac{1}{2\pi}$']))
        ax1.xaxis.set_major_locator(FixedLocator([480,240,120,60,30,12,2]))
        ax1.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','12','2']))
        secax = ax1.secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
        secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
        secax.xaxis.set_major_locator(FixedLocator([np.pi,np.pi/6.,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
        secax.xaxis.set_major_formatter(FixedFormatter([r'$\pi$',r'$\frac{\pi}{6}$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
    
    figgr.suptitle(f'{ptlong[pt]}, RMSE, FT{ft1}')
    figsp.suptitle(f'{ptlong[pt]}, error, FT{ft1}')
    figvp.suptitle(f'{ptlong[pt]}, background, FT{ft1}')
    figgr.savefig(figdir/f"{model}_xd_ft{ft1:03d}_{op}_{pt}.png",dpi=300)
    #figgr.savefig(figdir/f"{model}_xd_ft{ft}_{op}.pdf")
    figsp.savefig(figdir/f"{model}_errspectra_ft{ft1:03d}_{op}_{pt}.png",dpi=300)
    #figsp.savefig(figdir/f"{model}_errspectra_f_{op}.pdf")
    figvp.savefig(figdir/f"{model}_spectra_ft{ft1:03d}_{op}_{pt}.png",dpi=300)
    #figvp.savefig(figdir/f"{model}_errspectra_ft{ft1:03d}_{op}_{pt}.pdf")
    plt.show(block=False)
    plt.close(fig=figgr)
    plt.close(fig=figsp)
    plt.close(fig=figvp)
"""
    # t-test
    methods = methods_lam
    nmethods = len(methods)
    if nmethods>1:
        pmatrix = np.eye(nmethods)
        tmatrix = np.eye(nmethods)
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if m1==m2:
                    tmatrix[i,j] = np.nan
                    pmatrix[i,j] = np.nan
                    continue
                e1 = etlam_dict[m1][ift,]
                e2 = etlam_dict[m2][ift,]
                res_ttest = stats.ttest_rel(e1,e2)
                tmatrix[i,j] = res_ttest.statistic
                pmatrix[i,j] = res_ttest.pvalue
                if pmatrix[i,j] < 1e-16:
                    pmatrix[i,j] = 1e-16
                if tmatrix[i,j]>0.0:
                    pmatrix[i,j] = pmatrix[i,j] * -1.0
                print(f"{m1},{m2} t-stat:{tmatrix[i,j]:.3f} pvalue:{pmatrix[i,j]:.3e}")
        print("")
        fig, ax = plt.subplots(figsize=[8,6])
        im, _ = heatmap(pmatrix,methods,methods,ax=ax,\
            cmap=mycmap.resampled(8), norm=norm,\
            cbar_kw=dict(ticks=[-0.075,-0.03,-0.005,0.005,0.03,0.075],format=fmt,extend="both"),\
            cbarlabel="significance")
        annotate_heatmap(im,data=tmatrix,thdata=np.abs(pmatrix),\
            valfmt="{x:.2f}",fontweight="bold",\
            threshold=0.05,textcolors=("white","black"))
        ax.set_title(f"t-test for LAM {ptlong[pt]} FT{ft1}: RMSE row-col")
        fig.tight_layout()
        fig.savefig(figdir/"{}_ef{:03d}_t-test_for_lam_{}_{}.png".format(model, ft1, op, pt),dpi=300)
        plt.close()
"""
