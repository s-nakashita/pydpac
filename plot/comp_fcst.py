import sys
import os
import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
from nmc_tools import NMC_tools, wnum2wlen, wlen2wnum
from pathlib import Path

import matplotlib as mpl
from scipy import stats
from plot_heatmap import heatmap, annotate_heatmap

plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
preGM = False
if len(sys.argv)>4:
    preGM = (sys.argv[4]=='T')
ldscl=False
if len(sys.argv)>5:
    ldscl = (sys.argv[5]=='T')

t = np.arange(na)+1
ns = 40 # spinup

wdir = Path(f'/Volumes/FF520/nested_envar/data/{model}')
wdir = Path(f'../work/{model}')
dscldir = wdir / 'var_vs_envar_dscl_m80obs30'
preGMpt = "envar"
datadir = wdir / 'var_vs_envar_dscl_m80obs30'
#datadir  = wdir / 'var_vs_envar_shrink_dct_preGM_partialr_m80obs30'
figdir = datadir

perts = ["envar", "envar_nest","var","var_nest"]
labels = {"envar":"EnVar", "envar_nest":"Nested EnVar", "var":"3DVar", "var_nest":"Nested 3DVar"}
linecolor = {"envar":'tab:orange',"envar_nest":'tab:green',"var":"tab:olive","var_nest":"tab:brown"}

ix_t = np.loadtxt(datadir/"ix_true.txt")
ix_gm = np.loadtxt(datadir/"ix_gm.txt")
ix_lam = np.loadtxt(datadir/"ix_lam.txt")
nx_t = ix_t.size
nx_gm = ix_gm.size
nx_lam = ix_lam.size
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

if not preGM:
    fig, [axg, axl] = plt.subplots(nrows=2,sharex=True,figsize=[10,8],constrained_layout=True)
else:
    fig = plt.figure(figsize=[10,6],constrained_layout=True)
    axl = fig.add_subplot(111)

# nature 
f = datadir/"truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)
print(xt.shape)
xt2x = interp1d(ix_t,xt)
#wnum_t, psd_bg = nmc_t.psd(xt,axis=1)
#axsp.loglog(wnum_t,psd_bg,c='b',lw=1.0,label='Nature bg')

etgm_dict = {}
xdgm_dict = {}
psdgm_dict = {}
etlam_dict = {}
xdlam_dict = {}
psdlam_dict = {}
if ldscl:
    # downscaling
    f = dscldir/"{}_lam_ufext_{}_{}.npy".format(model,op,pt)
    if not f.exists():
        print("not exist {}".format(f))
        exit()
    xfdscl = np.load(f)
    print(xfdscl.shape)
    ft = []
    et_lam = []
    xd_lam = []
    psd_lam = []
    nft = xfdscl.shape[1]
    for ift in range(nft):
        xd1 = xfdscl[ns:na,ift,:] - xt2x(ix_lam)[ns+ift:na+ift,:]
        ettmp = np.sqrt(np.mean(xd1**2,axis=1))
        xdtmp = np.sqrt(np.mean(xd1**2,axis=0))
        wnum, psdtmp = nmc_lam.psd(xd1,axis=1)
        ft.append(ift*6) # hours
        et_lam.append(ettmp)
        xd_lam.append(xdtmp)
        psd_lam.append(psdtmp)
    et_lam = np.array(et_lam)
    xd_lam = np.array(xd_lam)
    psd_lam = np.array(psd_lam)
    print(et_lam.shape)
    print(xd_lam.shape)
    print(psd_lam.shape)
    #axl.errorbar(ft,et_lam.mean(axis=1),yerr=et_lam.std(axis=1),c='k',label='downscaling')
    bplot = axl.boxplot(et_lam.T,patch_artist=True,whis=(0,100))
    for patch in bplot['boxes']:
        patch.set_facecolor('black')
        patch.set_alpha(0.3)
    axl.plot(np.arange(1,nft+1),et_lam.mean(axis=1),c='k',label='downscaling')
    etlam_dict["dscl"] = et_lam
    xdlam_dict["dscl"] = xd_lam
    psdlam_dict["dscl"] = psd_lam

if preGM:
    f = dscldir/"{}_gm_ufext_{}_{}.npy".format(model,op,pt)
    if not f.exists():
        print("not exist {}".format(f))
        exit()
    xfgm = np.load(f)
    print(xfgm.shape)
    ft = []
    xd_gm = []
    psd_gm = []
    nft = xfgm.shape[1]
    for ift in range(nft):
        xd1 = xfgm[ns:na,ift,:] - xt2x(ix_gm)[ns+ift:na+ift,:]
        xdtmp = np.sqrt(np.mean(xd1**2,axis=0))
        wnum_gm, psdtmp = nmc_gm.psd(xd1,axis=1)
        ft.append(ift*6) # hours
        xd_gm.append(xdtmp)
        psd_gm.append(psdtmp)
    xd_gm = np.array(xd_gm)
    psd_gm = np.array(psd_gm)
    xdgm_dict[preGMpt] = xd_gm
    psdgm_dict[preGMpt] = psd_gm

for pt in perts:
    # GM
    if not preGM:
        f = datadir/"{}_gm_ufext_{}_{}.npy".format(model,op,pt)
        if not f.exists():
            print("not exist {}".format(f))
            continue
        xfgm = np.load(f)
        print(xfgm.shape)
        ft = []
        et_gm = []
        xd_gm = []
        psd_gm = []
        nft = xfgm.shape[1]
        for ift in range(nft):
            xd1 = xfgm[ns:na,ift,:] - xt2x(ix_gm)[ns+ift:na+ift,:]
            ettmp = np.sqrt(np.mean(xd1**2,axis=1))
            xdtmp = np.sqrt(np.mean(xd1**2,axis=0))
            wnum_gm, psdtmp = nmc_gm.psd(xd1,axis=1)
            ft.append(ift*6) # hours
            et_gm.append(ettmp)
            xd_gm.append(xdtmp)
            psd_gm.append(psdtmp)
        et_gm = np.array(et_gm)
        xd_gm = np.array(xd_gm)
        psd_gm = np.array(psd_gm)
        print(et_gm.shape)
        print(xd_gm.shape)
        print(psd_gm.shape)
        #axg.errorbar(ft,et_gm.mean(axis=1),yerr=et_gm.std(axis=1),c=linecolor[pt],label=labels[pt])
        bplot = axg.boxplot(et_gm.T,patch_artist=True,whis=(0,100))
        for patch in bplot['boxes']:
            patch.set_facecolor(linecolor[pt])
            patch.set_alpha(0.3)
        axg.plot(np.arange(1,nft+1),et_gm.mean(axis=1),c=linecolor[pt],label=labels[pt])
        etgm_dict[pt] = et_gm
        xdgm_dict[pt] = xd_gm
        psdgm_dict[pt] = psd_gm
    # LAM
    f = datadir/"{}_lam_ufext_{}_{}.npy".format(model,op,pt)
    if not f.exists():
        print("not exist {}".format(f))
        continue
    xflam = np.load(f)
    print(xflam.shape)
    ft = []
    et_lam = []
    xd_lam = []
    psd_lam = []
    nft = xflam.shape[1]
    for ift in range(nft):
        xd1 = xflam[ns:na,ift,:] - xt2x(ix_lam)[ns+ift:na+ift,:]
        ettmp = np.sqrt(np.mean(xd1**2,axis=1))
        xdtmp = np.sqrt(np.mean(xd1**2,axis=0))
        wnum, psdtmp = nmc_lam.psd(xd1,axis=1)
        ft.append(ift*6) # hours
        et_lam.append(ettmp)
        xd_lam.append(xdtmp)
        psd_lam.append(psdtmp)
    et_lam = np.array(et_lam)
    xd_lam = np.array(xd_lam)
    psd_lam = np.array(psd_lam)
    print(et_lam.shape)
    print(xd_lam.shape)
    print(psd_lam.shape)
    #axl.errorbar(ft,et_lam.mean(axis=1),yerr=et_lam.std(axis=1),c=linecolor[pt],label=labels[pt])
    bplot = axl.boxplot(et_lam.T,patch_artist=True,whis=(0,100))
    for patch in bplot['boxes']:
        patch.set_facecolor(linecolor[pt])
        patch.set_alpha(0.3)
    axl.plot(np.arange(1,nft+1),et_lam.mean(axis=1),c=linecolor[pt],label=labels[pt])
    etlam_dict[pt] = et_lam
    xdlam_dict[pt] = xd_lam
    psdlam_dict[pt] = psd_lam
if not preGM:
    axg.hlines([1.0],0,1,colors='gray',ls='dotted',transform=axg.get_yaxis_transform())
    axg.set_ylabel('GM Forecast RMSE')
axl.set_ylabel('LAM Forecast RMSE')
axl.set_xticks(np.arange(1,nft+1,2))
axl.set_xticklabels(ft[::2])
axl.set_xlabel('forecast hours')
axl.hlines([1.0],0,1,colors='gray',ls='dotted',transform=axl.get_yaxis_transform())
axl.legend(loc='upper left',bbox_to_anchor=(1.01,0.95))

fig.savefig(figdir/f"{model}_efcst_{op}.png",dpi=300)
#fig.savefig(figdir/f"{model}_efcst_{op}.pdf")
plt.show()
#exit()

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

for ift in range(1,nft):
    ft1 = ft[ift]
    figgr, axgr = plt.subplots(figsize=[8,5],constrained_layout=True)
    figsp, axsp = plt.subplots(figsize=[8,6],constrained_layout=True)

    if preGM:
        axgr.plot(ix_gm, xdgm_dict[preGMpt][ift,],\
            c='gray',lw=4.0,label='GM')
        axsp.loglog(wnum_gm,psdgm_dict[preGMpt][ift,],c='gray',lw=4.0,label='GM')
    else:
        for pt in xdgm_dict.keys():
            axgr.plot(ix_gm,xdgm_dict[pt][ift,],\
                c=linecolor[pt],alpha=0.5,lw=4.0)#,label=labels[pt])
            axsp.loglog(wnum_gm,psdgm_dict[pt][ift,],\
                c=linecolor[pt],alpha=0.5,lw=4.0)#,label=labels[pt])

    for pt in xdlam_dict.keys():
        axgr.plot(ix_lam,xdlam_dict[pt][ift,],\
            c=linecolor[pt],label=labels[pt])
        axsp.loglog(wnum,psdlam_dict[pt][ift,],\
            c=linecolor[pt],label=labels[pt])

    #axgr.set_ylabel('RMSE')
    axgr.set_xlabel('grid')
    axgr.set_xlim(ix_t[0],ix_t[-1])
    #axgr.hlines([1],0,1,colors='gray',ls='dotted',transform=ax.get_yaxis_transform())
    axgr.legend(loc='upper right')
    #ymin, ymax = axgr.get_ylim()
    #ymax = 1.0
    #ymin = 0.15
    #axgr.set_ylim(ymin,ymax)

    axsp.grid()
    axsp.legend()
    axsp.set_xlabel(r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)")
    #axsp.xaxis.set_major_locator(FixedLocator([180./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi,0.5/np.pi]))
    #axsp.xaxis.set_major_formatter(FixedFormatter([r'$\frac{180}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$',r'$\frac{1}{2\pi}$']))
    axsp.xaxis.set_major_locator(FixedLocator([480,240,120,60,30,12,2]))
    axsp.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','12','2']))
    secax = axsp.secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
    secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
    secax.xaxis.set_major_locator(FixedLocator([np.pi,np.pi/6.,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
    secax.xaxis.set_major_formatter(FixedFormatter([r'$\pi$',r'$\frac{\pi}{6}$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
    
    figgr.suptitle(f'FT{ft1}')
    figsp.suptitle(f'FT{ft1}')
    figgr.savefig(figdir/f"{model}_xd_ft{ft1}_{op}.png",dpi=300)
    #figgr.savefig(figdir/f"{model}_xd_ft{ft}_{op}.pdf")
    figsp.savefig(figdir/f"{model}_errspectra_ft{ft1}_{op}.png",dpi=300)
    #figsp.savefig(figdir/f"{model}_errspectra_f_{op}.pdf")
    plt.show(block=False)
    plt.close()

    # t-test
    methods = methods_gm
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
                e1 = etgm_dict[m1][ift,]
                e2 = etgm_dict[m2][ift,]
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
        ax.set_title(f"t-test for GM FT{ft1}: RMSE row-col")
        fig.tight_layout()
        fig.savefig(figdir/"{}_ef{}_t-test_for_gm_{}.png".format(model, ft1, op),dpi=300)
        plt.close()

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
        ax.set_title(f"t-test for LAM FT{ft1}: RMSE row-col")
        fig.tight_layout()
        fig.savefig(figdir/"{}_ef{}_t-test_for_lam_{}.png".format(model, ft1, op),dpi=300)
        plt.close()
