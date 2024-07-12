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

plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = 'var' #var or envar
if len(sys.argv)>4:
    pt = sys.argv[4]
anl = True
if len(sys.argv)>5:
    anl = (sys.argv[5]=='T')

t = np.arange(na)+1
ns = 40 # spinup

datadir = Path(f'/Volumes/FF520/nested_envar/data/{model}')
datadir = Path(f'../work/{model}')
preGMpt = 'envar'
ldscl=True
#obsloc = ''
#obsloc = '_partiall'
#obsloc = '_partialc'
#obsloc = '_partialr'
obsloc = '_partialm'
dscldir = datadir / 'var_vs_envar_dscl_m80obs30'
lsbdir  = datadir / f'var_vs_envar_lsb_preGM{obsloc}_m80obs30'
lamdir  = datadir / f'var_vs_envar_shrink_dct_preGM{obsloc}_m80obs30'
#if ldscl:
#    figdir = datadir
#else:
figdir = lsbdir

ptlong = {"envar":"EnVar","var":"3DVar"}
labels = {"conv":"DA", "lsb":"DA+LSB", "nest":"Nested DA"}
linecolor = {"conv":"tab:blue","lsb":'tab:orange',"nest":'tab:green'}

ix_t = np.loadtxt(dscldir/"ix_true.txt")
ix_gm = np.loadtxt(dscldir/"ix_gm.txt")
ix_lam = np.loadtxt(dscldir/"ix_lam.txt")
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

fig, ax = plt.subplots(figsize=[8,6],constrained_layout=True)
figsp, axsp = plt.subplots(figsize=[8,7],constrained_layout=True)
psd_dict = {}

# nature background psd
#f = datadir/"truth.npy"
f = dscldir/"truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[ns:na,]
print(xt.shape)
nx_t = xt.shape[1]
xt2x = interp1d(ix_t,xt)
wnum_t, psd_bg = nmc_t.psd(xt,axis=1)
#axsp.loglog(wnum_t,psd_bg,c='b',lw=1.0,label='Nature bg')

# GM
if anl:
    f = dscldir/"xagm_{}_{}.npy".format(op,preGMpt)
else:
    f = dscldir/"{}_xfgm_{}_{}.npy".format(model,op,preGMpt)
if not f.exists():
    print("not exist {}".format(f))
    exit()
xagm = np.load(f)
xdgm = xagm[ns:na,:] - xt2x(ix_gm)
ax.plot(ix_gm, np.sqrt(np.mean(xdgm**2,axis=0)),\
    c='gray',lw=4.0,label='GM')
wnum_gm, psd_gm = nmc_gm.psd(xdgm,axis=1)
axsp.loglog(wnum_gm,psd_gm,c='gray',lw=4.0,label='GM')

if ldscl:
    # downscaling
    if anl:
        f = dscldir/"xalam_{}_{}.npy".format(op,preGMpt)
    else:
        f = dscldir/"{}_xflam_{}_{}.npy".format(model,op,preGMpt)
    if not f.exists():
        print("not exist {}".format(f))
        exit()
    xadscl = np.load(f)
    xddscl = xadscl[ns:na,:] - xt2x(ix_lam)
    ax.plot(ix_lam, np.sqrt(np.mean(xddscl**2,axis=0)),\
        c='k',lw=2.0,label='downscaling')
    wnum, psd_dscl = nmc_lam.psd(xddscl,axis=1,average=False)
    axsp.loglog(wnum,psd_dscl.mean(axis=0),c='k',lw=2.0,label='Downscaling')
    psd_dict["dscl"] = psd_dscl
    #f = dscldir/"xsalam_{}_{}.npy".format(op,preGMpt)
    #xsadscl = np.load(f)
    #ax.plot(ix_lam, np.mean(xsadscl,axis=0),\
    #    c='k',ls='dashed')
# LAM
for key in ['conv','lsb','nest']:
    if key=='conv':
        if anl:
            f = lamdir/"xalam_{}_{}.npy".format(op,pt)
        else:
            f = lamdir/"xflam_{}_{}.npy".format(op,pt)
    elif key=='nest':
        if anl:
            f = lamdir/"xalam_{}_{}_nest.npy".format(op,pt)
        else:
            f = lamdir/"xflam_{}_{}_nest.npy".format(op,pt)
    else:
        if anl:
            f = lsbdir/"xalam_{}_{}.npy".format(op,pt)
        else:
            f = lsbdir/"xflam_{}_{}.npy".format(op,pt)
    if not f.exists():
        print("not exist {}".format(f))
        exit()
    xalam = np.load(f)
    xdlam = xalam[ns:na,:] - xt2x(ix_lam)
    xdlam1d = np.sqrt(np.mean(xdlam**2,axis=0))
    print(f"{key}, RMSE={np.mean(xdlam1d)}")
    ax.plot(ix_lam,xdlam1d,\
    c=linecolor[key],lw=2.0,label=labels[key])
    wnum, psd_lam = nmc_lam.psd(xdlam,axis=1,average=False)
    axsp.loglog(wnum,psd_lam.mean(axis=0),\
        c=linecolor[key],lw=2.0,label=labels[key])
    psd_dict[key] = psd_lam
    #f = lamdir/"xsalam_{}_{}.npy".format(op,pt)
    #xsalam = np.load(f)
    #if pt == 'var' or pt == 'var_nest':
    #    ax.plot(ix_lam[1:-1], np.mean(xsalam,axis=0),\
    #    c=linecolor[pt],ls='dashed')
    #else:
    #    ax.plot(ix_lam, np.mean(xsalam,axis=0),\
    #    c=linecolor[pt],ls='dashed')
#ax.set_ylabel('RMSE')
ax.set_xlabel('grid')
ax.set_xlim(ix_t[0],ix_t[-1])
#ax.hlines([1],0,1,colors='gray',ls='dotted',transform=ax.get_yaxis_transform())
ax.legend(loc='upper right')
#ymin, ymax = ax.get_ylim()
if obsloc != '':
    ymax = 1.0
    ymin = 0.15
    ax.set_ylim(ymin,ymax)
fig.suptitle(ptlong[pt])
if anl:
    fig.savefig(figdir/f"{model}_xd_{op}_{pt}.png",dpi=300)
    fig.savefig(figdir/f"{model}_xd_{op}_{pt}.pdf")
else:
    fig.savefig(figdir/f"{model}_xdf_{op}_{pt}.png",dpi=300)
    fig.savefig(figdir/f"{model}_xdf_{op}_{pt}.pdf")

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
figsp.suptitle(ptlong[pt])
if anl:
    figsp.savefig(figdir/f"{model}_errspectra_{op}_{pt}.png",dpi=300)
    figsp.savefig(figdir/f"{model}_errspectra_{op}_{pt}.pdf")
else:
    figsp.savefig(figdir/f"{model}_errspectra_f_{op}_{pt}.png",dpi=300)
    figsp.savefig(figdir/f"{model}_errspectra_f_{op}_{pt}.pdf")
plt.show()
plt.close()

# t-test
from scipy import stats
##LAM
methods = psd_dict.keys()
for i,m1 in enumerate(methods):
    for j,m2 in enumerate(methods):
        if j<=i: continue
        print(f"{m1}-{m2}")
        psd1 = psd_dict[m1]
        psd2 = psd_dict[m2]
        res = stats.ttest_rel(psd1,psd2,axis=0)
        t_stat = res.statistic
        if np.isnan(t_stat).all(): continue
        pvalue = res.pvalue
        t_conf = res.confidence_interval(confidence_level=0.95)
        dmean = (t_conf.high + t_conf.low)/2.
        fig,axs = plt.subplots(figsize=[6,9],nrows=3,sharex=True)
        axs[0].fill_between(wnum,t_conf.high,t_conf.low,color='tab:blue',alpha=0.5)
        axs[0].set_ylabel("95% interval")
        axs[0].set_xscale("log")
        axs[1].semilogx(wnum,t_stat,lw=2.0)
        axs[1].set_ylabel("t-statistic")
        ymin, ymax = axs[1].get_ylim()
        if ymin*ymax < 0.0:
            axs[1].hlines([0.0],0,1,colors='r',transform=axs[1].get_yaxis_transform(),zorder=1)
        axs[2].semilogx(wnum,pvalue,marker='^',lw=1.0)
        axs[2].hlines([0.05],0,1,colors='r',transform=axs[2].get_yaxis_transform(),zorder=1)
        axs[2].set_ylabel("p-value")
        axs[2].set_ylim(0.,0.1)
        for ax in axs:
            ax.grid(zorder=0)
            #ax.set_xscale("log")
        axs[2].set_xlabel(r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)")
        #ax.xaxis.set_major_locator(FixedLocator([180./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi,0.5/np.pi]))
        #ax.xaxis.set_major_formatter(FixedFormatter([r'$\frac{180}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$',r'$\frac{1}{2\pi}$']))
        axs[2].xaxis.set_major_locator(FixedLocator([480,240,120,60,30,12,2]))
        axs[2].xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','12','2']))
        secax = axs[0].secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
        secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
        secax.xaxis.set_major_locator(FixedLocator([np.pi,np.pi/6.,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
        secax.xaxis.set_major_formatter(FixedFormatter([r'$\pi$',r'$\frac{\pi}{6}$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
        fig.suptitle(f"LAM {op}: {m1} - {m2}")
        fig.tight_layout()
        if anl:
            fig.savefig(figdir/"{}_errspectra_lam_t-test_{}_{}-{}.png".format(model,op,m1,m2),dpi=300)
        else:
            fig.savefig(figdir/"{}_errspectra_f_lam_t-test_{}_{}-{}.png".format(model,op,m1,m2),dpi=300)
        #plt.show()
        plt.close()
