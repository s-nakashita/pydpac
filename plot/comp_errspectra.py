import sys
import os
import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
from nmc_tools import psd, wnum2wlen, wlen2wnum
from pathlib import Path

plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])

t = np.arange(na)+1
ns = 40 # spinup

datadir = Path(f'/Volumes/FF520/nested_envar/data/{model}')
preGMpt = 'envar'
dscldir = datadir / 'var_vs_envar_dscl_m80obs30'
lamdir  = datadir / 'var_vs_envar_preGM_m80obs30'

perts = ["envar", "envar_nest","var","var_nest"]
labels = {"envar":"EnVar", "envar_nest":"Nested EnVar", "var":"3DVar", "var_nest":"Nested 3DVar"}
linecolor = {"envar":'tab:orange',"envar_nest":'tab:green',"var":"tab:olive","var_nest":"tab:brown"}

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

fig, ax = plt.subplots(figsize=[10,5],constrained_layout=True)
figsp, axsp = plt.subplots(figsize=[10,6],constrained_layout=True)
psd_dict = {}

# nature background psd
f = datadir/"truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[:na,]
print(xt.shape)
nx_t = xt.shape[1]
xt2x = interp1d(ix_t,xt)
wnum_t, psd_bg = psd(xt,ix_t_rad,axis=1)
axsp.loglog(wnum_t,psd_bg,c='b',lw=1.0,label='Nature bg')

# GM
f = dscldir/"xagm_{}_{}.npy".format(op,preGMpt)
if not f.exists():
    print("not exist {}".format(f))
    exit()
xagm = np.load(f)
xdgm = xagm - xt2x(ix_gm)
ax.plot(ix_gm, np.sqrt(np.mean(xdgm**2,axis=0)),\
    c='gray',lw=4.0,label='GM')
wnum_gm, psd_gm = psd(xdgm,ix_gm_rad,axis=1)
axsp.loglog(wnum_gm,psd_gm,c='gray',lw=4.0,label='GM')

# downscaling
f = dscldir/"xalam_{}_{}.npy".format(op,preGMpt)
if not f.exists():
    print("not exist {}".format(f))
    exit()
xadscl = np.load(f)
xddscl = xadscl - xt2x(ix_lam)
ax.plot(ix_lam, np.sqrt(np.mean(xddscl**2,axis=0)),\
    c='k',lw=2.0,label='downscaling')
wnum, psd_dscl, xdetrend = psd(xddscl,ix_lam_rad,axis=1,\
    cyclic=False,nghost=0,detrend=True,average=False)
axsp.loglog(wnum,psd_dscl.mean(axis=0),c='k',lw=2.0,label='Downscaling')
psd_dict["dscl"] = psd_dscl
#f = dscldir/"xsalam_{}_{}.npy".format(op,preGMpt)
#xsadscl = np.load(f)
#ax.plot(ix_lam, np.mean(xsadscl,axis=0),\
#    c='k',ls='dashed')
# LAM
for pt in perts:
    f = lamdir/"xalam_{}_{}.npy".format(op,pt)
    if not f.exists():
        print("not exist {}".format(f))
        exit()
    xalam = np.load(f)
    xdlam = xalam - xt2x(ix_lam)
    ax.plot(ix_lam,np.sqrt(np.mean(xdlam**2,axis=0)),\
    c=linecolor[pt],lw=2.0,label=labels[pt])
    wnum, psd_lam, xdetrend = psd(xdlam,ix_lam_rad,axis=1,\
    cyclic=False,nghost=0,detrend=True,average=False)
    axsp.loglog(wnum,psd_lam.mean(axis=0),\
        c=linecolor[pt],lw=2.0,label=labels[pt])
    psd_dict[pt] = psd_lam
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
ax.legend()
fig.savefig(datadir/f"{model}_xd_{op}.png",dpi=300)
fig.savefig(datadir/f"{model}_xd_{op}.pdf")

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
figsp.savefig(datadir/f"{model}_errspectra_{op}.png",dpi=300)
figsp.savefig(datadir/f"{model}_errspectra_{op}.pdf")
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
        fig.savefig(datadir/"{}_errspectra_lam_t-test_{}_{}-{}.png".format(model,op,m1,m2),dpi=300)
        #plt.show()
        plt.close()
