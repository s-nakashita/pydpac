import sys
import os
import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from nmc_tools import NMC_tools, wnum2wlen, wlen2wnum
from pathlib import Path

plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 14

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = 'envar'

ns = 40 # spinup
tc = np.arange(ns,na)+1
t = tc / 4.
ntime = na - ns
nt1 = ntime // 3

datadir = Path(f'/Volumes/FF520/nested_envar/data/{model}')
datadir = Path(f'../work/{model}')
preGMpt = 'envar'
ldscl=True
obsloc = ''
if len(sys.argv)>4:
    obsloc = sys.argv[4]
#obsloc = '_partiall'
#obsloc = '_partialc'
#obsloc = '_partialr'
#obsloc = '_partialm'
dscldir = datadir / 'var_vs_envar_dscl_m80obs30'
lsbdir  = datadir / f'var_vs_envar_lsb_preGM{obsloc}_m80obs30'
lamdir  = datadir / f'var_vs_envar_shrink_dct_preGM{obsloc}_m80obs30'
#if ldscl:
#    figdir = datadir
#else:
figdir = lsbdir

ptlong = {"envar":"EnVar","var":"3DVar"}
labels = {"dscl":"Downscaling","conv":"DA", "lsb":"DA+LSB", "nest":"Nested DA"}
linecolor = {"dscl":"k","conv":"tab:blue","lsb":'tab:orange',"nest":'tab:green'}

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
nmc_lam = NMC_tools(ix_lam_rad[1:-1],cyclic=False,ttype='c')

f = dscldir/"truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[:na,:]
xt2x = interp1d(ix_t,xt)

psd_mse = {}
psd_sprd = {}
if ldscl:
    keys=['dscl','conv','lsb','nest']
else:
    keys=['conv','lsb','nest']
for key in keys:
    psde = []
    psds = []
    for i in range(ns,na):
        if key=='dscl':
            f = dscldir/"data/{2}/{0}_gm_ua_{1}_{2}_cycle{3}.npy".format(model,op,preGMpt,i)
            xagm = np.load(f)
            gm2lam = interp1d(ix_gm,xagm,axis=0)
            xalam = gm2lam(ix_lam[1:-1])
        else:
            if key=='conv':
                f = lamdir/"data/{2}/{0}_lam_ua_{1}_{2}_cycle{3}.npy".format(model,op,pt,i)
            elif key=='nest':
                f = lamdir/"data/{2}_nest/{0}_lam_ua_{1}_{2}_nest_cycle{3}.npy".format(model,op,pt,i)
            else:
                f = lsbdir/"data/{2}/{0}_lam_ua_{1}_{2}_cycle{3}.npy".format(model,op,pt,i)
            xalam = np.load(f)
        xdlam = xt2x(ix_lam[1:-1])[i,] - np.mean(xalam,axis=1)
        wnum_e, psd1 = nmc_lam.psd(xdlam,average=False)
        psde.append(psd1)
        xalam = xalam - np.mean(xalam,axis=1)[:,None]
        wnum_s, psd1 = nmc_lam.psd(xalam,axis=0,average=False)
        psds.append(np.sum(psd1,axis=1)/(psd1.shape[1]-1))
    psd_mse[key] = np.array(psde).mean(axis=0)
    psd_sprd[key] = np.array(psds).mean(axis=0)

fig, (axe,axs,axr) = plt.subplots(nrows=3,sharex=True,figsize=[8,8],constrained_layout=True)

for key in psd_sprd.keys():
    axe.loglog(wnum_e,psd_mse[key],c=linecolor[key],lw=2.0,label=labels[key])
    axs.loglog(wnum_s,psd_sprd[key],c=linecolor[key],lw=2.0,label=labels[key])
    psd_r = psd_mse[key] / psd_sprd[key]
    axr.loglog(wnum_e,psd_r,c=linecolor[key],lw=2.0,label=labels[key])
for i,ax in enumerate([axe,axs,axr]):
    ax.grid()
    ax.vlines([24],0,1,colors='magenta',ls='dashed',zorder=0,transform=ax.get_xaxis_transform())
    #ax.xaxis.set_major_locator(FixedLocator([180./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi,0.5/np.pi]))
    #ax.xaxis.set_major_formatter(FixedFormatter([r'$\frac{180}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$',r'$\frac{1}{2\pi}$']))
    ax.xaxis.set_major_locator(FixedLocator([480,240,120,60,30,12,2]))
    secax = ax.secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
    secax.xaxis.set_major_locator(FixedLocator([np.pi,np.pi/6.,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
    if i==2:
        ax.set_xlabel(r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)")
        ax.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','12','2']))
    else:
        ax.xaxis.set_major_formatter(NullFormatter())
    if i==0:
        secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
        secax.xaxis.set_major_formatter(FixedFormatter([r'$\pi$',r'$\frac{\pi}{6}$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
    else:
        secax.xaxis.set_major_formatter(NullFormatter())
#ymin1, ymax1 = axe.get_ylim()
#ymin2, ymax2 = axs.get_ylim()
#ymin = min(ymin1,ymin2)
#ymax = max(ymax1,ymax2)
#axe.set_ylim(ymin,ymax)
#axs.set_ylim(ymin,ymax)
axe.legend(loc='upper left',bbox_to_anchor=(1.0,1.0))
axe.set_ylabel('error')
axs.set_ylabel('spread')
axr.set_ylabel('error/spread')
axr.set_ylim(1e-1,1e1)
axr.hlines([1],0,1,colors='r',ls='dashed',lw=1,transform=ax.get_yaxis_transform())
fig.suptitle(ptlong[pt])
fig.savefig(figdir/f"{model}_err+sprdspectra_{op}_{pt}.png",dpi=300)
fig.savefig(figdir/f"{model}_err+sprdspectra_{op}_{pt}.pdf")
plt.show() #block=False)
plt.close()