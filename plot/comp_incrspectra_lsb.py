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
pt = 'var' #var or envar
if len(sys.argv)>4:
    pt = sys.argv[4]

t = np.arange(na)+1
ns = 40 # spinup

datadir = Path(f'/Volumes/FF520/nested_envar/data/{model}')
datadir = Path(f'../work/{model}')
preGMpt = 'envar'
ldscl=True
obsloc = ''
if len(sys.argv)>5:
    obsloc = sys.argv[5]
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

fig, (axb,axa,axi) = plt.subplots(nrows=3,sharex=True,figsize=[8,8],constrained_layout=True)

## GM
#fa = dscldir/"xagm_{}_{}.npy".format(op,preGMpt)
#fb = dscldir/"{}_xfgm_{}_{}.npy".format(model,op,preGMpt)
#if not fa.exists():
#    print("not exist {}".format(fa))
#    exit()
#if not fb.exists():
#    print("not exist {}".format(fb))
#    exit()
#xagm = np.load(fa)
#xbgm = np.load(fb)
#incrgm = xagm[ns:na,:] - xbgm[ns:na,:]
#wnum_gm, psd_gm = nmc_gm.psd(xbgm,axis=1)
#axb.loglog(wnum_gm,psd_gm,c='gray',lw=4.0,label='GM')
#wnum_gm, psd_gm = nmc_gm.psd(xagm,axis=1)
#axa.loglog(wnum_gm,psd_gm,c='gray',lw=4.0,label='GM')
#wnum_gm, psd_gm = nmc_gm.psd(incrgm,axis=1)
#axi.loglog(wnum_gm,psd_gm,c='gray',lw=4.0,label='GM')

if ldscl:
    # downscaling
    fa = dscldir/"xalam_{}_{}.npy".format(op,preGMpt)
    fb = dscldir/"{}_xflam_{}_{}.npy".format(model,op,preGMpt)
    if not fa.exists():
        print("not exist {}".format(fa))
        exit()
    xadscl = np.load(fa)
    xbdscl = np.load(fb)
    incrdscl = xadscl[ns:na,:] - xbdscl[ns:na,:]
    wnum, psd_dscl = nmc_lam.psd(xbdscl,axis=1)
    axb.loglog(wnum,psd_dscl,c='k',lw=2.0,label='Downscaling')
    wnum, psd_dscl = nmc_lam.psd(xadscl,axis=1)
    axa.loglog(wnum,psd_dscl,c='k',lw=2.0,label='Downscaling')
    wnum, psd_dscl = nmc_lam.psd(incrdscl,axis=1)
    axi.loglog(wnum,psd_dscl,c='k',lw=2.0,label='Downscaling')
    #f = dscldir/"xsalam_{}_{}.npy".format(op,preGMpt)
    #xsadscl = np.load(f)
    #ax.plot(ix_lam, np.mean(xsadscl,axis=0),\
    #    c='k',ls='dashed')
# LAM
for key in ['conv','lsb','nest']:
    if key=='conv':
        fa = lamdir/"xalam_{}_{}.npy".format(op,pt)
        fb = lamdir/"xflam_{}_{}.npy".format(op,pt)
    elif key=='nest':
        fa = lamdir/"xalam_{}_{}_nest.npy".format(op,pt)
        fb = lamdir/"xflam_{}_{}_nest.npy".format(op,pt)
    else:
        fa = lsbdir/"xalam_{}_{}.npy".format(op,pt)
        fb = lsbdir/"xflam_{}_{}.npy".format(op,pt)
    if not fa.exists():
        print("not exist {}".format(fa))
        exit()
    xalam = np.load(fa)
    xblam = np.load(fb)
    incrlam = xalam[ns:na,:] - xblam[ns:na,:]
    wnum, psd_lam = nmc_lam.psd(xblam,axis=1)
    axb.loglog(wnum,psd_lam,c=linecolor[key],lw=2.0,label=labels[key])
    wnum, psd_lam = nmc_lam.psd(xalam,axis=1)
    axa.loglog(wnum,psd_lam,c=linecolor[key],lw=2.0,label=labels[key])
    wnum, psd_lam = nmc_lam.psd(incrlam,axis=1)
    axi.loglog(wnum,psd_lam,c=linecolor[key],lw=2.0,label=labels[key])
    #f = lamdir/"xsalam_{}_{}.npy".format(op,pt)
    #xsalam = np.load(f)
    #if pt == 'var' or pt == 'var_nest':
    #    ax.plot(ix_lam[1:-1], np.mean(xsalam,axis=0),\
    #    c=linecolor[pt],ls='dashed')
    #else:
    #    ax.plot(ix_lam, np.mean(xsalam,axis=0),\
    #    c=linecolor[pt],ls='dashed')

for i,ax in enumerate([axb,axa,axi]):
    ax.grid()
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
ybmin, ybmax = axb.get_ylim()
yamin, yamax = axa.get_ylim()
ymin = min(ybmin,yamin)
ymax = max(ybmax,yamax)
axb.set_ylim(ymin,ymax)
axa.set_ylim(ymin,ymax)
axb.legend(loc='upper left',bbox_to_anchor=(1.0,1.0))
axb.set_ylabel('first guess')
axa.set_ylabel('analysis')
axi.set_ylabel('increment')
fig.suptitle(ptlong[pt])
fig.savefig(figdir/f"{model}_incrspectra_{op}_{pt}.png",dpi=300)
fig.savefig(figdir/f"{model}_incrspectra_{op}_{pt}.pdf")
plt.show()
plt.close()
