import numpy as np 
from numpy.random import default_rng
import matplotlib.pyplot as plt 
from matplotlib.ticker import FixedLocator, FixedFormatter
import sys 
sys.path.append('../plot')
from nmc_tools import NMC_tools
from pathlib import Path

figdir = Path('1dspectrum')
if not figdir.exists(): figdir.mkdir()

dx = 2.0*np.pi / 100.0
ix = np.linspace(0.5*dx,2.0*np.pi-0.5*dx,100)
nmc_dct = NMC_tools(ix,cyclic=False,ttype='c')
nmc_dft = NMC_tools(ix,cyclic=False,ttype='f',detrend=False)
nmc_dft_detrend = NMC_tools(ix,cyclic=False,ttype='f')

rng = default_rng(517)
slope = -3.0
a0 = 1.0
kmax = float(ix.size)
ntest = 100
nwaves = 10

slopelist = [0.0,-1.0,-2.0]
#slopelist = [-3.0,-4.0,-5.0]
fig, ax = plt.subplots(figsize=[6,6],constrained_layout=True)
cmap = plt.get_cmap('tab10')
for i,slope in enumerate(slopelist):
    for itest in range(ntest):
        hx = np.zeros(ix.size)
        ks = [1] + (rng.random(size=nwaves)*kmax).tolist()
        ks = sorted(ks)
        phis = rng.random(size=nwaves+1)*np.pi
        #print(f"test:{itest} k={ks}")
        elist = []
        for n in range(len(ks)):
            k = ks[n]
            phi = phis[n]
            loge = np.log(a0) + slope*np.log(k)
            e = np.exp(loge)
            a = np.sqrt( np.pi * e )
            hx = hx + a*np.cos(k*ix + phi)
            elist.append(e)
        wnum1, sp1 = nmc_dct.psd(hx)
        wnum2, sp2 = nmc_dft.psd(hx)
        wnum3, sp3, _ = nmc_dft_detrend.psd(hx)
        if itest==0:
            sp1m = sp1.copy()
            sp2m = sp2.copy()
            sp3m = sp3.copy()
        else:
            sp1m = sp1m + sp1
            sp2m = sp2m + sp2
            sp3m = sp3m + sp3
    sp1m = sp1m / float(ntest)
    sp2m = sp2m / float(ntest)
    sp3m = sp3m / float(ntest)
    #axs[0].plot(ix,hx)
    #axs[0].set_xticks([0.0,np.pi/6,np.pi/3,np.pi/2,2*np.pi/3,5*np.pi/6,np.pi])
    #axs[0].xaxis.set_major_locator(FixedLocator([0.0,np.pi/6,np.pi/3,np.pi/2,2*np.pi/3,5*np.pi/6,np.pi]))
    #axs[0].xaxis.set_major_formatter(FixedFormatter(['0',r'$\frac{\pi}{6}$',r'$\frac{\pi}{3}$',r'$\frac{\pi}{2}$',r'$\frac{2\pi}{3}$',r'$\frac{5\pi}{6}$',r'$\pi$']))
    #axs[0].grid()
    #axs[0].set_title(r'$\sum a_p\cos(px+\phi)$')
    p0,=ax.plot(ks,elist,c='k',zorder=0)
    ax.annotate(f'{slope:.0f}',xy=(1e2,elist[-1]),xycoords='data',\
        xytext=(10,0), textcoords='offset points',fontsize=12,color=cmap(i))
    #print(wnum1)
    #print(wnum2)
    p1,=ax.loglog(wnum1,sp1m,marker='o',c=cmap(i))
    p2,=ax.loglog(wnum2,sp2m,c=cmap(i))
    p3,=ax.loglog(wnum3,sp3m,ls='dotted',c=cmap(i))
    if slope == slopelist[0]:
        plist = [p0,p1,p2,p3]
        llist = ['REF','DCT','DFT','DFT, detrend']
if slopelist[-1] > -3:
    ax.set_ylim(1e-4,1e1)
#else:
#    ax.set_ylim(1e-9,1e0)
ax.set_xlim(1e0,1e2)
ax.legend(plist,llist,loc='lower left')
ax.set_title(f'slope={",".join([f"{s:.0f}" for s in slopelist])} ntest={ntest}')
fig.savefig(figdir/f'slope{"_".join([f"{s:.0f}" for s in slopelist])}.png')
plt.show()