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

Lx = np.pi
nx = 100
dx = Lx / float(nx)
ix = np.linspace(0.5*dx,Lx-0.5*dx,nx)
nmc_dct = NMC_tools(ix,ttype='c')#,cyclic=False)
nmc_dft = NMC_tools(ix,ttype='f')#,cyclic=False,detrend=False)
nmc_dft_detrend = NMC_tools(ix,ttype='f',cyclic=False)

rng = default_rng()
slope = -3.0
a0 = 1.0
kmax = float(ix.size)
ntest = 100
nwaves = 30

#slopelist = [0.0,-1.0,-2.0]
slopelist = [-3.0,-4.0,-5.0]
fig, axs = plt.subplots(ncols=2,figsize=[10,6],constrained_layout=True)
cmap = plt.get_cmap('tab10')
for i,slope in enumerate(slopelist):
    hxm = np.zeros(ix.size)
    for itest in range(ntest):
        hx = np.zeros(ix.size)
        ks = ((rng.random(size=nwaves)+1.0/kmax)*kmax).tolist()
        ks = sorted(ks)
        phis = rng.random(size=nwaves)*np.pi
        #phis = np.zeros(nwaves)
        #print(f"test:{itest} k={ks}")
        elist = []
        for n in range(len(ks)):
            k = ks[n]
            phi = phis[n]
            loge = np.log(a0) + slope*np.log(k)
            e = np.exp(loge)
            a = np.sqrt( Lx * e ) # * 2.0
            hx = hx + a*np.cos(k*ix + phi)
            elist.append(e)
        hx = hx - np.mean(hx)
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
        hxm = hxm + hx
        #axs[0].plot(ix,hx,c=cmap(i),alpha=0.3,lw=0.3)
        #axs[1].plot(ks,elist,c='k',alpha=0.3,lw=0.3,zorder=0)
    sp1m = sp1m / float(ntest)
    sp2m = sp2m / float(ntest)
    sp3m = sp3m / float(ntest)
    hxm = hxm / float(ntest)
    axs[0].plot(ix,hxm,label=f'{slope:.0f}')
    if Lx==np.pi:
        axs[0].set_xticks([0.0,np.pi/6,np.pi/3,np.pi/2,2*np.pi/3,5*np.pi/6,np.pi])
        axs[0].xaxis.set_major_locator(FixedLocator([0.0,np.pi/6,np.pi/3,np.pi/2,2*np.pi/3,5*np.pi/6,np.pi]))
        axs[0].xaxis.set_major_formatter(FixedFormatter(['0',r'$\frac{\pi}{6}$',r'$\frac{\pi}{3}$',r'$\frac{\pi}{2}$',r'$\frac{2\pi}{3}$',r'$\frac{5\pi}{6}$',r'$\pi$']))
    elif Lx==2.0*np.pi:
        axs[0].set_xticks([0.0,np.pi/3,2*np.pi/3,np.pi,4*np.pi/3,5*np.pi/3,2*np.pi])
        axs[0].xaxis.set_major_locator(FixedLocator([0.0,np.pi/3,2*np.pi/3,np.pi,4*np.pi/3,5*np.pi/3,2*np.pi]))
        axs[0].xaxis.set_major_formatter(FixedFormatter(['0',r'$\frac{\pi}{3}$',r'$\frac{2\pi}{3}$',r'$\pi$',r'$\frac{4\pi}{3}$',r'$\frac{5\pi}{3}$',r'$2\pi$']))
    axs[0].grid()
    #axs[0].set_title(r'$\sum a_p\cos(px+\phi)$')
    kref = np.linspace(1,50,49)
    eref = np.exp(np.log(a0) + slope*np.log(kref))
    p0,=axs[1].plot(kref,eref,c='k',zorder=0)
    axs[1].annotate(f'{slope:.0f}',xy=(5e1,eref[-1]),xycoords='data',\
        xytext=(10,0), textcoords='offset points',fontsize=12,color=cmap(i))
    #print(wnum1)
    #print(wnum2)
    p1,=axs[1].loglog(wnum1,sp1m,marker='o',c=cmap(i))
    p2,=axs[1].loglog(wnum2,sp2m,c=cmap(i))
    p3,=axs[1].loglog(wnum3,sp3m,ls='dotted',c=cmap(i))
    if slope == slopelist[0]:
        plist = [p0,p1,p2,p3]
        llist = ['REF','DCT','DFT','DFT, detrend']
axs[0].legend()
if slopelist[-1] > -3:
    axs[1].set_ylim(1e-4,1e1)
else:
    axs[1].set_ylim(1e-9,1e0)
axs[1].set_xlim(1e0,5e1)
axs[1].legend(plist,llist,loc='lower left')
axs[1].set_title(f'slope={",".join([f"{s:.0f}" for s in slopelist])} ntest={ntest}')
fig.savefig(figdir/f'slope{"_".join([f"{s:.0f}" for s in slopelist])}.png')
plt.show()