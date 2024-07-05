import numpy as np 
import sys
sys.path.append('../model')
from lorenz2 import L05II
from lorenz3 import L05III
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FixedFormatter
plt.rcParams['font.size'] = 16
from scipy.interpolate import interp1d
import scipy.fft as fft
sys.path.append('../plot')
from nmc_tools import psd, wnum2wlen, wlen2wnum
from pathlib import Path

klist = [64,32,16,8,4,2]
figdir = Path(f'lorenz05_test')
if not figdir.exists(): figdir.mkdir()
fig2, axs2 = plt.subplots(nrows=2,ncols=len(klist)//2,figsize=[12,10],\
    sharey=True,constrained_layout=True)
fig3, axs3 = plt.subplots(nrows=2,ncols=len(klist)//2,figsize=[12,10],\
    sharey=True,constrained_layout=True)
cmap = plt.get_cmap('coolwarm')
cmap2 = plt.get_cmap('tab10')
figsp, axsp = plt.subplots(figsize=[10,6],constrained_layout=True)
figen, axen = plt.subplots(figsize=[10,6],constrained_layout=True)

for k, nk in enumerate(klist):
    nx_2 = 240
    nk_2 = nk
    F = 15.0
    dt = 0.05 / 36
    step2 = L05II(nx_2,nk_2,dt,F)

    nx_3 = 960
    nk_3 = nk * int(nx_3/nx_2)
    ni_3 = 12
    b = 10.0
    c = 0.6
    step3 = L05III(nx_3,nk_3,ni_3,b,c,dt,F)
    
    ix_2 = np.arange(nx_2) / nx_2 * nx_3
    ix_3 = np.arange(nx_3)

    ## spinup for 10 days
    x0_3 = np.random.randn(nx_3)
    for i in range(1440):
        x0_3 = step3(x0_3)
    l3to2 = interp1d(ix_3,x0_3)
    x0_2 = l3to2(ix_2)

    ## time development for 30 days
    t = []
    ### state
    x_3 = []
    x_2 = []
    ### energy
    e_3 = []
    e_2 = []
    ### spectrum
    ix_3_rad = ix_3 * 2.0 * np.pi / nx_3
    ix_2_rad = ix_2 * 2.0 * np.pi / nx_3
    sp_3 = []
    sp_2 = []
    Lx = 2.0 * np.pi
    dx_3 = Lx / nx_3
    dx_2 = Lx / nx_2
    for i in range(30*4):
        for j in range(36):
            x0_2 = step2(x0_2)
            x0_3 = step3(x0_3)
        #if i%36==0:
        t.append(i/4.)
        x_2.append(x0_2)
        x_3.append(x0_3)
        e_2.append(np.mean(x0_2**2)/2.)
        e_3.append(np.mean(x0_3**2)/2.)
        # fft
        #y0_t = fft.rfft(x0_t)
        #sp0_t = np.abs(y0_t)**2
        #sp0_t *= dx_t**2 / Lx_t
        wnum_2, sp0_2 = psd(x0_2,ix_2_rad)
        wnum_3, sp0_3 = psd(x0_3,ix_3_rad)
        sp_2.append(sp0_2)
        sp_3.append(sp0_3)

    ## hovmöller
    t = np.array(t)
    x_2 = np.array(x_2)
    x_3 = np.array(x_3)
    print("II,  nk={:d}, mean={:.3f}".format(nk_2,x_2.mean()))
    print("III, nk={:d}, mean={:.3f}".format(nk_3,x_3.mean()))
    ax = axs2.flatten()[k]
    mp0 = ax.pcolormesh(ix_2,t,x_2, shading='auto',\
        cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
    fig2.colorbar(mp0,ax=ax,pad=0.01,shrink=0.6)
    ax.set_xticks(ix_2[::(nx_2//8)])
    ax.set_yticks(t[::12])
    ax.set_yticks(t[::4],minor=True)
    ax.set_xlabel('site')
    if k==0: ax.set_ylabel('day')
    ax.set_title(f'nk={nk_2:d}')

    ax = axs3.flatten()[k]
    mp0 = ax.pcolormesh(ix_3,t,x_3, shading='auto',\
        cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
    fig3.colorbar(mp0,ax=ax,pad=0.01,shrink=0.6)
    ax.set_xticks(ix_3[::(nx_3//8)])
    ax.set_yticks(t[::12])
    ax.set_yticks(t[::4],minor=True)
    ax.set_xlabel('site')
    if k==0: ax.set_ylabel('day')
    ax.set_title(f'nk={nk_3:d}')

    ## power spectrum
    sp_2 = np.array(sp_2).mean(axis=0)
    sp_3 = np.array(sp_3).mean(axis=0)
    #wnum_t = fft.rfftfreq(x0_t.size,d=dx_t) * 2.0 * np.pi
    #width = 0.8*(wnum_t[1]-wnum_t[0])
    #ax2.bar(wnum_t-(1.0-k)*width,sp_t,width=width,alpha=0.5,label=f'b={b:.1f} c={c:.1f}')
    axsp.plot(wnum_2,sp_2,c=cmap2(k),label=f'2,nk={nk_2:d}')
    axsp.plot(wnum_3,sp_3,c=cmap2(k),alpha=0.5,label=f'3,nk={nk_3:d}')

    ## energy
    axen.plot(t,e_2,c=cmap2(k),label=f'2,nk={nk_2:d}')
    axen.plot(t,e_3,c=cmap2(k),alpha=0.5,label=f'3,nk={nk_3:d}')

fig2.suptitle("Lorenz II")
fig2.savefig(figdir/f'test_l05II_nk.png',dpi=300)
fig3.suptitle("Lorenz III")
fig3.savefig(figdir/f'test_l05III_nk.png',dpi=300)

axsp.set_title("time averaged power spectrum")
axsp.set(xlabel=r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)",title='variance power spectra')
#axsp.set_xscale('log')
#axsp.xaxis.set_major_locator(FixedLocator([240./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi]))
#axsp.xaxis.set_major_formatter(FixedFormatter([r'$\frac{240}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$']))
axsp.xaxis.set_major_locator(FixedLocator([480,240,120,60,30,1]))
axsp.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','1']))
#axsp.set_xlim(0.5/np.pi,wnum[-1])
secax = axsp.secondary_xaxis('top',functions=(wnum2wlen, wlen2wnum))
secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
axsp.set_yscale("log")
axsp.legend(loc='upper left',bbox_to_anchor=(1.01,1.0))
figsp.savefig(figdir/f'test_psd_nk.png',dpi=300)

axen.set_title("spatial averaged energy")
axen.set(xlabel="time",ylabel="energy")
axen.legend(loc='upper left',bbox_to_anchor=(1.01,1.0))
figen.savefig(figdir/f'test_en_nk.png',dpi=300)
plt.show()