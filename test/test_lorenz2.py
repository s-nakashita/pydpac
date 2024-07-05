import numpy as np 
import sys
sys.path.append('../model')
from lorenz2 import L05II
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
figdir = Path(f'lorenz2_test')
if not figdir.exists(): figdir.mkdir()
fig, axs = plt.subplots(nrows=2,ncols=len(klist)//2,figsize=[12,10],\
    sharey=True,constrained_layout=True)
cmap = plt.get_cmap('coolwarm')
fig2, ax2 = plt.subplots(figsize=[10,6],constrained_layout=True)
fig3, ax3 = plt.subplots(figsize=[10,6],constrained_layout=True)

for k, nk in enumerate(klist):
    nx_true = 240
    nk_true = nk
    F_true = 15.0
    dt_true = 0.05 / 36
    nature_step = L05II(nx_true,nk_true,dt_true,F_true)
    ix_t = np.arange(nx_true)

    ## spinup for 10 days
    x0_t = np.random.randn(nx_true)
    for i in range(1440):
        x0_t = nature_step(x0_t)

    ## time development for 30 days
    t = []
    ### state
    x_t = []
    ### energy
    e_t = []
    ### spectrum
    ix_t_rad = ix_t * 2.0 * np.pi / nx_true
    sp_t = []
    Lx_t = 2.0 * np.pi
    dx_t = Lx_t / nx_true
    for i in range(30*4):
        for j in range(36):
            x0_t = nature_step(x0_t)
        #if i%36==0:
        t.append(i/4.)
        x_t.append(x0_t)
        e_t.append(np.mean(x0_t**2/2.))
        # fft
        #y0_t = fft.rfft(x0_t)
        #sp0_t = np.abs(y0_t)**2
        #sp0_t *= dx_t**2 / Lx_t
        wnum_t, sp0_t = psd(x0_t,ix_t_rad)
        sp_t.append(sp0_t)

    ## hovmöller
    t = np.array(t)
    x_t = np.array(x_t)
    print("nk={:d}, mean={:.3f}".format(nk,x_t.mean()))
    ax = axs.flatten()[k]
    mp0 = ax.pcolormesh(ix_t,t,x_t, shading='auto',\
        cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
    fig.colorbar(mp0,ax=ax,pad=0.01,shrink=0.6)
    ax.set_xticks(ix_t[::(nx_true//8)])
    ax.set_yticks(t[::12])
    ax.set_yticks(t[::4],minor=True)
    ax.set_xlabel('site')
    if k==0: ax.set_ylabel('day')
    ax.set_title(f'nk={nk:d}')

    ## power spectrum
    sp_t = np.array(sp_t).mean(axis=0)
    #wnum_t = fft.rfftfreq(x0_t.size,d=dx_t) * 2.0 * np.pi
    width = 0.8*(wnum_t[1]-wnum_t[0])
    #ax2.bar(wnum_t-(1.0-k)*width,sp_t,width=width,alpha=0.5,label=f'b={b:.1f} c={c:.1f}')
    ax2.plot(wnum_t,sp_t,label=f'nk={nk:d}')

    ## energy
    ax3.plot(t,e_t,label=f'nk={nk:d}')

fig.savefig(figdir/f'test_nature_nk.png',dpi=300)

ax2.set_title("time averaged power spectrum")
ax2.set(xlabel=r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)",title='variance power spectra')
#ax2.set_xscale('log')
#ax2.xaxis.set_major_locator(FixedLocator([240./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi]))
#ax2.xaxis.set_major_formatter(FixedFormatter([r'$\frac{240}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$']))
ax2.xaxis.set_major_locator(FixedLocator([480,240,120,60,30,1]))
ax2.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','1']))
#ax2.set_xlim(0.5/np.pi,wnum[-1])
secax = ax2.secondary_xaxis('top',functions=(wnum2wlen, wlen2wnum))
secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
ax2.set_yscale("log")
ax2.legend(loc='upper left',bbox_to_anchor=(1.01,1.0))
fig2.savefig(figdir/f'test_nature_psd_nk.png',dpi=300)

ax3.set_title("spatial averaged energy")
ax3.set(xlabel="time",ylabel="energy")
ax3.legend(loc='upper left',bbox_to_anchor=(1.01,1.0))
fig3.savefig(figdir/f'test_nature_en_nk.png',dpi=300)
plt.show()