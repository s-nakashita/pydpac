import numpy as np 
import sys
sys.path.append('../model')
from lorenz2 import L05II
from lorenz3 import L05III
from lorenz_nest import L05nest
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

## nature
nx_true = 960
nk_true = 32
ni_true = 12
b_true = 10.0
c_true = 0.6
F_true = 15.0
dt_true = 0.05 / 36
nature_step = L05III(nx_true,nk_true,ni_true,b_true,c_true,dt_true,F_true)
ix_t = np.arange(nx_true)
## GM (low-res nature)
intgm = 4
nx_gm = nx_true // intgm
nk_gm = nk_true // intgm
ni_gm = ni_true // intgm
dt_gm = dt_true * intgm
gm_step = L05III(nx_gm,nk_gm,ni_gm,b_true,c_true,dt_gm,F_true)
ix_gm = np.arange(nx_gm) * intgm
## GM (LorenzII)
gm2_step = L05II(nx_gm,nk_gm,dt_gm,F_true)
figdir = Path(f'nature_test')
if not figdir.exists(): figdir.mkdir()
## spinup for 10 days
x0_t = np.random.randn(nx_true)
for i in range(1440):
    x0_t = nature_step(x0_t)

## initialize GM
true2exp = interp1d(ix_t, x0_t)
x0_gm = true2exp(ix_gm)
x0_gm2 = x0_gm.copy()

## time development for 30 days
t = []
### state
x_t = []
x_gm = []
x_gm2 = []
### spectrum
ix_t_rad = ix_t * 2.0 * np.pi / nx_true
ix_gm_rad = ix_gm * 2.0 * np.pi / nx_true
sp_t = []
sp_gm = []
sp_gm2 = []
Lx_t = 2.0 * np.pi
Lx_gm = 2.0 * np.pi
dx_t = Lx_t / nx_true
dx_gm = Lx_gm / nx_gm
for i in range(30*4):
    for j in range(36):
        x0_t = nature_step(x0_t)
    for j in range(36//intgm):
        x0_gm = gm_step(x0_gm)
        x0_gm2 = gm2_step(x0_gm2)
    #if i%36==0:
    t.append(i/4.)
    x_t.append(x0_t)
    x_gm.append(x0_gm)
    x_gm2.append(x0_gm2)
    # fft
    #y0_t = fft.rfft(x0_t)
    #sp0_t = np.abs(y0_t)**2
    #sp0_t *= dx_t**2 / Lx_t
    wnum_t, sp0_t = psd(x0_t,ix_t_rad)
    sp_t.append(sp0_t)
    #y0_gm = fft.rfft(x0_gm)
    #sp0_gm = np.abs(y0_gm)**2
    #sp0_gm *= dx_gm**2 / Lx_gm
    wnum_gm, sp0_gm = psd(x0_gm,ix_gm_rad)
    sp_gm.append(sp0_gm)
    wnum_gm, sp0_gm2 = psd(x0_gm2,ix_gm_rad)
    sp_gm2.append(sp0_gm2)
    if i%20==0:
        plt.plot(ix_t,x0_t,label='nature')
        plt.plot(ix_gm,x0_gm,label='GM (same model)')
        plt.plot(ix_gm,x0_gm2,label='GM')
        plt.title(f"t={i/4.:.1f}d")
        plt.legend()
        plt.savefig(figdir/f'test_nature_intgm{intgm}_t{int(i/4)}.png',dpi=300)
        plt.show(block=False)
        plt.close()
        #wnum_t = fft.rfftfreq(x0_t.size,d=dx_t) * 2.0 * np.pi
        plt.plot(wnum_t,sp0_t,label='nature')
        #wnum_gm = fft.rfftfreq(x0_gm.size,d=dx_gm) * 2.0 * np.pi
        plt.plot(wnum_gm,sp0_gm,label='GM (same model)')
        plt.plot(wnum_gm,sp0_gm2,label='GM')
        plt.title(f"t={i/4.:.1f}d, power spectrum")
        plt.xlabel("wave number")
        plt.xscale("log")
        plt.yscale("log")
        plt.legend()
        plt.savefig(figdir/f'test_nature_intgm{intgm}_psd_t{int(i/4)}.png',dpi=300)
        plt.show(block=False)
        plt.close()
plt.plot(ix_t,x0_t,label='nature')
plt.plot(ix_gm,x0_gm,label='GM (same model)')
plt.plot(ix_gm,x0_gm2,label='GM')
plt.title(f"t={i/4.:.1f}d")
plt.legend()
plt.savefig(figdir/f'test_nature_intgm{intgm}_t{int(i/4.)}.png',dpi=300)
plt.show()
plt.close()
#exit()

fig, axs = plt.subplots(ncols=3,figsize=[12,5],\
    sharey=True,constrained_layout=True)
cmap = plt.get_cmap('coolwarm')
#nature
t = np.array(t)
x_t = np.array(x_t)
print("true           mean={:.3f}".format(x_t.mean()))
mp0 = axs[0].pcolormesh(ix_t,t,x_t, shading='auto',\
    cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
axs[0].set_xticks(ix_t[::120])
axs[0].set_yticks(t[::12])
axs[0].set_yticks(t[::4],minor=True)
axs[0].set_xlabel('site')
axs[0].set_ylabel('day')
axs[0].set_title('nature')
#GM
x_gm = np.array(x_gm)
print("GM(same model) mean={:.3f}".format(x_gm.mean()))
mp1 = axs[1].pcolormesh(ix_gm,t,x_gm, shading='auto',\
    cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
axs[1].set_xticks(ix_gm[::120//intgm])
axs[1].set_xlabel('site')
axs[1].set_title('GM (same model)')
#GM
x_gm2 = np.array(x_gm2)
print("GM             mean={:.3f}".format(x_gm2.mean()))
mp2 = axs[2].pcolormesh(ix_gm,t,x_gm2, shading='auto',\
    cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
axs[2].set_xticks(ix_gm[::120//intgm])
axs[2].set_xlabel('site')
axs[2].set_title('GM')
fig.colorbar(mp2,ax=axs[2],pad=0.01,shrink=0.6)
fig.savefig(figdir/f'test_nature_intgm{intgm}.png',dpi=300)
plt.show()
plt.close()

fig, axs = plt.subplots(ncols=3,figsize=[12,5],\
    sharey=True,constrained_layout=True)
cmap = plt.get_cmap('coolwarm')
#nature
mp0 = axs[0].pcolormesh(ix_t,t,x_t, shading='auto',\
    cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
axs[0].set_xticks(ix_t[::120])
axs[0].set_yticks(t[::12])
axs[0].set_yticks(t[::4],minor=True)
axs[0].set_xlabel('site')
axs[0].set_ylabel('day')
axs[0].set_title('nature')
#GM
gm2true = interp1d(ix_gm, x_gm, fill_value="extrapolate")
xgm2t = gm2true(ix_t)
xd = xgm2t - x_t
mp1 = axs[1].pcolormesh(ix_t,t,xd, shading='auto',\
    cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
axs[1].set_xticks(ix_t[::120])
axs[1].set_xlabel('site')
axs[1].set_title('GM (same model) error')
#GM
gm2true = interp1d(ix_gm, x_gm2, fill_value="extrapolate")
xgm2t = gm2true(ix_t)
xd = xgm2t - x_t
mp2 = axs[2].pcolormesh(ix_t,t,xd, shading='auto',\
    cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
axs[2].set_xticks(ix_t[::120])
axs[2].set_xlabel('site')
axs[2].set_title('GM error')
fig.colorbar(mp2,ax=axs[2],pad=0.01,shrink=0.6)
fig.savefig(figdir/f'test_nature_err_intgm{intgm}.png',dpi=300)
plt.show()
plt.close()

## power spectrum
sp_t = np.array(sp_t).mean(axis=0)
sp_gm = np.array(sp_gm).mean(axis=0)
sp_gm2 = np.array(sp_gm2).mean(axis=0)
fig, ax = plt.subplots(figsize=[8,6],constrained_layout=True)
#wnum_t = fft.rfftfreq(x0_t.size,d=dx_t) * 2.0 * np.pi
ax.plot(wnum_t,sp_t,label='nature')
#wnum_gm = fft.rfftfreq(x0_gm.size,d=dx_gm) * 2.0 * np.pi
ax.plot(wnum_gm,sp_gm,label='GM (same model)')
ax.plot(wnum_gm,sp_gm2,label='GM')
ax.set_title("time averaged power spectrum")
ax.set(xlabel=r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)",title='variance power spectra')
ax.set_xscale('log')
#ax.xaxis.set_major_locator(FixedLocator([240./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi]))
#ax.xaxis.set_major_formatter(FixedFormatter([r'$\frac{240}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$']))
ax.xaxis.set_major_locator(FixedLocator([480,240,120,60,8,2,1]))
ax.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','8','2','1']))
#ax.set_xlim(0.5/np.pi,wnum[-1])
secax = ax.secondary_xaxis('top',functions=(wnum2wlen, wlen2wnum))
secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi,np.pi/4,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\pi$',r'$\frac{\pi}{4}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
ax.set_yscale("log")
ax.legend()
fig.savefig(figdir/f'test_nature_psd_intgm{intgm}.png',dpi=300)
plt.show()