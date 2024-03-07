import numpy as np 
import sys
sys.path.append('../model')
from lorenz3 import L05III
from lorenz_nest import L05nest
# multiple advection version
from lorenz3m import L05IIIm
from lorenz_nestm import L05nestm

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
nks_true = [256,128,64,32]
ni_true = 12
b_true = 10.0
c_true = 0.6
F_true = 15.0
lamstep = 1
dt_true = 0.05 / 36 / lamstep
#nature_step = L05III(nx_true,nk_true,ni_true,b_true,c_true,dt_true,F_true)
nature_step = L05IIIm(nx_true,nks_true,ni_true,b_true,c_true,dt_true,F_true)
## GM
intgm = 4
nx_gm = nx_true // intgm
nk_gm = nk_true // intgm
nks_gm = np.array(nks_true,dtype=np.int64) // intgm
## LAM
nx_lam = 240
ist_lam = 240
lamstep = 1
nsp = 10
po = 1
intrlx = 1
nk_lam = nk_true
nks_lam = nks_true
ni_lam = ni_true
b_lam = b_true
c_lam = c_true
F = F_true
dt = dt_true * lamstep
#nest_step = L05nest(nx_true, nx_gm, nx_lam, nk_gm, nk_lam, ni_lam, b_lam, c_lam, dt, F, \
#    intgm, ist_lam, nsp, po=po, lamstep=lamstep, intrlx=intrlx)
nest_step = L05nestm(nx_true, nx_gm, nx_lam, nks_gm, nks_lam, ni_lam, b_lam, c_lam, dt, F, \
    intgm, ist_lam, nsp, po=po, lamstep=lamstep, intrlx=intrlx)
#figdir = Path(f'nsp{nsp}p{nest_step.po}intrlx{nest_step.intrlx}')
figdir = Path(f'm{"+".join([str(n) for n in nks_gm])}/nsp{nsp}p{nest_step.po}intrlx{nest_step.intrlx}')
if not figdir.exists(): figdir.mkdir(parents=True)
## spinup for 10 days
x0_t = np.random.randn(nx_true)
for i in range(1440):
    x0_t = nature_step(x0_t)

## initialize GM and LAM
true2exp = interp1d(nest_step.ix_true, x0_t)
x0_gm = true2exp(nest_step.ix_gm)
gm2lam = interp1d(nest_step.ix_gm, x0_gm)
x0_lam = gm2lam(nest_step.ix_lam)

## time development for 30 days
t = []
### state
x_t = []
x_gm = []
x_lam = []
### spectrum
ix_t_rad = nest_step.ix_true * 2.0 * np.pi / nx_true
ix_gm_rad = nest_step.ix_gm * 2.0 * np.pi / nx_true
ix_lam_rad = nest_step.ix_lam * 2.0 * np.pi / nx_true
sp_t = []
sp_gm = []
sp_lam = []
Lx_t = 2.0 * np.pi
Lx_gm = 2.0 * np.pi
nghost = 0 # ghost region for periodicity in LAM
#dwindow = (1.0 + np.cos(np.pi*np.arange(1,nghost+1)/nghost))*0.5
Lx_lam = 2.0 * np.pi * (nx_lam + 2*nghost) / nx_true
dx_t = Lx_t / nx_true
dx_gm = Lx_gm / nx_gm
dx_lam = Lx_lam / (nx_lam + 2*nghost)
for i in range(60*4):
    for j in range(36*lamstep):
        x0_t = nature_step(x0_t)
    x0_gm, x0_lam = nest_step(x0_gm, x0_lam)
    #if i%36==0:
    t.append(i/4.)
    x_t.append(x0_t)
    x_gm.append(x0_gm)
    x_lam.append(x0_lam)
    if i%20==0:
        plt.plot(nest_step.ix_true,x0_t,label='nature')
        plt.plot(nest_step.ix_gm,x0_gm,label='GM')
        plt.plot(nest_step.ix_lam,x0_lam,ls='dashed',label='LAM')
        #plt.plot(ix_lam_ext,x0_lam_ext,ls='dotted')
        plt.title(f"t={i/4.:.1f}d")
        plt.legend()
        plt.savefig(figdir/f'test_nest_t{int(i/4)}.png',dpi=300)
        plt.show(block=False)
        plt.close()
    if i>=30*4:
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
        #x0_lam_ext = np.zeros(nx_lam + 2*nghost)
        #x0_lam_ext[nghost:nghost+nx_lam] = x0_lam[:]
        #x0_lam_ext[0:nghost] = x0_lam[0] * dwindow[::-1]
        #x0_lam_ext[nghost+nx_lam:] = x0_lam[-1] * dwindow
        #ix_lam_ext = nest_step.ix_true[ist_lam-nghost:ist_lam+nx_lam+nghost]
        #y0_lam_ext = fft.rfft(x0_lam_ext)
        #sp0_lam = np.abs(y0_lam_ext[:nx_lam//2+1])**2
        #sp0_lam *= dx_lam**2 / Lx_lam
        wnum_lam, sp0_lam = psd(x0_lam,ix_lam_rad)
        sp_lam.append(sp0_lam)
        if i%20==0:
            #wnum_t = fft.rfftfreq(x0_t.size,d=dx_t) * 2.0 * np.pi
            plt.plot(wnum_t,sp0_t,label='nature')
            #wnum_gm = fft.rfftfreq(x0_gm.size,d=dx_gm) * 2.0 * np.pi
            plt.plot(wnum_gm,sp0_gm,label='GM')
            #wnum_lam = fft.rfftfreq(x0_lam_ext.size,d=dx_lam)[:sp0_lam.size] * 2.0 * np.pi
            plt.plot(wnum_lam,sp0_lam,label='LAM')
            plt.title(f"t={i/4.:.1f}d, power spectrum")
            plt.xlabel("wave number")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            plt.savefig(figdir/f'test_nest_psd_t{int(i/4)}.png',dpi=300)
            plt.show(block=False)
            plt.close()
plt.plot(nest_step.ix_true,x0_t,label='nature')
plt.plot(nest_step.ix_gm,x0_gm,label='GM')
plt.plot(nest_step.ix_lam,x0_lam,ls='dashed',label='LAM')
plt.title(f"t={i/4.:.1f}d")
plt.legend()
plt.savefig(figdir/f'test_nest_lamstep{lamstep}_t{int(i/4.)}.png',dpi=300)
plt.show()
plt.close()
#exit()

fig, axs = plt.subplots(ncols=3,figsize=[12,5],\
    sharey=True,constrained_layout=True)
cmap = plt.get_cmap('coolwarm')
#nature
t = np.array(t)
x_t = np.array(x_t)
print("true mean={:.3f}".format(x_t.mean()))
mp0 = axs[0].pcolormesh(nest_step.ix_true,t,x_t, shading='auto',\
    cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
axs[0].set_xticks(nest_step.ix_true[::120])
axs[0].set_yticks(t[::12])
axs[0].set_yticks(t[::4],minor=True)
axs[0].set_xlabel('site')
axs[0].set_ylabel('day')
axs[0].set_title('nature')
#GM
x_gm = np.array(x_gm)
print("GM   mean={:.3f}".format(x_gm.mean()))
mp1 = axs[1].pcolormesh(nest_step.ix_gm,t,x_gm, shading='auto',\
    cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
axs[1].vlines([nest_step.ix_lam[0],nest_step.ix_lam[-1]],0,1,\
    colors='k',linestyle='dashdot',transform=axs[1].get_xaxis_transform())
axs[1].set_xticks(nest_step.ix_gm[::30])
axs[1].set_xlabel('site')
axs[1].set_title('GM')
#LAM
x_lam = np.array(x_lam)
print("LAM  mean={:.3f}".format(x_lam.mean()))
mp2 = axs[2].pcolormesh(nest_step.ix_lam,t,x_lam, shading='auto',\
    cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
axs[2].set_xlim(nest_step.ix_gm[0],nest_step.ix_gm[-1])
axs[2].set_xticks(nest_step.ix_gm[::30])
axs[2].set_xlabel('site')
axs[2].set_title('LAM')
axs[2].vlines([nest_step.ix_lam[nsp],nest_step.ix_lam[-nsp]],0,1,\
    colors='white',linestyle='dashdot',transform=axs[2].get_xaxis_transform())
fig.colorbar(mp2,ax=axs[2],shrink=0.6,pad=0.01)
fig.savefig(figdir/f'test_nest_lamstep{lamstep}.png',dpi=300)
plt.show()
plt.close()

fig, axs = plt.subplots(ncols=3,figsize=[12,5],\
    sharey=True,constrained_layout=True)
cmap = plt.get_cmap('coolwarm')
#nature
mp0 = axs[0].pcolormesh(nest_step.ix_true,t,x_t, shading='auto',\
    cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
axs[0].set_xticks(nest_step.ix_true[::120])
axs[0].set_yticks(t[::12])
axs[0].set_yticks(t[::4],minor=True)
axs[0].set_xlabel('site')
axs[0].set_ylabel('day')
axs[0].set_title('nature')
#GM
gm2true = interp1d(nest_step.ix_gm, x_gm, fill_value="extrapolate")
xgm2t = gm2true(nest_step.ix_true)
xd = xgm2t - x_t
mp1 = axs[1].pcolormesh(nest_step.ix_true,t,xd, shading='auto',\
    cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
axs[1].vlines([nest_step.ix_lam[0],nest_step.ix_lam[-1]],0,1,\
    colors='k',linestyle='dashdot',transform=axs[1].get_xaxis_transform())
axs[1].set_xticks(nest_step.ix_true[::120])
axs[1].set_xlabel('site')
axs[1].set_title('GM error')
#LAM
lam2true = interp1d(nest_step.ix_lam, x_lam, fill_value="extrapolate")
xlam2t = lam2true(nest_step.ix_true)
xd = xlam2t - x_t
xd[:,:ist_lam] = np.nan
xd[:,ist_lam+nx_lam:] = np.nan
mp2 = axs[2].pcolormesh(nest_step.ix_true,t,xd, shading='auto',\
    cmap=cmap, norm=Normalize(vmin=-15.0,vmax=15.0))
axs[2].set_xlim(nest_step.ix_true[0],nest_step.ix_true[-1])
axs[2].set_xticks(nest_step.ix_true[::120])
axs[2].set_xlabel('site')
axs[2].set_title('LAM error')
axs[2].vlines([nest_step.ix_lam[nsp],nest_step.ix_lam[-nsp]],0,1,\
    colors='white',linestyle='dashdot',transform=axs[2].get_xaxis_transform())
fig.colorbar(mp2,ax=axs[2],shrink=0.6,pad=0.01)
fig.savefig(figdir/f'test_nest_err_lamstep{lamstep}.png',dpi=300)
plt.show()
plt.close()

np.savetxt(figdir/'ix_t.txt',nest_step.ix_true)
np.savetxt(figdir/'ix_gm.txt',nest_step.ix_gm)
np.savetxt(figdir/'ix_lam.txt',nest_step.ix_lam)
np.save(figdir/'x_t.npy',x_t)
np.save(figdir/'x_gm.npy',x_gm)
np.save(figdir/'x_lam.npy',x_lam)

## power spectrum
sp_t = np.array(sp_t).mean(axis=0)
sp_gm = np.array(sp_gm).mean(axis=0)
sp_lam = np.array(sp_lam).mean(axis=0)
fig, ax = plt.subplots(figsize=[8,6],constrained_layout=True)
#wnum_t = fft.rfftfreq(x0_t.size,d=dx_t) * 2.0 * np.pi
ax.plot(wnum_t,sp_t,c='b',marker='x',label='nature')
#mmax = np.argmin(np.abs(wnum_t - 30)) + 1
#ax.plot(wnum_t[:mmax],sp_t[:mmax],c='tab:blue',marker='x',lw=0.0)
#wnum_gm = fft.rfftfreq(x0_gm.size,d=dx_gm) * 2.0 * np.pi
ax.plot(wnum_gm,sp_gm,c='orange',marker='x',label='GM')
#mmax = np.argmin(np.abs(wnum_gm - 30)) + 1
#ax.plot(wnum_gm[:mmax],sp_gm[:mmax],c='tab:orange',marker='x',lw=0.0)
#wnum_lam = fft.rfftfreq(x0_lam_ext.size,d=dx_lam)[:sp_lam.size] * 2.0 * np.pi
ax.plot(wnum_lam,sp_lam,c='r',marker='x',label='LAM')
#mmax = np.argmin(np.abs(wnum_lam - 30)) + 1
#ax.plot(wnum_lam[:mmax],sp_lam[:mmax],c='tab:green',marker='x',lw=0.0)
ax.set_title("time averaged power spectrum")
ax.set(xlabel=r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)",title='variance power spectra')
#ax.xaxis.set_major_locator(FixedLocator([240./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi]))
#ax.xaxis.set_major_formatter(FixedFormatter([r'$\frac{240}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$']))
#ax.set_xlim(0.5/np.pi,wnum[-1])
secax = ax.secondary_xaxis('top',functions=(wnum2wlen, wlen2wnum))
secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
#ax.xaxis.set_major_locator(FixedLocator([480,240,120,60,8,2,1]))
#ax.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','8','2','1']))
#secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi,np.pi/4,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
#secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\pi$',r'$\frac{\pi}{4}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
ax.xaxis.set_major_locator(FixedLocator([480,240,120,60,30,1]))
ax.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','1']))
secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi/15,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
#ax.set_xscale('log')
ax.set_yscale("log")
ax.vlines([12],0,1,colors='gray',alpha=0.5,transform=ax.get_xaxis_transform())
ax.legend()
fig.savefig(figdir/f'test_nest_psd.png',dpi=300)
plt.show()