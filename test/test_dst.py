import numpy as np 
from scipy.interpolate import interp1d
import scipy.fft as fft
import os
import sys
sys.path.append('../analysis')
from trunc1d import Trunc1d
import matplotlib.pyplot as plt

nx_true = 96
intgm = 4
nx_gm = nx_true // intgm
nx_lam = 48
ist_lam = 6

ix_true = np.arange(nx_true)
ix_gm = np.arange(0,nx_gm)*intgm
ix_lam = np.arange(ist_lam,nx_lam+ist_lam)

ntrunc = None
fig = plt.figure(figsize=[8,4],constrained_layout=True)
ax0 = fig.add_subplot(121,projection='polar')
#ax1 = fig.add_subplot(132,projection='polar')
ax2 = fig.add_subplot(122)
r = 3.0
dx = 2.0 * np.pi / nx_true
ix = np.arange(nx_true) * dx
#x = np.random.randn(nx_true)
n = np.array([2.,6.,12.,20.,30.,40.])
x = np.sum(np.sin(n[None,:]*ix[:,None])/np.sqrt(n[None,:]),axis=1)
ax0.plot(ix,x+r)
trunc1d = Trunc1d(ix,ntrunc,cyclic=True,ttype='s')
xtrunc,f,y = trunc1d(x,return_coef=True)
print(trunc1d.ix_trunc)
ax0.plot(trunc1d.ix_trunc,xtrunc+r)
#ax1.plot(ix,x-xtrunc+r)
#ax1.set_title("diff")
width=0.8
ax2.bar(2.0*np.pi*f,np.abs(y*dx)**2,width=width,alpha=0.5)
_,f,ytrunc = trunc1d(xtrunc,return_coef=True)
#ytrunc = fft.dst(xtrunc[1:-1],type=1)
#f = fft.fftshift(\
#    fft.fftfreq(trunc1d.ix_trunc.size,trunc1d.dx_trunc)\
#    )
ax2.bar(2.0*np.pi*f,np.abs(ytrunc*trunc1d.dx_trunc)**2,width=width,alpha=0.5)
ax2.set_xlabel(r'$\omega$')
#for ax in [ax0,ax1]:
#    #ax.grid()
#    ax.set_ylim(0,2*r)
ax0.set_ylim(0,2*r)
fig.savefig(f"test_dst_gm_ntrunc{trunc1d.ntrunc}.png",dpi=300)
plt.show()
plt.close()
#exit()
fig = plt.figure(figsize=[8,4],constrained_layout=True)
ax0 = fig.add_subplot(121,projection='polar')
#ax1 = fig.add_subplot(132,projection='polar')
ax2 = fig.add_subplot(122)
ix_lam = ix_lam * dx
#x_lam = np.random.randn(nx_lam)
i0 = np.argmin(np.abs(ix-ix_lam[0]))
i1 = np.argmin(np.abs(ix-ix_lam[-1]))
x_lam = x[i0:i1+1]
ax0.plot(ix_lam,x_lam+r)
trunc1d = Trunc1d(ix_lam,ntrunc,cyclic=False,ttype='s',nghost=0)#,nglobal=nx_true)
xtrunc,f,y = trunc1d(x_lam,return_coef=True)
ax0.plot(trunc1d.ix_trunc,xtrunc+r)
#ax1.plot(ix_lam,x_lam-xtrunc+r)
#ax1.set_title("diff")
ax2.bar(2.0*np.pi*f,np.abs(y*dx)**2,width=width,alpha=0.5)
_,f,ytrunc = trunc1d(xtrunc,return_coef=True)
#ytrunc = fft.fftshift(fft.fft(xtrunc))
#f = fft.fftshift(\
#    fft.fftfreq(trunc1d.ix_trunc.size,trunc1d.dx_trunc)\
#    )
ax2.bar(2.0*np.pi*f,np.abs(ytrunc*trunc1d.dx_trunc)**2,width=width,alpha=0.5)
ax2.set_xlabel(r'$\omega$')
#for ax in [ax0,ax1]:
#    #ax.grid()
#    ax.set_ylim(0,2*r)
ax0.set_ylim(0,2*r)
fig.savefig(f"test_dst_lam_ntrunc{trunc1d.ntrunc}.png",dpi=300)
plt.show()