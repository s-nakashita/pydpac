import numpy as np 
from numpy.random import default_rng
from scipy.interpolate import interp1d
import scipy.fft as fft
import os
import sys
sys.path.append('../analysis')
from trunc1d import Trunc1d
import matplotlib.pyplot as plt
from pathlib import Path

figdir = Path('trunc')
if not figdir.exists(): figdir.mkdir()

rng = default_rng(509)
ttype = 'f'
ntrunc = 24
if len(sys.argv)>1:
    ttype = sys.argv[1]
if len(sys.argv)>2:
    ntrunc = int(sys.argv[2])

nx_true = 96
intgm = 4
nx_gm = nx_true // intgm
nx_lam = 24
if ttype == 'f':
    nx_lam += 1
ist_lam = 6

ix_true = np.arange(nx_true)
ix_gm = np.arange(0,nx_gm)*intgm
ix_lam = np.arange(ist_lam,nx_lam+ist_lam)

r = 3.0
dx = 2.0 * np.pi / nx_true
ix = np.arange(nx_true) * dx
#x = np.random.randn(nx_true)
n = np.array([2.,6.,12.,20.,30.,40.])
phi = rng.random(n.size)*np.pi
#x = np.sum(np.sin(n[None,:]*ix[:,None])/np.sqrt(n[None,:]),axis=1)
x = np.sum(np.exp(1.0j*(n[None,:]*ix[:,None]+phi[None,:])).real/np.sqrt(n[None,:]),axis=1)
trunc1d = Trunc1d(ix,ntrunc,cyclic=True,ttype=ttype)
xtrunc,f,y,f_trunc,ytrunc = trunc1d(x,return_coef=True)
print(trunc1d.ix_trunc)
if ntrunc is not None:
    trunc1d_nor = Trunc1d(ix,ntrunc,cyclic=True,resample=False,ttype=ttype)
    xtrunc2 = trunc1d_nor(x)
fig = plt.figure(figsize=[8,4],constrained_layout=True)
ax0 = fig.add_subplot(121,projection='polar')
#ax1 = fig.add_subplot(132,projection='polar')
ax2 = fig.add_subplot(122)
ax0.plot(ix,x+r)
ax0.plot(trunc1d.ix_trunc,xtrunc+r)
if ntrunc is not None:
    ax0.plot(trunc1d_nor.ix_trunc,xtrunc2+r,ls='dashed')
if ix.size == trunc1d.ix_trunc.size:
    ax0.plot(ix,x-xtrunc+r,ls='dashed')
#ax1.set_title("diff")
width=0.8
if trunc1d.ttype!='c':
    ax2.bar(f,np.abs(y*dx)**2,width=width,alpha=0.5)
else:
    ax2.bar(f,np.abs(y)**2,width=width,alpha=0.5)
##_,f,ytrunc = trunc1d(xtrunc,return_coef=True)
#ytrunc = fft.fftshift(fft.fft(xtrunc))
#f_trunc = fft.fftshift(\
#    fft.fftfreq(trunc1d.ix_trunc.size,trunc1d.dx_trunc)\
#    )
print(f_trunc)
if trunc1d.ttype=='f':
    ax2.bar(f_trunc,np.abs(ytrunc*dx)**2,width=width,alpha=0.5)
elif trunc1d.ttype=='s':
    ax2.bar(f_trunc,np.abs(ytrunc*trunc1d.dx_trunc)**2,width=width,alpha=0.5)
else:
    ax2.bar(f_trunc,np.abs(ytrunc)**2,width=width,alpha=0.5)
ax2.set_xlabel(r'$\omega$')
#for ax in [ax0,ax1]:
#    #ax.grid()
#    ax.set_ylim(0,2*r)
ax0.set_ylim(0,2*r)
fig.suptitle(f"transform={trunc1d.tname[trunc1d.ttype]}")
fig.savefig(figdir/f"test_{trunc1d.ttype}_gm_ntrunc{trunc1d.ntrunc}.png",dpi=300)
plt.show()
plt.close()
#exit()
ix_lam = ix_lam * dx
#x_lam = np.random.randn(nx_lam)
i0 = np.argmin(np.abs(ix-ix_lam[0]))
i1 = np.argmin(np.abs(ix-ix_lam[-1]))
x_lam = x[i0:i1+1]
ftrunc = trunc1d.ftrunc
trunc1d = Trunc1d(ix_lam,ftrunc=ftrunc,cyclic=False,ttype=ttype)#,nglobal=nx_true)
xtrunc,f,y,f_trunc,ytrunc = trunc1d(x_lam,return_coef=True)
print(trunc1d.ix_trunc)
if ntrunc is not None:
    trunc1d_nor = Trunc1d(ix_lam,ftrunc=ftrunc,cyclic=False,ttype=ttype,resample=False)
    xtrunc2 = trunc1d_nor(x_lam)
fig = plt.figure(figsize=[8,4],constrained_layout=True)
ax0 = fig.add_subplot(121,projection='polar')
#ax1 = fig.add_subplot(132,projection='polar')
ax2 = fig.add_subplot(122)
ax0.plot(ix_lam,x_lam+r)
ax0.plot(trunc1d.ix_trunc,xtrunc+r)
if ntrunc is not None:
    ax0.plot(trunc1d_nor.ix_trunc,xtrunc2+r,ls='dashed')
if ix_lam.size == trunc1d.ix_trunc.size:
    ax0.plot(ix_lam,x_lam-xtrunc+r,ls='dashed')
#ax1.plot(ix_lam,x_lam-xtrunc+r)
#ax1.set_title("diff")
#print(2.0*np.pi*f)
if trunc1d.ttype!='c':
    ax2.bar(f,np.abs(y*dx)**2,width=width,alpha=0.5)
else:
    ax2.bar(f,np.abs(y)**2,width=width,alpha=0.5)
##_,f,ytrunc = trunc1d(xtrunc,return_coef=True)
#ytrunc = fft.fftshift(fft.fft(xtrunc))
#f_trunc = fft.fftshift(\
#    fft.fftfreq(trunc1d.ix_trunc.size,trunc1d.dx_trunc)\
#    )
#print(2.0*np.pi*f_trunc)
if trunc1d.ttype=='f':
    ax2.bar(f_trunc,np.abs(ytrunc*dx)**2,width=width,alpha=0.5)
elif trunc1d.ttype=='s':
    ax2.bar(f_trunc,np.abs(ytrunc*trunc1d.dx_trunc)**2,width=width,alpha=0.5)
else:
    ax2.bar(f_trunc,np.abs(ytrunc)**2,width=width,alpha=0.5)
ax2.set_xlabel(r'$\omega$')
#for ax in [ax0,ax1]:
#    #ax.grid()
#    ax.set_ylim(0,2*r)
ax0.set_ylim(0,2*r)
fig.suptitle(f"transform={trunc1d.tname[trunc1d.ttype]}")
fig.savefig(figdir/f"test_{trunc1d.ttype}_lam_ntrunc{trunc1d.ntrunc}.png",dpi=300)
plt.show()