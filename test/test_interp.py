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
plt.plot(ix_true,ix_true,ls='dashed')
plt.plot(ix_gm,ix_gm,lw=0.0,marker='x')
plt.plot(ix_lam,ix_lam)
plt.show(block=False)
plt.close()

x_gm = np.random.randn(nx_gm, 5)
gm2lam = interp1d(ix_gm, x_gm, axis=0, fill_value='extrapolate')
x_lam = gm2lam(ix_lam)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=[10,5],constrained_layout=True)
cmap = plt.get_cmap('tab10')
for j, ax in zip(range(x_gm.shape[1]),axs.flatten()):
    ax.plot(ix_gm, x_gm[:,j], lw=2.0, c=cmap(j), alpha=0.5,marker='x')
    ax.plot(ix_lam, x_lam[:,j], lw=1.0, c=cmap(j), marker='.')
axs[1,2].remove()
plt.show(block=False)
plt.close()

x_lam = np.random.randn(nx_lam)
i0 = np.argmin(np.abs(ix_gm - ix_lam[0]))
if ix_gm[i0] < ix_lam[0]: i0+=1
i1 = np.argmin(np.abs(ix_gm - ix_lam[-1]))
print(ix_lam)
print(ix_gm[i0:i1+1])
lam2gm = interp1d(ix_lam,x_lam)
x_lam2gm = lam2gm(ix_gm[i0:i1+1])
plt.plot(ix_lam,x_lam)
plt.plot(ix_gm[i0:i1+1],x_lam2gm)
JH2 = np.zeros((i1-i0+1,nx_lam))
tmp_lam2gm = interp1d(ix_lam,np.eye(nx_lam),axis=0)
JH2 = tmp_lam2gm(ix_gm[i0:i1+1])
plt.plot(ix_gm[i0:i1+1],JH2@x_lam)
plt.show(block=False)
plt.close()

fig, axs = plt.subplots(ncols=2)
dx = 2.0 * np.pi / nx_true
ix = np.arange(nx_true) * dx
x = np.random.randn(nx_true)
axs[0].plot(ix,x)
y = fft.fft(x)
f = fft.fftfreq(nx_true,dx)
axs[1].plot(fft.fftshift(f),fft.fftshift(y))
fmax = f[nx_true//2-1]
print(f"f={f}")
print(f"fmax={fmax}")
ftmax = 10.0/np.pi
print(f"ftmax={ftmax}")
ntrunc = np.argmin(np.abs(abs(f)-ftmax))
print(f"ntrunc={ntrunc}")
ftrunc = f.copy()
ftrunc[ntrunc+1:f.size-ntrunc] = 0.0
print(f"ftrunc={ftrunc}")
ytrunc = y.copy()
ytrunc[ntrunc+1:y.size-ntrunc] = 0.0
axs[1].plot(fft.fftshift(f),fft.fftshift(ytrunc))
xtrunc = fft.ifft(ytrunc)
axs[0].plot(ix,xtrunc)
#F = fft.fft(np.eye(x.size),axis=0)
#print(f"F.shape={F.shape}")
#T = np.eye(F.shape[0])
#T[ntrunc+1:f.size-ntrunc,:] = 0.0
#Fi = fft.ifft(np.eye(T.shape[0]),axis=0)
#print(f"Fi.shape={Fi.shape}")
#xtrunc2 = np.dot(Fi,np.dot(T,np.dot(F,x)))
trunc1d = Trunc1d(ix,ntrunc,cyclic=True)
xtrunc2 = trunc1d(x)
axs[0].plot(ix,xtrunc2)
plt.show()
plt.close()
plt.plot(ix,xtrunc-xtrunc2)
plt.show()
#exit()
fig, axs = plt.subplots(ncols=2)
#xg = np.random.randn(nx_gm)
#ix_gm = ix_gm * dx
#gm2lam = interp1d(ix_gm,xg)
#plt.plot(ix_gm,xg,lw=2.0)
ix_lam = ix_lam * dx
#x = gm2lam(ix_lam)
x = np.random.randn(nx_lam)
axs[0].plot(ix_lam,x)
nghost = ix_lam.size // 10 # ghost region for periodicity in LAM
dwindow = (1.0 + np.cos(np.pi*np.arange(1,nghost+1)/nghost))*0.5
Lx_lam = 2.0 * np.pi * (nx_lam + 2*nghost) / nx_true
x_ext = np.zeros(nx_lam + 2*nghost - 1)
x_ext[nghost:nghost+nx_lam] = x[:]
x_ext[0:nghost] = x[0] * dwindow[::-1]
x_ext[nghost+nx_lam:] = x[-1] * dwindow[1:]
ix_lam_ext = ix_true[ist_lam-nghost:ist_lam+nx_lam+nghost-1] * dx
axs[0].plot(ix_lam_ext,x_ext,ls='dotted')
y_ext = fft.fft(x_ext)
f = fft.fftfreq(x_ext.size,dx)
axs[1].plot(fft.fftshift(f),fft.fftshift(y_ext))
fmax = f[x_ext.size//2-1]
print(f"f={f}")
print(f"fmax={fmax}")
ftmax = 10.0/np.pi
print(f"ftmax={ftmax}")
ntrunc = np.argmin(np.abs(abs(f)-ftmax))
print(f"ntrunc={ntrunc}")
ftrunc = f.copy()
ftrunc[ntrunc+1:f.size-ntrunc] = 0.0
print(f"ftrunc={ftrunc}")
ytrunc = y_ext.copy()
ytrunc[ntrunc+1:y_ext.size-ntrunc] = 0.0
axs[1].plot(fft.fftshift(f),fft.fftshift(ytrunc))
xtrunc_ext = fft.ifft(ytrunc)
axs[0].plot(ix_lam_ext,xtrunc_ext,ls='dashed')
xtrunc = xtrunc_ext[nghost:nghost+nx_lam]
axs[0].plot(ix_lam,xtrunc)
plt.show()
plt.close()
fig, axs = plt.subplots(ncols=2)
axs[0].plot(ix_lam,xtrunc)
#E = np.zeros((nx_lam+2*nghost-1,nx_lam))
#E[0:nghost,0] = dwindow[::-1]
#E[nghost:nghost+nx_lam,:] = np.eye(nx_lam)[:,:]
#E[nghost+nx_lam:,-1] = dwindow[1:]
#F = fft.fft(np.eye(E.shape[0]),axis=0)
#print(f"F.shape={F.shape}")
#T = np.eye(F.shape[0])
#T[ntrunc+1:f.size-ntrunc,:] = 0.0
#Fi = fft.ifft(np.eye(T.shape[0]),axis=0)
#print(f"Fi.shape={Fi.shape}")
#Ei = np.zeros((nx_lam,nx_lam+2*nghost-1))
#Ei[:,nghost:nghost+nx_lam] = np.eye(nx_lam)[:,:]
#xtrunc2 = np.dot(Ei,np.dot(Fi,np.dot(T,np.dot(F,np.dot(E,x)))))
trunc1d = Trunc1d(ix_lam,ntrunc,cyclic=False)
xtrunc2 = trunc1d(x)
axs[0].plot(ix_lam,xtrunc2)
axs[1].plot(ix_lam,xtrunc-xtrunc2)
plt.show()