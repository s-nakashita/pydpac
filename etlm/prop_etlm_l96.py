import numpy as np 
from numpy.random import default_rng
import numpy.linalg as la 
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200
import sys
sys.path.append('../')
from model.lorenz import L96
from analysis.corrfunc import Corrfunc
from pathlib import Path
import argparse

# model parameters
nx = 40
dt = 0.05 # 6 hour
F = 8.0
model = L96(nx, dt, F)
ix = np.arange(nx)
r = np.array([min(ix[i],nx-ix[i]) for i in range(nx)])

# settings
parser = argparse.ArgumentParser()
parser.add_argument("-vt","--vt",type=int,default=24,\
    help="verification time (hours)")
parser.add_argument("-ne","--nens",type=int,default=8,\
    help="ensemble size")
argsin = parser.parse_args()
vt = argsin.vt
ioffset = vt // 6
nens = argsin.nens

# data
modelname = 'l96'
pt = 'letkf'
datadir = Path(f'/Volumes/dandelion/pyesa/data/{modelname}')
if nens==200:
    xfall = np.load(datadir/f"extfcst_letkf_m{nens}/{modelname}_ufext_linear_{pt}.npy")
else:
    xfall = np.load(datadir/f"extfcst_m{nens}/{modelname}_ufext_linear_{pt}.npy")
print(xfall.shape)
P = np.load(datadir/'Aest.npy')
Lam, v = la.eigh(P)
Psqrt = v @ np.diag(np.sqrt(Lam)) @ v.T
#plt.matshow(Psqrt)
#plt.colorbar()
#plt.show()

figdir = Path('l96')
if not figdir.exists(): figdir.mkdir(parents=True)

icyc0 = 50
beta = 0.1 # Tikhonov regularization
rng = default_rng(seed=509)
# arbitrally perturbations
dx0 = Psqrt @ rng.normal(loc=0.0,scale=1.0,size=Psqrt.shape[1])
# ensemble
Xens = xfall[icyc0,:,:,:]
xb = np.mean(Xens[:,:,:],axis=2)
Xprtb = Xens - xb[:,:,None]
fig, axs = plt.subplots(nrows=3,sharex=True,figsize=[6,6],constrained_layout=True)
axs[0].set_title(f'FT00 K={nens}')
axs[0].plot(ix,Xens[0,],c='magenta',alpha=0.3,lw=1.0)
axs[0].plot(ix,xb[0,],c='k',label='ens mean')
axs[2].plot(ix,dx0,c='k',label=r'$\delta\mathbf{x}_0$',zorder=0)
dx_dyn = dx0.copy()
dx_ens = dx0.copy()
xp = xb[0] + dx0
axs[0].plot(ix,xp,c='gray',label='+prtb')
tlmdyn_list = []
for i in range(ioffset):
    # nonlinear model
    xp = model(xp)
    # dynamical TLM
    dx_dyn = model.step_t(xb[i],dx_dyn)
    axs[2].plot(ix,dx_dyn,c='b',ls='dashed',alpha=0.5,lw=0.5)
    tlm = np.eye(nx)
    for j in range(nx):
        tlm[:,j] = model.step_t(xb[i],tlm[:,j])
    tlmdyn_list.append(tlm)
    # ensemble TLM
    Xi = Xprtb[i,:,:]
    Xj = Xprtb[i+1,:,:]
    cimat = Xi.T @ Xi + beta*np.eye(Xi.shape[1])
    y = Xi.T @ dx_ens
    z = la.pinv(cimat,rcond=1e-4,hermitian=True) @ y
    dx_ens = Xj @ z
    axs[2].plot(ix,dx_ens,c='r',ls='dashed',alpha=0.5,lw=0.5)
axs[1].set_title(f'FT{vt:02d} K={nens}')
axs[1].plot(ix,Xens[ioffset,],c='magenta',alpha=0.3,lw=1.0)
axs[1].plot(ix,xb[ioffset,],c='k',label='ens mean')
axs[1].plot(ix,xp,c='gray',label='nl fcst')
axs[2].set_title('prtb development')
axs[2].plot(ix,xp-xb[ioffset],c='gray',label='nl diff')
axs[2].plot(ix,dx_dyn,c='b',ls='dashed',label='TLM dyn')
axs[2].plot(ix,dx_ens,c='r',ls='dashed',label='TLM ens')
for ax in axs:
    ax.legend()
fig.savefig(figdir/f'dx_c{icyc0}vt{vt}ne{nens}.png')
plt.show()

# localization functions (rho0=gc5, rho1=boxcar)
cfunc = Corrfunc(1.5)
rho0 = np.roll(cfunc(r, ftype='gc5'),nx//2)
ic = ix[nx//2]
rho1 = np.where(np.abs(ix-ic)<2.0,1.0,0.0)
figr, axsr = plt.subplots(nrows=2,sharex=True,figsize=[6,6],constrained_layout=True)
axsr[0].set_title('GC5')
axsr[0].plot(ix,rho0,c='k',lw=3.0,label=r'$\rho_t$')
axsr[1].set_title('Boxcar')
axsr[1].plot(ix,rho1,c='k',lw=3.0,label=r'$\rho_t$')
rho0_dyn = rho0.copy()
rho1_dyn = rho1.copy()
rho0_ens = rho0.copy()
rho1_ens = rho1.copy()
n = len(tlmdyn_list)
for i in range(ioffset):
    # dynamical TLM
    tlm = tlmdyn_list[n-i-1]
    itlm = la.inv(tlm)
    rho0_dyn = np.diag(itlm@np.diag(rho0_dyn)@tlm)
    rho1_dyn = np.diag(itlm@np.diag(rho1_dyn)@tlm)
    axsr[0].plot(ix,rho0_dyn,c='b',ls='dashed',alpha=0.5,lw=0.5)
    axsr[1].plot(ix,rho1_dyn,c='b',ls='dashed',alpha=0.5,lw=0.5)
    # ensemble TLM
    Xj = Xprtb[ioffset-i,:,:]
    Xi = Xprtb[ioffset-i-1,:,:]
    cimat = Xi.T @ Xi + beta*np.eye(Xi.shape[1])
    cjmat = Xj.T @ Xj + beta*np.eye(Xj.shape[1])
    iloc = np.arange(nx)[np.abs(rho0_ens)>1.0e-2]
    Xjl = rho0_ens[iloc][:,None] * Xj[iloc]
    cjlmat = Xj[iloc].T @ Xjl
    cmat = la.pinv(cjmat, rcond=1e-4, hermitian=True) @ cjlmat @ la.pinv(cimat, rcond=1e-4, hermitian=True)
    for i in range(nx):
        rho0_ens[i] = np.dot(Xi[i],np.dot(cmat,Xi[i]))
    rho0_ens = rho0_ens / np.max(rho0_ens) # scaling
    iloc = np.arange(nx)[np.abs(rho1_ens)>1.0e-2]
    Xjl = rho1_ens[iloc][:,None] * Xj[iloc]
    cjlmat = Xj[iloc].T @ Xjl
    cmat = la.pinv(cjmat, rcond=1e-4, hermitian=True) @ cjlmat @ la.pinv(cimat, rcond=1e-4, hermitian=True)
    for i in range(nx):
        rho1_ens[i] = np.dot(Xi[i],np.dot(cmat,Xi[i]))
    rho1_ens = rho1_ens / np.max(rho1_ens) # scaling
    axsr[0].plot(ix,rho0_ens,c='r',ls='dashed',alpha=0.5,lw=0.5)
    axsr[1].plot(ix,rho1_ens,c='r',ls='dashed',alpha=0.5,lw=0.5)
axsr[0].plot(ix,rho0_dyn,c='b',ls='dashed',label='dyn')
axsr[1].plot(ix,rho1_dyn,c='b',ls='dashed',label='dyn')
axsr[0].plot(ix,rho0_ens,c='r',ls='dashed',label='ens')
axsr[1].plot(ix,rho1_ens,c='r',ls='dashed',label='ens')
axsr[0].legend()
figr.suptitle(f'FT{vt:02d} K={nens}')
figr.savefig(figdir/f'prop_c{icyc0}vt{vt}ne{nens}.png')
plt.show()
