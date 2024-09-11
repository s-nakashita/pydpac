import numpy as np 
from numpy.random import default_rng
import numpy.linalg as la 
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200
from matplotlib.lines import Line2D
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
parser.add_argument("-lp","--lpatch",type=int,default=1,\
    help="patch size")
parser.add_argument("-lh","--lhalo",type=int,default=1,\
    help="halo size")
argsin = parser.parse_args()
vt = argsin.vt
ioffset = vt // 6
nens = argsin.nens
lpatch = argsin.lpatch
lhalo = argsin.lhalo
# patch locations
npatch = nx // lpatch
ispatch = np.zeros(npatch,dtype=int)
iepatch = np.zeros(npatch,dtype=int)
ishalo = np.zeros(npatch,dtype=int)
iehalo = np.zeros(npatch,dtype=int)
lletlm = np.zeros(npatch,dtype=int)
for i in range(npatch):
    ispatch[i] = i*lpatch
    iepatch[i] = ispatch[i]+lpatch-1
    ishalo[i] = ispatch[i] - lhalo
    iehalo[i] = iepatch[i] + lhalo
    lletlm[i] = iehalo[i] - ishalo[i] + 1
    if iehalo[i] > nx-1:
        iehalo[i] = nx - iehalo[i]
    print(f"ispatch={ispatch[i]} iepatch={iepatch[i]} ishalo={ishalo[i]} iehalo={iehalo[i]} lletlm={lletlm[i]}")

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
    # localized ensemble TLM
    Xi = Xprtb[i,:,:]
    Xj = Xprtb[i+1,:,:]
    for l in range(npatch):
        Xip = np.roll(Xi,-ishalo[l],axis=0)[:lletlm[l]]
        Xjp = np.roll(Xj,-ishalo[l],axis=0)[:lletlm[l]]
        cimat = Xip.T @ Xip + beta*np.eye(Xip.shape[1])
        y = Xip.T @ np.roll(dx_ens,-ishalo[l])[:lletlm[l]]
        z = la.pinv(cimat,rcond=1e-4,hermitian=True) @ y
        dx_tmp = Xjp @ z
        dx_ens[ispatch[l]:iepatch[l]+1] = dx_tmp[lhalo:-lhalo]
    axs[2].plot(ix,dx_ens,c='r',ls='dashed',alpha=0.5,lw=0.5)
axs[1].set_title(f'FT{vt:02d} K={nens}')
axs[1].plot(ix,Xens[ioffset,],c='magenta',alpha=0.3,lw=1.0)
axs[1].plot(ix,xb[ioffset,],c='k',label='ens mean')
axs[1].plot(ix,xp,c='gray',label='nl fcst')
axs[2].set_title(f'prtb development lpatch={lpatch} lhalo={lhalo}')
axs[2].plot(ix,xp-xb[ioffset],c='gray',label='nl diff')
axs[2].plot(ix,dx_dyn,c='b',ls='dashed',label='TLM dyn')
axs[2].plot(ix,dx_ens,c='r',ls='dashed',label='LETLM')
for ax in axs:
    ax.legend()
fig.savefig(figdir/f'dx_letlm_c{icyc0}vt{vt}ne{nens}.png')
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
for i in range(ioffset):
    # dynamical TLM
    tlm = tlmdyn_list[ioffset-i-1]
    itlm = la.inv(tlm)
    rho0_dyn = np.diag(itlm@np.diag(rho0_dyn)@tlm)
    rho1_dyn = np.diag(itlm@np.diag(rho1_dyn)@tlm)
    axsr[0].plot(ix,rho0_dyn,c='b',ls='dashed',alpha=0.5,lw=0.5)
    axsr[1].plot(ix,rho1_dyn,c='b',ls='dashed',alpha=0.5,lw=0.5)
    # ensemble TLM
    Xj = Xprtb[ioffset-i,:,:]
    Xi = Xprtb[ioffset-i-1,:,:]
    for l in range(npatch):
        Xip = np.roll(Xi,-ishalo[l],axis=0)[:lletlm[l]]
        Xjp = np.roll(Xj,-ishalo[l],axis=0)[:lletlm[l]]
        cimat = Xip.T @ Xip + beta*np.eye(Xip.shape[1])
        cjmat = Xjp.T @ Xjp + beta*np.eye(Xjp.shape[1])
        rho0tmp = np.roll(rho0_ens,-ishalo[l])[:lletlm[l]]
        rho1tmp = np.roll(rho1_ens,-ishalo[l])[:lletlm[l]]
        Xjl = rho0tmp[:,None] * Xjp
        cjlmat = Xjp.T @ Xjl
        cmat0 = la.pinv(cjmat, rcond=1e-4, hermitian=True) @ cjlmat @ la.pinv(cimat, rcond=1e-4, hermitian=True)
        Xjl = rho1tmp[:,None] * Xjp
        cjlmat = Xjp.T @ Xjl
        cmat1 = la.pinv(cjmat, rcond=1e-4, hermitian=True) @ cjlmat @ la.pinv(cimat, rcond=1e-4, hermitian=True)
        ii = lhalo
        for i in range(ispatch[l],iepatch[l]+1):
            rho0_ens[i] = np.dot(Xip[ii],np.dot(cmat0,Xip[ii]))
            rho1_ens[i] = np.dot(Xip[ii],np.dot(cmat1,Xip[ii]))
            ii += 1
    rho0_ens = rho0_ens / np.max(rho0_ens) # scaling
    rho1_ens = rho1_ens / np.max(rho1_ens) # scaling
    axsr[0].plot(ix,rho0_ens,c='r',ls='dashed',alpha=0.5,lw=0.5)
    axsr[1].plot(ix,rho1_ens,c='r',ls='dashed',alpha=0.5,lw=0.5)
axsr[0].plot(ix,rho0_dyn,c='b',ls='dashed',label='dyn')
axsr[1].plot(ix,rho1_dyn,c='b',ls='dashed',label='dyn')
axsr[0].plot(ix,rho0_ens,c='r',ls='dashed',label='LETLM')
axsr[1].plot(ix,rho1_ens,c='r',ls='dashed',label='LETLM')
axsr[0].legend()
figr.suptitle(f'FT{vt:02d} K={nens} lpatch={lpatch} lhalo={lhalo}')
figr.savefig(figdir/f'prop_letlm_c{icyc0}vt{vt}ne{nens}.png')
plt.show()

# nonlinear check
y = Psqrt @ rng.normal(loc=0.0,scale=1.0,size=Psqrt.shape[1])
xp = xb[0] + y
rho0y_dyn = rho0_dyn * y
rho1y_dyn = rho1_dyn * y
rho0y_ens = rho0_ens * y
rho1y_ens = rho1_ens * y
xp0_dyn = xb[0] + rho0y_dyn
xp1_dyn = xb[0] + rho1y_dyn
xp0_ens = xb[0] + rho0y_ens
xp1_ens = xb[0] + rho1y_ens

for i in range(ioffset):
    # TLM
    y = model.step_t(xb[i],y)
    rho0y_dyn = model.step_t(xb[i],rho0y_dyn)
    rho1y_dyn = model.step_t(xb[i],rho1y_dyn)
    rho0y_ens = model.step_t(xb[i],rho0y_ens)
    rho1y_ens = model.step_t(xb[i],rho1y_ens)
    # NLM
    xp = model(xp)
    xp0_dyn = model(xp0_dyn)
    xp1_dyn = model(xp1_dyn)
    xp0_ens = model(xp0_ens)
    xp1_ens = model(xp1_ens)

fig, axs = plt.subplots(nrows=2,sharex=True,figsize=[6,6],constrained_layout=True)
axs[0].set_title('GC5')
axs[1].set_title('Boxcar')
axs[0].plot(ix,rho0*y,c='k',lw=2.0,label=r'$\rho_t\circ(\mathbf{M}\mathbf{y})$')
axs[1].plot(ix,rho1*y,c='k',lw=2.0,label=r'$\rho_t\circ(\mathbf{M}\mathbf{y})$')
axs[0].plot(ix,rho0y_dyn,c='b',label=r'$\mathbf{M}(\rho_0^\mathrm{dyn}\circ\mathbf{y})$')
axs[1].plot(ix,rho1y_dyn,c='b',label=r'$\mathbf{M}(\rho_0^\mathrm{dyn}\circ\mathbf{y})$')
axs[0].plot(ix,rho0y_ens,c='r',label=r'$\mathbf{M}(\rho_0^\mathrm{ens}\circ\mathbf{y})$')
axs[1].plot(ix,rho1y_ens,c='r',label=r'$\mathbf{M}(\rho_0^\mathrm{ens}\circ\mathbf{y})$')
znl = xp - xb[ioffset]
axs[0].plot(ix,rho0*znl,c='gray',lw=2.0,ls='dashed',label=r'$\rho_t\circ[M(\mathbf{x}+\mathbf{y})-M(\mathbf{x})]$')
axs[1].plot(ix,rho1*znl,c='gray',lw=2.0,ls='dashed',label=r'$\rho_t\circ[M(\mathbf{x}+\mathbf{y})-M(\mathbf{x})]$')
znl0_dyn = xp0_dyn - xb[ioffset]
znl1_dyn = xp1_dyn - xb[ioffset]
axs[0].plot(ix,znl0_dyn,c='cyan',ls='dashed',label=r'$M(\mathbf{x}+\rho_0^\mathrm{dyn}\circ\mathbf{y})-M(\mathbf{x})$')
axs[1].plot(ix,znl1_dyn,c='cyan',ls='dashed',label=r'$M(\mathbf{x}+\rho_0^\mathrm{dyn}\circ\mathbf{y})-M(\mathbf{x})$')
znl0_ens = xp0_ens - xb[ioffset]
znl1_ens = xp1_ens - xb[ioffset]
axs[0].plot(ix,znl0_ens,c='magenta',ls='dashed',label=r'$M(\mathbf{x}+\rho_0^\mathrm{ens}\circ\mathbf{y})-M(\mathbf{x})$')
axs[1].plot(ix,znl1_ens,c='magenta',ls='dashed',label=r'$M(\mathbf{x}+\rho_0^\mathrm{ens}\circ\mathbf{y})-M(\mathbf{x})$')
lines_lin = [
    Line2D([0],[0],color='k',lw=2.0),
    Line2D([0],[0],color='b'),
    Line2D([0],[0],color='r')
]
labels_lin = [
    r'$\rho_t\circ(\mathbf{M}\mathbf{y})$',
    r'$\mathbf{M}(\rho_0^\mathrm{dyn}\circ\mathbf{y})$',
    r'$\mathbf{M}(\rho_0^\mathrm{ens}\circ\mathbf{y})$'
]
lines_nonlin = [
    Line2D([0],[0],color='gray',lw=2.0,ls='dashed'),
    Line2D([0],[0],color='cyan',ls='dashed'),
    Line2D([0],[0],color='magenta',ls='dashed')
]
labels_nonlin = [
    r'$\rho_t\circ[M(\mathbf{x}+\mathbf{y})-M(\mathbf{x})]$',
    r'$M(\mathbf{x}+\rho_0^\mathrm{dyn}\circ\mathbf{y})-M(\mathbf{x})$',
    r'$M(\mathbf{x}+\rho_0^\mathrm{ens}\circ\mathbf{y})-M(\mathbf{x})$'
]
axs[0].legend(lines_lin,labels_lin)#loc='upper left',bbox_to_anchor=(1.01,1.0))
axs[1].legend(lines_nonlin,labels_nonlin)
fig.suptitle(f'FT{vt:02d} K={nens} lpatch={lpatch} lhalo={lhalo}')
fig.savefig(figdir/f'test_prop_letlm_c{icyc0}vt{vt}ne{nens}.png')
plt.show()
