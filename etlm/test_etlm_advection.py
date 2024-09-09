import numpy as np 
from numpy.random import default_rng
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
sys.path.append('../')
from model.advection import Advection, Advection_state
from analysis.corrfunc import Corrfunc
from pathlib import Path

figdir = Path('test_advection')
if not figdir.exists(): figdir.mkdir(parents=True)

# model parameters
nx = 60
dx = 100 # m
L = nx*dx
dt = 0.05 # sec
u0 = 10.0 # m / s
xnu0 = 0.0
F0 = 0.0
ix = np.arange(nx)*dx 
r = np.array([min(ix[i],L-ix[i]) for i in range(nx)])
# nature run
nst = Advection_state(nx, u0=u0)
nature = Advection(nst, dx, dt)
cfunc = Corrfunc(dx*10)
rho0 = np.roll(cfunc(r, ftype='gc5'),nx//4)
nst.rho[0,:] = rho0[:]
# ensemble state
nens = 16
rng = default_rng(seed=509)
est = []
for k in range(nens):
    ue = u0 + rng.normal(0.0,scale=2.0)
    st = Advection_state(nx, u0=ue)
    st.rho[0,:] = rho0[:] + rng.normal(0.0,scale=0.5,size=nx)
    est.append(st)
model = Advection(est[0], dx, dt)

tmax = 100.0
ntmax = int(tmax / dt)
for i in range(ntmax):
    nature(nst)
    for k in range(nens):
        model(est[k])

fig, ax = plt.subplots()
rhom = np.zeros(nx)
for k in range(nens):
    ax.plot(ix,est[k].rho[0,],c='magenta',alpha=0.3,lw=1.0)
    rhom = rhom + est[k].rho[0,]
rhom = rhom / nens
ax.plot(ix,rhom,c='k',ls='dashed',lw=3.0,label='ens mean')
ax.plot(ix,nst.rho[0,],c='k',lw=3.0,label='truth')
ax.legend()
ax.set_title(f't={tmax}s')
plt.show()
plt.close()

# TLM
ft = 100.0
nft = int(ft / dt)
prtb = Advection_state(nx,u0=u0)
dtlm = np.eye(nx)
rhoi = nst.rho[0,].copy()
for j in range(nft):
    nature(nst)
    for i in range(nx):
        prtb.rho[0,:] = dtlm[:,i]
        nature.step_t(nst,prtb)
        dtlm[:,i] = prtb.rho[0,:]
fig, ax = plt.subplots(figsize=[6,4],constrained_layout=True)
ax.plot(ix,rhoi,label=r'$\mathbf{\phi}_{0}$')
ax.plot(ix,nst.rho[0,],label=r'$\mathbf{\phi}_{100}$')
ax.plot(ix,dtlm@rhoi,ls='dashed',label=r'$\mathbf{M}\mathbf{\phi}_0$')
ax.legend()
fig.savefig(figdir/'phi.png')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=[4,4],constrained_layout=True)
p0 = ax.matshow(dtlm,cmap='RdBu_r',vmin=-1.2,vmax=1.2)
fig.colorbar(p0,ax=ax,shrink=0.6,pad=0.01)
ax.set_title(r'$\mathbf{M}^\mathrm{dyn}_{0,100}$')
fig.savefig(figdir/'dtlm.png')
plt.show()
plt.close()

dtlminv = la.inv(dtlm)
floct = nst.rho[0,]
mloct = np.diag(floct)
mloc0 = dtlminv @ mloct @ dtlm
floc0 = np.diag(mloc0)
fig = plt.figure(figsize=[6,6],constrained_layout=True)
gs = GridSpec(nrows=3,ncols=2,figure=fig)
ax00 = fig.add_subplot(gs[0,0])
ax01 = fig.add_subplot(gs[0,1])
ax10 = fig.add_subplot(gs[1:,0])
ax11 = fig.add_subplot(gs[1:,1])
ax01.plot(ix,floct)
ax01.set_title(r'$\rho_t$')
p11=ax11.matshow(mloct)
fig.colorbar(p11,ax=ax11,shrink=0.6,pad=0.01)
ax11.set_title(r'$\mathrm{diag}(\rho_t)$')
p10=ax10.matshow(mloc0)
fig.colorbar(p10,ax=ax10,shrink=0.6,pad=0.01)
ax10.set_title(r'$\mathrm{diag}(\rho_0)=\mathbf{M}^{-1}_{0,t}\mathrm{diag}(\rho_t)\mathbf{M}_{0,t}$')
ax00.plot(ix,floc0)
ax00.set_title(r'$\rho_0$')
fig.savefig(figdir/'prop_rho.png')
plt.show()
plt.close()

y = rng.normal(loc=0.0,scale=0.5,size=nx)
z = dtlm @ y
rhoy = floc0 * y
mrhoy = dtlm @ rhoy
rhoz = floct * z
fig, ax = plt.subplots()
ax.plot(ix,rhoz,label=r'$\rho_t \circ \mathbf{z}$')
ax.plot(ix,mrhoy,ls='dashed',label=r'$\mathbf{M}_{0,t}(\rho_0\circ\mathbf{y})$')
ax.legend()
fig.savefig(figdir/'test_prop_rho.png')
plt.show()
plt.close()
#exit()

# ETLM
tint = 100.0
nint = int(tint / dt)
etlm = np.eye(nx)
fig, axs = plt.subplots(nrows=2,sharex=True,figsize=[6,6],constrained_layout=True)
Xi = np.zeros((nx,nens))
for k in range(nens):
    Xi[:,k] = est[k].rho[0,:]
    axs[0].plot(ix,Xi[:,k],c='magenta',alpha=0.3,lw=1.0)
rhom = np.mean(Xi,axis=1)
axs[0].plot(ix,rhom,c='k',ls='dashed',lw=3.0,label='ens mean')
Xi = Xi - rhom[:,None]
axs[0].plot(ix,rhoi,c='k',lw=3.0,label='truth')
for j in range(nft):
    Xj = np.zeros((nx,nens))
    for k in range(nens):
        model(est[k])
        Xj[:,k] = est[k].rho[0,:]
    if (j+1)%nint==0:
        rhom = np.mean(Xj,axis=1)
        Xj = Xj - rhom[:,None]
        cmat = Xi.T @ Xi
        beta = 0.3 * nens * 1.0e-8 * la.norm(cmat,ord=2)
        u, s, vt = la.svd(cmat+beta*np.eye(cmat.shape[0]))
        zmat = u @ np.diag(1.0/s) @ u.T @ Xi.T
        etlmtmp = Xj @ zmat
        etlm = etlmtmp @ etlm
        Xi[:,:] = Xj[:,:]
for k in range(nens):
    Xj[:,k] = est[k].rho[0,:]
    axs[1].plot(ix,Xj[:,k],c='magenta',alpha=0.3,lw=1.0)
rhom = np.mean(Xj,axis=1)
axs[1].plot(ix,rhom,c='k',ls='dashed',lw=3.0,label='ens mean')
axs[1].plot(ix,nst.rho[0,],c='k',lw=3.0,label='truth')
axs[1].plot(ix,etlm@rhoi,c='gray',ls='dashed',lw=3.0,label=r'$\mathbf{M}^\mathrm{ens}\mathbf{\phi}_0$')
axs[0].legend(title='t=0')
axs[1].legend(title='t=100')
ax.set_title(r'$\mathbf{\phi}^\mathrm{ens}$, '+f'K={nens}')
fig.savefig(figdir/f'phi_k{nens}ni{nint}.png')
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=[4,4],constrained_layout=True)
p0 = ax.matshow(etlm,cmap='RdBu_r',vmin=-1.2,vmax=1.2)
fig.colorbar(p0,ax=ax,shrink=0.6,pad=0.01)
ax.set_title(r'$\mathbf{M}^\mathrm{ens}_{0,100}$, '+f'K={nens} Tint={tint:.1f}s')
fig.savefig(figdir/f'etlm_k{nens}ni{nint}.png')
plt.show()
plt.close()
