import numpy as np 
from numpy.random import default_rng
import numpy.linalg as la
from scipy import fft
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200
from matplotlib.gridspec import GridSpec
import sys
sys.path.append('../')
from model.advection import Advection, Advection_state
from analysis.corrfunc import Corrfunc
from pathlib import Path

figdir = Path('test_advection')
if not figdir.exists(): figdir.mkdir(parents=True)

# model parameters
nx = 300
dx = 100 # m
L = nx*dx # m
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
nens = 32
if len(sys.argv)>1:
    nens = int(sys.argv[1])
rng = default_rng(seed=509)
phi = fft.ifft(np.eye(nx)*np.sqrt(nx))
phih = phi.conj().T
kw = fft.fftfreq(nx,d=1/nx)
Lam = np.exp(-1.0*(kw/5.0)*(kw/5.0))
pmat = (phi@np.diag(Lam)@phih).real
aens0 = np.diag(np.sqrt(Lam))@rng.normal(0.0,scale=1.0,size=(nx,nens))
theta0 = rng.normal(0.0,scale=1.0,size=(nens))*2.0*np.pi
xens0 = (phi@(aens0*np.exp(1.0j*theta0[None,:]))).real
est = []
for k in range(nens):
    ue = u0 + rng.normal(0.0,scale=2.0)
    st = Advection_state(nx, u0=ue)
    st.rho[0,:] = rho0[:] + xens0[:,k] #+ rng.normal(0.0,scale=0.5,size=nx)
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
plt.show(block=False)
plt.close()

# TLM
ft = 10.0
if len(sys.argv)>2:
    ft = float(sys.argv[2])
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
ax.plot(ix,nst.rho[0,],label=r"$\mathbf{\phi}"+f"_{{{ft:.0f}}}$")
ax.plot(ix,dtlm@rhoi,ls='dashed',label=r'$\mathbf{M}_{0,'+f'{{{ft:.0f}}}'+r'}\mathbf{\phi}_0$')
ax.legend()
fig.savefig(figdir/f'phi_ft{ft:.0f}.png')
plt.show(block=False)
plt.close()
#exit()

dtlminv = la.inv(dtlm)
fig, axs = plt.subplots(ncols=2,figsize=[6,4],constrained_layout=True)
p0 = axs[0].matshow(dtlm,cmap='RdBu_r',vmin=-1.2,vmax=1.2)
fig.colorbar(p0,ax=axs[0],shrink=0.6,pad=0.01)
axs[0].set_title(r'$\mathbf{M}^\mathrm{dyn}_{0,'+f'{{{ft:.0f}}}'+r'}$')
p1 = axs[1].matshow(dtlminv,cmap='RdBu_r',vmin=-1.2,vmax=1.2)
fig.colorbar(p1,ax=axs[1],shrink=0.6,pad=0.01)
axs[1].set_title(r'$(\mathbf{M}^\mathrm{dyn}_{0,'+f'{{{ft:.0f}}}'+r'})^{-1}$')
fig.savefig(figdir/f'dtlm_ft{ft:.0f}.png')
plt.show(block=False)
plt.close()

floct = nst.rho[0,]
mloct = np.diag(floct)
mloc0 = dtlminv @ mloct @ dtlm
floc0 = np.diag(mloc0)
fig = plt.figure(figsize=[6,6],constrained_layout=True)
gs = GridSpec(nrows=3,ncols=2,figure=fig)
ax00 = fig.add_subplot(gs[0,0])
ax01 = fig.add_subplot(gs[0,1],sharey=ax00)
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
fig.suptitle(f'FT={ft:.0f}s')
fig.savefig(figdir/f'prop_rho_ft{ft:.0f}.png')
plt.show(block=False)
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
ax.set_title(f'FT={ft:.0f}s')
fig.savefig(figdir/f'test_prop_rho_ft{ft:.0f}.png')
plt.show(block=False)
plt.close()
#exit()

# ETLM
tint = 10.0
nint = int(tint / dt)
etlm = np.eye(nx)
etlmpinv = np.eye(nx)
fig, axs = plt.subplots(nrows=2,ncols=2,figsize=[8,6],constrained_layout=True)
Xi = np.zeros((nx,nens))
for k in range(nens):
    Xi[:,k] = est[k].rho[0,:]
    axs[0,0].plot(ix,Xi[:,k],c='magenta',alpha=0.3,lw=1.0)
rhom = np.mean(Xi,axis=1)
axs[0,0].plot(ix,rhom,c='k',ls='dashed',lw=3.0,label='ens mean')
axs[0,0].plot(ix,rhoi,c='k',lw=3.0,label='truth')
Xi = Xi - rhom[:,None]
u, s, vt = la.svd(Xi,full_matrices=False)
axs[0,1].semilogy(np.arange(1,s.size+1),s*s,marker='o')
i = np.argmin(np.abs((s*s)-(s[0]*s[0]*1e-4)))
axs[0,1].vlines([i+1],0,1,colors='k',ls='dashed',transform=axs[0,1].get_xaxis_transform())
axs[0,1].set_title(r'eigenvalues of $X_i^\mathrm{T} X_i$')
for j in range(nft):
    Xj = np.zeros((nx,nens))
    for k in range(nens):
        model(est[k])
        Xj[:,k] = est[k].rho[0,:]
    if (j+1)%nint==0:
        rhom = np.mean(Xj,axis=1)
        Xj = Xj - rhom[:,None]
        cmat = Xi.T @ Xi
        cjmat = Xj.T @ Xj
        #beta = 0.3 * nens * 1.0e-8 * la.norm(cmat,ord=2)
        #u, s, vt = la.svd(cmat+beta*np.eye(cmat.shape[0]))
        #zmat = u @ np.diag(1.0/s) @ u.T @ Xi.T
        zmat = la.pinv(cmat, rcond=1e-4, hermitian=True) @ Xi.T
        etlmtmp = Xj @ zmat
        etlm = etlmtmp @ etlm
        zimat = la.pinv(cjmat, rcond=1e-4, hermitian=True) @ Xj.T
        eitlmtmp = Xi @ zimat
        etlmpinv = etlmpinv @ eitlmtmp
        Xi = Xj.copy()
for k in range(nens):
    Xj[:,k] = est[k].rho[0,:]
    axs[1,0].plot(ix,Xj[:,k],c='magenta',alpha=0.3,lw=1.0)
rhom = np.mean(Xj,axis=1)
axs[1,0].plot(ix,rhom,c='k',ls='dashed',lw=3.0,label='ens mean')
axs[1,0].plot(ix,nst.rho[0,],c='k',lw=3.0,label='truth')
axs[1,0].plot(ix,etlm@rhoi,c='gray',ls='dashed',lw=3.0,label=r'$\mathbf{M}^\mathrm{ens}\mathbf{\phi}_0$')
Xj = Xj - rhom[:,None]
u, s, vt = la.svd(Xj,full_matrices=False)
axs[1,1].semilogy(np.arange(1,s.size+1),s*s,marker='o')
i = np.argmin(np.abs((s*s)-(s[0]*s[0]*1e-4)))
axs[1,1].vlines([i+1],0,1,colors='k',ls='dashed',transform=axs[1,1].get_xaxis_transform())
axs[1,1].set_title(r'eigenvalues of $X_j^\mathrm{T} X_j$')
axs[0,0].legend(title='t=0')
axs[1,0].legend(title=f't={ft}')
fig.suptitle(r'$\mathbf{\phi}^\mathrm{ens}$, '+f'K={nens}')
fig.savefig(figdir/f'phi_ft{ft:.0f}_k{nens}ni{nint}.png')
plt.show(block=False)
plt.close()

fig, axs = plt.subplots(ncols=2,figsize=[6,4],constrained_layout=True)
p0 = axs[0].matshow(etlm,cmap='RdBu_r') #,vmin=-1.2,vmax=1.2)
fig.colorbar(p0,ax=axs[0],shrink=0.6,pad=0.01)
axs[0].set_title(r'$\mathbf{M}^\mathrm{ens}_{0,'+f'{{{ft:.0f}}}'+r'}$')
p1 = axs[1].matshow(etlmpinv,cmap='RdBu_r') #,vmin=-1.2,vmax=1.2)
fig.colorbar(p1,ax=axs[1],shrink=0.6,pad=0.01)
axs[1].set_title(r'$(\mathbf{M}^\mathrm{ens}_{0,'+f'{{{ft:.0f}}}'+r'})^\dagger$')
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s')
fig.savefig(figdir/f'etlm_ft{ft:.0f}_k{nens}ni{nint}.png')
plt.show(block=False)
plt.close()

floct = nst.rho[0,]
mloct = np.diag(floct)
mloc0 = etlmpinv @ mloct @ etlm
floc0 = np.diag(mloc0)
fig = plt.figure(figsize=[6,6],constrained_layout=True)
gs = GridSpec(nrows=3,ncols=2,figure=fig)
ax00 = fig.add_subplot(gs[0,0])
ax01 = fig.add_subplot(gs[0,1],sharey=ax00)
ax10 = fig.add_subplot(gs[1:,0])
ax11 = fig.add_subplot(gs[1:,1])
ax01.plot(ix,floct)
ax01.set_title(r'$\rho_t$')
p11=ax11.matshow(mloct)
fig.colorbar(p11,ax=ax11,shrink=0.6,pad=0.01)
ax11.set_title(r'$\mathrm{diag}(\rho_t)$')
p10=ax10.matshow(mloc0)
fig.colorbar(p10,ax=ax10,shrink=0.6,pad=0.01)
ax10.set_title(r'$\mathrm{diag}(\rho_0)=(\mathbf{M}^\mathrm{ens}_{0,t})^\dagger\mathrm{diag}(\rho_t)\mathbf{M}^\mathrm{ens}_{0,t}$')
ax00.plot(ix,floc0)
ax00.set_title(r'$\rho_0$')
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s')
fig.savefig(figdir/f'prop_rho_ft{ft:.0f}_k{nens}ni{nint}.png')
plt.show(block=False)
plt.close()

y = rng.normal(loc=0.0,scale=0.5,size=nx)
z = dtlm @ y
rhoy = floc0 * y
mrhoy = dtlm @ rhoy
merhoy = etlm @ rhoy
rhoz = floct * z
fig, ax = plt.subplots()
ax.plot(ix,rhoz,label=r'$\rho_t \circ \mathbf{z}$')
ax.plot(ix,mrhoy,ls='dashed',label=r'$\mathbf{M}_{0,t}(\rho_0\circ\mathbf{y})$')
ax.plot(ix,merhoy,ls='dotted',label=r'$\mathbf{M}^\mathrm{ens}_{0,t}(\rho_0\circ\mathbf{y})$')
ax.legend()
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s')
fig.savefig(figdir/f'test_prop_rho_ft{ft:.0f}_k{nens}ni{nint}.png')
plt.show(block=False)
plt.close()