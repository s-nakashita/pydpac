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

figdir = Path('test_advection_reg0.1')
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
nst = Advection_state(nx, u0=u0, xnu0=xnu0, F0=F0)
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
    st = Advection_state(nx, u0=ue, xnu0=xnu0, F0=F0)
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
dtlmlist = []
for j in range(nft):
    nature(nst)
    dtlmtmp = np.eye(nx)
    for i in range(nx):
#        prtb.rho[0,:] = dtlm[:,i]
        prtb.rho[0,:] = dtlmtmp[:,i]
        nature.step_t(nst,prtb)
#        dtlm[:,i] = prtb.rho[0,:]
        dtlmtmp[:,i] = prtb.rho[0,:]
    dtlmlist.append(dtlmtmp)
    dtlm = dtlmtmp @ dtlm
fig, ax = plt.subplots(figsize=[6,4],constrained_layout=True)
ax.plot(ix,rhoi,label=r'$\mathbf{\phi}_{0}$')
ax.plot(ix,nst.rho[0,],label=r"$\mathbf{\phi}"+f"_{{{ft:.0f}}}$")
ax.plot(ix,dtlm@rhoi,ls='dashed',label=r'$\mathbf{M}_{0,'+f'{{{ft:.0f}}}'+r'}\mathbf{\phi}_0$')
ax.legend()
fig.savefig(figdir/f'phi_ft{ft:.0f}.png')
plt.show(block=False)
plt.close()
#exit()

#dtlminv = la.inv(dtlm)
dtlminv = np.eye(nx)
for i in range(len(dtlmlist)):
    dtlmtmp = dtlmlist[i]
    dtlminv = dtlminv @ la.inv(dtlmtmp)
fig, axs = plt.subplots(ncols=2,figsize=[6,4],constrained_layout=True)
p0 = axs[0].matshow(dtlm,cmap='RdBu_r',vmin=-1.2,vmax=1.2)
fig.colorbar(p0,ax=axs[0],shrink=0.6,pad=0.01)
axs[0].set_title(r'$\mathbf{M}^\mathrm{dyn}_{0,'+f'{{{ft:.0f}}}'+r'}$')
p1 = axs[1].matshow(dtlminv,cmap='RdBu_r',vmin=-1.2,vmax=1.2)
fig.colorbar(p1,ax=axs[1],shrink=0.6,pad=0.01)
axs[1].set_title(r'$(\mathbf{M}^\mathrm{dyn}_{0,'+f'{{{ft:.0f}}}'+r'})^{-1}$')
fig.suptitle(f'FT={ft:.0f}s')
fig.savefig(figdir/f'dtlm_ft{ft:.0f}.png')
plt.show(block=False)
plt.close()

fig, axs = plt.subplots(ncols=2,nrows=2,figsize=[6,6],constrained_layout=True)
p0 = axs[0,0].matshow(dtlm@dtlminv@dtlm,cmap='RdBu_r',vmin=-1.2,vmax=1.2)
fig.colorbar(p0,ax=axs[0,0],shrink=0.6,pad=0.01)
axs[0,0].set_title(r'$\mathbf{M}^\mathrm{dyn}_{0,'+f'{{{ft:.0f}}}'+r'}(\mathbf{M}^\mathrm{dyn}_{0,'+f'{{{ft:.0f}}}'+r'})^{-1}\mathbf{M}^\mathrm{dyn}_{0,'+f'{{{ft:.0f}}}'+r'}$')
p1 = axs[0,1].matshow(dtlminv@dtlm@dtlminv,cmap='RdBu_r',vmin=-1.2,vmax=1.2)
fig.colorbar(p1,ax=axs[0,1],shrink=0.6,pad=0.01)
axs[0,1].set_title(r'$(\mathbf{M}^\mathrm{dyn}_{0,'+f'{{{ft:.0f}}}'+r'})^{-1}\mathbf{M}^\mathrm{dyn}_{0,'+f'{{{ft:.0f}}}'+r'}(\mathbf{M}^\mathrm{dyn}_{0,'+f'{{{ft:.0f}}}'+r'})^{-1}$')
p2 = axs[1,0].matshow(dtlminv@dtlm,cmap='RdBu_r',vmin=-1.2,vmax=1.2)
fig.colorbar(p2,ax=axs[1,0],shrink=0.6,pad=0.01)
axs[1,0].set_title(r'$(\mathbf{M}^\mathrm{dyn}_{0,'+f'{{{ft:.0f}}}'+r'})^{-1}\mathbf{M}^\mathrm{dyn}_{0,'+f'{{{ft:.0f}}}'+r'}$')
p3 = axs[1,1].matshow(dtlm@dtlminv,cmap='RdBu_r',vmin=-1.2,vmax=1.2)
fig.colorbar(p3,ax=axs[1,1],shrink=0.6,pad=0.01)
axs[1,1].set_title(r'$\mathbf{M}^\mathrm{dyn}_{0,'+f'{{{ft:.0f}}}'+r'}(\mathbf{M}^\mathrm{dyn}_{0,'+f'{{{ft:.0f}}}'+r'})^{-1}$')
fig.suptitle(f'FT={ft:.0f}s')
fig.savefig(figdir/f'dtlm_check_ft{ft:.0f}.png')
plt.show(block=False)
plt.close()

## propagation of localization function (i) GC
floct = nst.rho[0,]
mloct = np.diag(floct)
mloc0 = mloct.copy()
for i in range(len(dtlmlist)):
    dtlmtmp = dtlmlist[len(dtlmlist)-i-1]
    mloc0 = la.inv(dtlmtmp) @ mloc0 @ dtlmtmp
#mloc0 = dtlminv @ mloct @ dtlm
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
rhoy = floc0 * y
z = y.copy()
mrhoy = rhoy.copy()
for i in range(len(dtlmlist)):
    dtlmtmp = dtlmlist[i]
    z = dtlmtmp @ z
    mrhoy = dtlmtmp @ mrhoy
#z = dtlm @ y
#mrhoy = dtlm @ rhoy
rhoz = floct * z
fig, ax = plt.subplots()
ax.plot(ix,rhoz,label=r'$\rho_t \circ \mathbf{z}$')
ax.plot(ix,mrhoy,ls='dashed',label=r'$\mathbf{M}_{0,t}(\rho_0\circ\mathbf{y})$')
ax.legend()
ax.set_title(f'FT={ft:.0f}s')
fig.savefig(figdir/f'test_prop_rho_ft{ft:.0f}.png')
plt.show(block=False)
plt.close()

## propagation of localization function (ii) Boxcar
L = dx*20
ic = ix[nx//4]
floct = np.where(np.abs(ix-ic)<2*L,1.0,0.0)
mloct = np.diag(floct)
mloc0 = mloct.copy()
for i in range(len(dtlmlist)):
    dtlmtmp = dtlmlist[len(dtlmlist)-i-1]
    mloc0 = la.inv(dtlmtmp) @ mloc0 @ dtlmtmp
#mloc0 = dtlminv @ mloct @ dtlm
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
fig.savefig(figdir/f'prop_bc_ft{ft:.0f}.png')
plt.show(block=False)
plt.close()

rhoy = floc0 * y
z = y.copy()
mrhoy = rhoy.copy()
for i in range(len(dtlmlist)):
    dtlmtmp = dtlmlist[i]
    z = dtlmtmp @ z
    mrhoy = dtlmtmp @ mrhoy
#z = dtlm @ y
#mrhoy = dtlm @ rhoy
rhoz = floct * z
fig, ax = plt.subplots()
ax.plot(ix,rhoz,label=r'$\rho_t \circ \mathbf{z}$')
ax.plot(ix,mrhoy,ls='dashed',label=r'$\mathbf{M}_{0,t}(\rho_0\circ\mathbf{y})$')
ax.legend()
ax.set_title(f'FT={ft:.0f}s')
fig.savefig(figdir/f'test_prop_bc_ft{ft:.0f}.png')
plt.show(block=False)
plt.close()
exit()

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
#u, s, vt = la.svd(Xi,full_matrices=False)
#axs[0,1].semilogy(np.arange(1,s.size+1),s*s,marker='o')
#i = np.argmin(np.abs((s*s)-(s[0]*s[0]*1e-4)))
#axs[0,1].vlines([i+1],0,1,colors='k',ls='dashed',transform=axs[0,1].get_xaxis_transform())
axs[0,1].set_title(r'eigenvalues of $X_i^\mathrm{T} X_i$')
Xilist = [Xi]
etlmlist = []
etlmpinvlist = []
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
        ## Tikhonov regularization
        beta = 0.1 #0.3 * nens * 1.0e-8 * la.norm(cmat,ord=2)
        cmat = cmat + beta*np.eye(cmat.shape[0])
        beta = 0.1 #0.3 * nens * 1.0e-8 * la.norm(cjmat,ord=2)
        cjmat = cjmat + beta*np.eye(cjmat.shape[0])
        ##
        v, s, vt = la.svd(cmat, hermitian=True)
        i = np.argmin(np.abs(s-(s[0]*1e-4)))
        if j+1==nint:
            axs[0,1].semilogy(np.arange(1,s.size+1),s,marker='o')
            axs[0,1].vlines([i+1],0,1,colors='k',ls='dashed',transform=axs[0,1].get_xaxis_transform())
        zmat = la.pinv(cmat, rcond=1e-4, hermitian=True) @ Xi.T
        #ui, si, vit = la.svd(Xi,full_matrices=False)
        #zmat = v[:,:i] @ np.diag(1.0/np.sqrt(s[:i])) @ ui[:,:i].T 
        etlmtmp = Xj @ zmat
        etlmlist.append(etlmtmp)
        etlm = etlmtmp @ etlm
        zimat = la.pinv(cjmat, rcond=1e-4, hermitian=True) @ Xj.T
        #v, s, vt = la.svd(cjmat, hermitian=True)
        #i = np.argmin(np.abs(s-(s[0]*1e-4)))
        #uj, sj, vjt = la.svd(Xj,full_matrices=False)
        #zimat = v[:,:i] @ np.diag(1.0/np.sqrt(s[:i])) @ uj[:,:i].T 
        eitlmtmp = Xi @ zimat
        etlmpinvlist.append(eitlmtmp)
        etlmpinv = etlmpinv @ eitlmtmp
        Xilist.append(Xj)
        Xi = Xj.copy()
print(f"len(Xilist)={len(Xilist)}")
for k in range(nens):
    Xj[:,k] = est[k].rho[0,:]
    axs[1,0].plot(ix,Xj[:,k],c='magenta',alpha=0.3,lw=1.0)
rhom = np.mean(Xj,axis=1)
axs[1,0].plot(ix,rhom,c='k',ls='dashed',lw=3.0,label='ens mean')
axs[1,0].plot(ix,nst.rho[0,],c='k',lw=3.0,label='truth')
axs[1,0].plot(ix,etlm@rhoi,c='gray',ls='dashed',lw=3.0,label=r'$\mathbf{M}^\mathrm{ens}\mathbf{\phi}_0$')
#Xj = Xj - rhom[:,None]
#u, s, vt = la.svd(Xj,full_matrices=False)
#axs[1,1].semilogy(np.arange(1,s.size+1),s*s,marker='o')
#i = np.argmin(np.abs((s*s)-(s[0]*s[0]*1e-4)))
u, s, vt = la.svd(cjmat, hermitian=True)
axs[1,1].semilogy(np.arange(1,s.size+1),s,marker='o')
i = np.argmin(np.abs(s-(s[0]*1e-4)))
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

fig, axs = plt.subplots(ncols=2,nrows=2,figsize=[6,6],constrained_layout=True)
p0 = axs[0,0].matshow(etlm@etlmpinv@etlm,cmap='RdBu_r')
fig.colorbar(p0,ax=axs[0,0],shrink=0.6,pad=0.01)
axs[0,0].set_title(r'$\mathbf{M}^\mathrm{ens}_{0,'+f'{{{ft:.0f}}}'+r'}(\mathbf{M}^\mathrm{ens}_{0,'+f'{{{ft:.0f}}}'+r'})^\dagger\mathbf{M}^\mathrm{ens}_{0,'+f'{{{ft:.0f}}}'+r'}$')
p1 = axs[0,1].matshow(etlmpinv@etlm@etlmpinv,cmap='RdBu_r')
fig.colorbar(p1,ax=axs[0,1],shrink=0.6,pad=0.01)
axs[0,1].set_title(r'$(\mathbf{M}^\mathrm{ens}_{0,'+f'{{{ft:.0f}}}'+r'})^\dagger\mathbf{M}^\mathrm{ens}_{0,'+f'{{{ft:.0f}}}'+r'}(\mathbf{M}^\mathrm{ens}_{0,'+f'{{{ft:.0f}}}'+r'})^\dagger$')
p2 = axs[1,0].matshow(etlmpinv@etlm,cmap='RdBu_r') #,vmin=-1.2,vmax=1.2)
fig.colorbar(p2,ax=axs[1,0],shrink=0.6,pad=0.01)
axs[1,0].set_title(r'$(\mathbf{M}^\mathrm{ens}_{0,'+f'{{{ft:.0f}}}'+r'})^\dagger\mathbf{M}^\mathrm{ens}_{0,'+f'{{{ft:.0f}}}'+r'}$')
p3 = axs[1,1].matshow(etlm@etlmpinv,cmap='RdBu_r') #,vmin=-1.2,vmax=1.2)
fig.colorbar(p3,ax=axs[1,1],shrink=0.6,pad=0.01)
axs[1,1].set_title(r'$\mathbf{M}^\mathrm{ens}_{0,'+f'{{{ft:.0f}}}'+r'}(\mathbf{M}^\mathrm{ens}_{0,'+f'{{{ft:.0f}}}'+r'})^\dagger$')
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s')
fig.savefig(figdir/f'etlm_check_ft{ft:.0f}_k{nens}ni{nint}.png')
plt.show(block=False)
plt.close()

## propagation of localization function (i) GC
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
scale = 1.0 / np.max(floc0)
ax00.plot(ix,floc0*scale,ls='dotted')
ax00.set_title(r'$\rho_0$')
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s')
fig.savefig(figdir/f'prop_rho_ft{ft:.0f}_k{nens}ni{nint}.png')
plt.show(block=False)
plt.close()

cmap=plt.get_cmap('viridis')
### recursive propagation
floc0d = floct.copy() # initial=target
floc0r = floct.copy() # initial=target
nr = len(Xilist)-1
fig, ax = plt.subplots(figsize=[4,3],constrained_layout=True)
ax.plot(ix,floc0d,c=cmap(0),alpha=0.7,label='matmat')
ax.plot(ix,floc0r,c=cmap(0),alpha=0.7,ls='dashed',label='component')
icol = 16
for ir in range(nr):
    ftlm1 = etlmlist[nr-ir-1]
    itlm1 = etlmpinvlist[nr-ir-1]
    floc0d = np.diag(itlm1@np.diag(floc0d)@ftlm1)
    floc0d = floc0d / np.max(floc0d) # scaling
    #
    Xj = Xilist[nr-ir]
    Xi = Xilist[nr-ir-1]
    cimat = Xi.T @ Xi
    cjmat = Xj.T @ Xj
    ## Tikhonov regularization
    beta = 0.1 #0.3 * nens * 1.0e-8 * la.norm(cmat,ord=2)
    cimat = cimat + beta*np.eye(cimat.shape[0])
    beta = 0.1 #0.3 * nens * 1.0e-8 * la.norm(cjmat,ord=2)
    cjmat = cjmat + beta*np.eye(cjmat.shape[0])
    ##
    # localized region
    iloc = np.arange(nx)[np.abs(floc0r)>0.1]
    print(f"iloc={iloc}")
    Xjl = floc0r[iloc][:,None] * Xj[iloc]
    cjlmat = Xj[iloc].T @ Xjl
    #Xjl = floc0r[:,None] * Xj 
    #cjlmat = Xj.T @ Xjl
    cmat = la.pinv(cjmat, rcond=1e-4, hermitian=True) @ cjlmat @ la.pinv(cimat, rcond=1e-4, hermitian=True)
    for i in range(nx):
        floc0r[i] = np.dot(Xi[i],np.dot(cmat,Xi[i]))
    floc0r = floc0r / np.max(floc0r) # scaling
    ax.plot(ix,floc0d,c=cmap(icol*(ir+1)),alpha=0.7)
    ax.plot(ix,floc0r,c=cmap(icol*(ir+1)),ls='dashed',alpha=0.7)
ax.plot(ix,floc0d,c=cmap(icol*nr))
ax.plot(ix,floc0r,c=cmap(icol*nr),ls='dashed')
ax.legend()
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s')
fig.savefig(figdir/f'prop_rho_recursive_ft{ft:.0f}_k{nens}ni{nint}.png')
plt.show(block=False)
plt.close()

z = dtlm @ y
rhoy = floc0 * y
rhoy2 = floc0 * y * scale
mrhoy = dtlm @ rhoy
merhoy = etlm @ rhoy
mrhoy2 = dtlm @ rhoy2
merhoy2 = etlm @ rhoy2
rhoz = floct * z
fig, ax = plt.subplots()
cmap=plt.get_cmap('tab10')
ax.plot(ix,rhoz,label=r'$\rho_t \circ \mathbf{z}$')
ax.plot(ix,mrhoy,c=cmap(1),label=r'$\mathbf{M}_{0,t}(\rho_0\circ\mathbf{y})$')
ax.plot(ix,mrhoy2,c=cmap(1),ls='dotted',label=r'$\mathbf{M}_{0,t}(\rho_0^*\circ\mathbf{y})$')
ax.plot(ix,merhoy,c=cmap(2),label=r'$\mathbf{M}^\mathrm{ens}_{0,t}(\rho_0\circ\mathbf{y})$')
ax.plot(ix,merhoy2,c=cmap(2),ls='dotted',label=r'$\mathbf{M}^\mathrm{ens}_{0,t}(\rho_0^*\circ\mathbf{y})$')
ax.legend()
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s')
fig.savefig(figdir/f'test_prop_rho_ft{ft:.0f}_k{nens}ni{nint}.png')
plt.show(block=False)
plt.close()

## propagation of localization function (ii) Boxcar
L = dx*20
ic = ix[nx//4]
floct = np.where(np.abs(ix-ic)<2*L,1.0,0.0)
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
scale = 1.0 / np.max(floc0)
ax00.plot(ix,floc0*scale,ls='dotted')
ax00.set_title(r'$\rho_0$')
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s')
fig.savefig(figdir/f'prop_bc_ft{ft:.0f}_k{nens}ni{nint}.png')
plt.show(block=False)
plt.close()

cmap=plt.get_cmap('viridis')
### recursive propagation
floc0d = floct.copy() # initial=target
floc0r = floct.copy() # initial=target
fig, ax = plt.subplots(figsize=[4,3],constrained_layout=True)
ax.plot(ix,floc0d,c=cmap(0),label='matmat')
ax.plot(ix,floc0r,c=cmap(1),ls='dashed',label='component')
nr = len(Xilist)-1
icol = 16
for ir in range(nr):
    ftlm1 = etlmlist[nr-ir-1]
    itlm1 = etlmpinvlist[nr-ir-1]
    floc0d = np.diag(itlm1@np.diag(floc0d)@ftlm1)
    floc0d = floc0d / np.max(floc0d) # scaling
    #
    Xj = Xilist[nr-ir]
    Xi = Xilist[nr-ir-1]
    cimat = Xi.T @ Xi
    cjmat = Xj.T @ Xj
    ## Tikhonov regularization
    beta = 0.1 #0.3 * nens * 1.0e-8 * la.norm(cmat,ord=2)
    cimat = cimat + beta*np.eye(cimat.shape[0])
    beta = 0.1 #0.3 * nens * 1.0e-8 * la.norm(cjmat,ord=2)
    cjmat = cjmat + beta*np.eye(cjmat.shape[0])
    ##
    # localized region
    iloc = np.arange(nx)[np.abs(floc0r)>0.1]
    print(f"iloc={iloc}")
    Xjl = floc0r[iloc][:,None] * Xj[iloc]
    cjlmat = Xj[iloc].T @ Xjl
    #Xjl = floc0r[:,None] * Xj 
    #cjlmat = Xj.T @ Xjl
    cmat = la.pinv(cjmat, rcond=1e-4, hermitian=True) @ cjlmat @ la.pinv(cimat, rcond=1e-4, hermitian=True)
    for i in range(nx):
        floc0r[i] = np.dot(Xi[i],np.dot(cmat,Xi[i]))
    floc0r = floc0r / np.max(floc0r) # scaling
    ax.plot(ix,floc0d,c=cmap(icol*(ir+1)),alpha=0.7)
    ax.plot(ix,floc0r,c=cmap(icol*(ir+1)),ls='dashed',alpha=0.7)
ax.plot(ix,floc0d,c=cmap(icol*nr))
ax.plot(ix,floc0r,c=cmap(icol*nr),ls='dashed')
ax.legend()
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s')
fig.savefig(figdir/f'prop_bc_recursive_ft{ft:.0f}_k{nens}ni{nint}.png')
plt.show(block=False)
plt.close()

z = dtlm @ y
rhoy = floc0 * y
mrhoy = dtlm @ rhoy
merhoy = etlm @ rhoy
mrhoy2 = dtlm @ rhoy2
merhoy2 = etlm @ rhoy2
rhoz = floct * z
fig, ax = plt.subplots()
cmap=plt.get_cmap('tab10')
ax.plot(ix,rhoz,label=r'$\rho_t \circ \mathbf{z}$')
ax.plot(ix,mrhoy,c=cmap(1),label=r'$\mathbf{M}_{0,t}(\rho_0\circ\mathbf{y})$')
ax.plot(ix,mrhoy2,c=cmap(1),ls='dotted',label=r'$\mathbf{M}_{0,t}(\rho_0^*\circ\mathbf{y})$')
ax.plot(ix,merhoy,c=cmap(2),label=r'$\mathbf{M}^\mathrm{ens}_{0,t}(\rho_0\circ\mathbf{y})$')
ax.plot(ix,merhoy2,c=cmap(2),ls='dotted',label=r'$\mathbf{M}^\mathrm{ens}_{0,t}(\rho_0^*\circ\mathbf{y})$')
ax.legend()
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s')
fig.savefig(figdir/f'test_prop_bc_ft{ft:.0f}_k{nens}ni{nint}.png')
plt.show(block=False)
plt.close()