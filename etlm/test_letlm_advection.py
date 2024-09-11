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
# patch settings
lpatch = 30
npatch = nx // lpatch
lhalo = 20
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

tmax = 100.0
ntmax = int(tmax / dt)
for i in range(ntmax):
    nature(nst)
    for k in range(nens):
        model(est[k])

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

# LETLM
tint = 10.0
nint = int(tint / dt)
letlm = [np.eye(lletlm[i]) for i in range(npatch)]
letlmpinv = [np.eye(lletlm[i]) for i in range(npatch)]
fig, axs = plt.subplots(nrows=2,ncols=2,figsize=[8,6],constrained_layout=True)
Xi = np.zeros((nx,nens))
for k in range(nens):
    Xi[:,k] = est[k].rho[0,:]
    axs[0,0].plot(ix,Xi[:,k],c='magenta',alpha=0.3,lw=1.0)
rhom = np.mean(Xi,axis=1)
axs[0,0].plot(ix,rhom,c='k',ls='dashed',lw=3.0,label='ens mean')
axs[0,0].plot(ix,rhoi,c='k',lw=3.0,label='truth')
Xi = Xi - rhom[:,None]
Xilist = [Xi]
#u, s, vt = la.svd(Xi,full_matrices=False)
#axs[0,1].semilogy(np.arange(1,s.size+1),s*s,marker='o')
#i = np.argmin(np.abs((s*s)-(s[0]*s[0]*1e-4)))
#axs[0,1].vlines([i+1],0,1,colors='k',ls='dashed',transform=axs[0,1].get_xaxis_transform())
axs[0,1].set_title(r'eigenvalues of $X_i^\mathrm{T} X_i$')
for j in range(nft):
    Xj = np.zeros((nx,nens))
    for k in range(nens):
        model(est[k])
        Xj[:,k] = est[k].rho[0,:]
    if (j+1)%nint==0:
        rhom = np.mean(Xj,axis=1)
        Xj = Xj - rhom[:,None]
        for l in range(npatch):
            Xip = np.roll(Xi,-ishalo[l],axis=0)[:lletlm[l]]
            Xjp = np.roll(Xj,-ishalo[l],axis=0)[:lletlm[l]]
            cmat = Xip.T @ Xip
            cjmat = Xjp.T @ Xjp
            ## Tikhonov regularization
            beta = 0.1 # 0.3 * nens * 1.0e-8 * la.norm(cmat,ord=2)
            cmat = cmat + beta*np.eye(cmat.shape[0])
            beta = 0.1 # 0.3 * nens * 1.0e-8 * la.norm(cmat,ord=2)
            cjmat = cjmat + beta*np.eye(cjmat.shape[0])
            ##
            if j+1==nint:
                v, s, vt = la.svd(cmat, hermitian=True)
                i = np.argmin(np.abs(s-(s[0]*1e-4)))
                axs[0,1].semilogy(np.arange(1,s.size+1),s,marker='o')
                axs[0,1].vlines([i+1],0,1,colors='k',ls='dashed',transform=axs[0,1].get_xaxis_transform())
            if j+1==nft:
                v, s, vt = la.svd(cjmat, hermitian=True)
                i = np.argmin(np.abs(s-(s[0]*1e-4)))
                axs[1,1].semilogy(np.arange(1,s.size+1),s,marker='o')
                axs[1,1].vlines([i+1],0,1,colors='k',ls='dashed',transform=axs[1,1].get_xaxis_transform())
            #u, s, vt = la.svd(cmat+beta*np.eye(cmat.shape[0]))
            #zmat = u @ np.diag(1.0/s) @ u.T @ Xi.T
            zmat = la.pinv(cmat, rcond=1e-4, hermitian=True) @ Xip.T
            letlmtmp = Xjp @ zmat
            letlm[l] = letlmtmp @ letlm[l]
            zimat = la.pinv(cjmat, rcond=1e-4, hermitian=True) @ Xjp.T
            leitlmtmp = Xip @ zimat
            letlmpinv[l] = letlmpinv[i] @ leitlmtmp
        Xilist.append(Xj)
        Xi = Xj.copy()
print(f"len(Xilist)={len(Xilist)}")
for k in range(nens):
    Xj[:,k] = est[k].rho[0,:]
    axs[1,0].plot(ix,Xj[:,k],c='magenta',alpha=0.3,lw=1.0)
rhom = np.mean(Xj,axis=1)
axs[1,0].plot(ix,rhom,c='k',ls='dashed',lw=3.0,label='ens mean')
axs[1,0].plot(ix,nst.rho[0,],c='k',lw=3.0,label='truth')
etlm = np.eye(nx)
etlmpinv = np.eye(nx)
rhot = rhoi.copy()
for l in range(npatch):
    i0 = ispatch[l]
    i1 = iepatch[l]+1
    ii0 = ishalo[l]
    ii1 = iehalo[l]+1
    rhot[i0:i1] = (letlm[l]@np.roll(rhoi,-ii0)[:lletlm[l]])[lhalo:-lhalo]
    etlm[i0:i1,i0:i1] = letlm[l][lhalo:-lhalo,lhalo:-lhalo]
    etlmpinv[i0:i1,i0:i1] = letlmpinv[l][lhalo:-lhalo,lhalo:-lhalo]
axs[1,0].plot(ix,rhot,c='gray',ls='dashed',lw=3.0,label=r'$\mathbf{M}^\mathrm{LETLM}\mathbf{\phi}_0$')
Xj = Xj - rhom[:,None]
#u, s, vt = la.svd(Xj,full_matrices=False)
#axs[1,1].semilogy(np.arange(1,s.size+1),s*s,marker='o')
#i = np.argmin(np.abs((s*s)-(s[0]*s[0]*1e-4)))
#axs[1,1].vlines([i+1],0,1,colors='k',ls='dashed',transform=axs[1,1].get_xaxis_transform())
axs[1,1].set_title(r'eigenvalues of $X_j^\mathrm{T} X_j$')
axs[0,0].legend(title='t=0')
axs[1,0].legend(title=f't={ft}')
fig.suptitle(r'$\mathbf{\phi}^\mathrm{ens}$, '+f'K={nens} lpatch={lpatch} lhalo={lhalo}')
fig.savefig(figdir/f'phi_ft{ft:.0f}_k{nens}ni{nint}_letlm_lp{lpatch}lh{lhalo}.png')
plt.show(block=False)
plt.close()

fig, axs = plt.subplots(ncols=2,figsize=[6,4],constrained_layout=True)
p0 = axs[0].matshow(etlm,cmap='RdBu_r') #,vmin=-1.2,vmax=1.2)
fig.colorbar(p0,ax=axs[0],shrink=0.6,pad=0.01)
axs[0].set_title(r'$\mathbf{M}^\mathrm{LETLM}_{0,'+f'{{{ft:.0f}}}'+r'}$')
p1 = axs[1].matshow(etlmpinv,cmap='RdBu_r') #,vmin=-1.2,vmax=1.2)
fig.colorbar(p1,ax=axs[1],shrink=0.6,pad=0.01)
axs[1].set_title(r'$(\mathbf{M}^\mathrm{LETLM}_{0,'+f'{{{ft:.0f}}}'+r'})^\dagger$')
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s lpatch={lpatch} lhalo={lhalo}')
fig.savefig(figdir/f'letlm_lp{lpatch}lh{lhalo}_ft{ft:.0f}_k{nens}ni{nint}.png')
plt.show(block=False)
plt.close()

fig, axs = plt.subplots(ncols=2,nrows=2,figsize=[6,6],constrained_layout=True)
p0 = axs[0,0].matshow(etlm@etlmpinv@etlm,cmap='RdBu_r')
fig.colorbar(p0,ax=axs[0,0],shrink=0.6,pad=0.01)
axs[0,0].set_title(r'$\mathbf{M}^\mathrm{LETLM}_{0,'+f'{{{ft:.0f}}}'+r'}(\mathbf{M}^\mathrm{LETLM}_{0,'+f'{{{ft:.0f}}}'+r'})^\dagger\mathbf{M}^\mathrm{LETLM}_{0,'+f'{{{ft:.0f}}}'+r'}$')
p1 = axs[0,1].matshow(etlmpinv@etlm@etlmpinv,cmap='RdBu_r')
fig.colorbar(p1,ax=axs[0,1],shrink=0.6,pad=0.01)
axs[0,1].set_title(r'$(\mathbf{M}^\mathrm{LETLM}_{0,'+f'{{{ft:.0f}}}'+r'})^\dagger\mathbf{M}^\mathrm{LETLM}_{0,'+f'{{{ft:.0f}}}'+r'}(\mathbf{M}^\mathrm{LETLM}_{0,'+f'{{{ft:.0f}}}'+r'})^\dagger$')
p2 = axs[1,0].matshow(etlmpinv@etlm,cmap='RdBu_r') #,vmin=-1.2,vmax=1.2)
fig.colorbar(p2,ax=axs[1,0],shrink=0.6,pad=0.01)
axs[1,0].set_title(r'$(\mathbf{M}^\mathrm{LETLM}_{0,'+f'{{{ft:.0f}}}'+r'})^\dagger\mathbf{M}^\mathrm{LETLM}_{0,'+f'{{{ft:.0f}}}'+r'}$')
p3 = axs[1,1].matshow(etlm@etlmpinv,cmap='RdBu_r') #,vmin=-1.2,vmax=1.2)
fig.colorbar(p3,ax=axs[1,1],shrink=0.6,pad=0.01)
axs[1,1].set_title(r'$\mathbf{M}^\mathrm{LETLM}_{0,'+f'{{{ft:.0f}}}'+r'}(\mathbf{M}^\mathrm{LETLM}_{0,'+f'{{{ft:.0f}}}'+r'})^\dagger$')
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s')
fig.savefig(figdir/f'letlm_check_lp{lpatch}lh{lhalo}_ft{ft:.0f}_k{nens}ni{nint}.png')
plt.show(block=False)
plt.close()

## propagation of localization function (i) GC
floct = nst.rho[0,]
mloct = np.diag(floct)
mloc0 = np.zeros_like(mloct)
for l in range(npatch):
    i0 = ispatch[l]
    i1 = iepatch[l]+1
    ii0 = ishalo[l]
    ii1 = iehalo[l]+1
    mlocttmp = np.roll(np.roll(mloct, -ii0, axis=0), -ii0, axis=1)
    mloc0[i0:i1,i0:i1] = (letlmpinv[l]@mlocttmp[:lletlm[l],:lletlm[l]]@letlm[l])[lhalo:-lhalo,lhalo:-lhalo]
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
ax10.set_title(r'$\mathrm{diag}(\rho_0)=(\mathbf{M}^\mathrm{LETLM}_{0,t})^\dagger\mathrm{diag}(\rho_t)\mathbf{M}^\mathrm{LETLM}_{0,t}$')
ax00.plot(ix,floc0)
scale = 1.0 / np.max(floc0)
ax00.plot(ix,floc0*scale,ls='dotted')
ax00.set_title(r'$\rho_0$')
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s lpatch={lpatch} lhalo={lhalo}')
fig.savefig(figdir/f'prop_rho_ft{ft:.0f}_k{nens}ni{nint}_letlm_lp{lpatch}lh{lhalo}.png')
plt.show(block=False)
plt.close()

cmap=plt.get_cmap('viridis')
### recursive propagation
floc0r = floct.copy()
nr = len(Xilist)-1
fig, ax = plt.subplots(figsize=[4,3],constrained_layout=True)
ax.plot(ix,floc0r,c=cmap(0),alpha=0.7,ls='dashed',label='component')
icol = 16
for ir in range(nr):
    Xj = Xilist[nr-ir]
    Xi = Xilist[nr-ir-1]
    for l in range(npatch):
        Xip = np.roll(Xi,-ishalo[l],axis=0)[:lletlm[l]]
        Xjp = np.roll(Xj,-ishalo[l],axis=0)[:lletlm[l]]
        cimat = Xip.T @ Xip
        cjmat = Xjp.T @ Xjp
        ## Tikhonov regularization
        beta = 0.1 #0.3 * nens * 1.0e-8 * la.norm(cmat,ord=2)
        cimat = cimat + beta*np.eye(cimat.shape[0])
        beta = 0.1 #0.3 * nens * 1.0e-8 * la.norm(cjmat,ord=2)
        cjmat = cjmat + beta*np.eye(cjmat.shape[0])
        ##
        # localized region
        floc0rp = np.roll(floc0r,-ishalo[l])[:lletlm[l]]
        #iloc = np.arange(lletlm[l])[np.abs(floc0rp)>1.0e-2]
        ##print(f"iloc={iloc}")
        #Xjl = floc0rp[iloc][:,None] * Xjp[iloc]
        #cjlmat = Xjp[iloc].T @ Xjl
        Xjl = floc0rp[:,None] * Xjp 
        cjlmat = Xjp.T @ Xjl
        cmat = la.pinv(cjmat, rcond=1e-4, hermitian=True) @ cjlmat @ la.pinv(cimat, rcond=1e-4, hermitian=True)
        ii=lhalo
        for i in range(ispatch[l],iepatch[l]+1):
            floc0r[i] = np.dot(Xip[ii],np.dot(cmat,Xip[ii]))
            ii+=1
    floc0r = floc0r / np.max(floc0r) # scaling
    ax.plot(ix,floc0r,c=cmap(icol*(ir+1)),ls='dashed',alpha=0.7)
ax.plot(ix,floc0r,c=cmap(icol*nr),ls='dashed')
ax.legend()
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s')
fig.savefig(figdir/f'prop_rho_recursive_ft{ft:.0f}_k{nens}ni{nint}_letlm_lp{lpatch}lh{lhalo}.png')
plt.show(block=False)
plt.close()

y = rng.normal(loc=0.0,scale=0.5,size=nx)
z = dtlm @ y
rhoy = floc0 * y
rhoy2 = floc0 * y * scale
mrhoy = dtlm @ rhoy
mrhoy2 = dtlm @ rhoy2
merhoy = np.zeros_like(mrhoy)
merhoy2 = np.zeros_like(mrhoy2)
for l in range(npatch):
    i0 = ispatch[l]
    i1 = iepatch[l]+1
    ii0 = ishalo[l]
    ii1 = iehalo[l]+1
    merhoy[i0:i1] = (letlm[l] @ np.roll(rhoy,-ii0)[:lletlm[l]])[lhalo:-lhalo]
    merhoy2[i0:i1] = (letlm[l] @ np.roll(rhoy2,-ii0)[:lletlm[l]])[lhalo:-lhalo]
rhoz = floct * z
fig, ax = plt.subplots()
cmap=plt.get_cmap('tab10')
ax.plot(ix,rhoz,label=r'$\rho_t \circ \mathbf{z}$')
ax.plot(ix,mrhoy,c=cmap(1),label=r'$\mathbf{M}_{0,t}(\rho_0\circ\mathbf{y})$')
ax.plot(ix,mrhoy2,c=cmap(1),ls='dotted',label=r'$\mathbf{M}_{0,t}(\rho_0^*\circ\mathbf{y})$')
ax.plot(ix,merhoy,c=cmap(2),label=r'$\mathbf{M}^\mathrm{ens}_{0,t}(\rho_0\circ\mathbf{y})$')
ax.plot(ix,merhoy2,c=cmap(2),ls='dotted',label=r'$\mathbf{M}^\mathrm{ens}_{0,t}(\rho_0^*\circ\mathbf{y})$')
ax.legend()
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s lpatch={lpatch} lhalo={lhalo}')
fig.savefig(figdir/f'test_prop_rho_ft{ft:.0f}_k{nens}ni{nint}_letlm_lp{lpatch}lh{lhalo}.png')
plt.show(block=False)
plt.close()

## propagation of localization function (ii) Boxcar
L = dx*20
ic = ix[nx//4]
floct = np.where(np.abs(ix-ic)<2*L,1.0,0.0)
mloct = np.diag(floct)
mloc0 = np.zeros_like(mloct)
for l in range(npatch):
    i0 = ispatch[l]
    i1 = iepatch[l]+1
    ii0 = ishalo[l]
    ii1 = iehalo[l]+1
    mlocttmp = np.roll(np.roll(mloct, -ii0, axis=0), -ii0, axis=1)
    mloc0[i0:i1,i0:i1] = (letlmpinv[l]@mlocttmp[:lletlm[l],:lletlm[l]]@letlm[l])[lhalo:-lhalo,lhalo:-lhalo]
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
ax10.set_title(r'$\mathrm{diag}(\rho_0)=(\mathbf{M}^\mathrm{LETLM}_{0,t})^\dagger\mathrm{diag}(\rho_t)\mathbf{M}^\mathrm{LETLM}_{0,t}$')
ax00.plot(ix,floc0)
scale = 1.0 / np.max(floc0)
ax00.plot(ix,floc0*scale,ls='dotted')
ax00.set_title(r'$\rho_0$')
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s lpatch={lpatch} lhalo={lhalo}')
fig.savefig(figdir/f'prop_bc_ft{ft:.0f}_k{nens}ni{nint}_letlm_lp{lpatch}lh{lhalo}.png')
plt.show(block=False)
plt.close()

cmap=plt.get_cmap('viridis')
### recursive propagation
floc0r = floct.copy()
nr = len(Xilist)-1
fig, ax = plt.subplots(figsize=[4,3],constrained_layout=True)
ax.plot(ix,floc0r,c=cmap(0),alpha=0.7,ls='dashed',label='component')
icol = 16
for ir in range(nr):
    Xj = Xilist[nr-ir]
    Xi = Xilist[nr-ir-1]
    for l in range(npatch):
        Xip = np.roll(Xi,-ishalo[l],axis=0)[:lletlm[l]]
        Xjp = np.roll(Xj,-ishalo[l],axis=0)[:lletlm[l]]
        cimat = Xip.T @ Xip
        cjmat = Xjp.T @ Xjp
        ## Tikhonov regularization
        beta = 0.1 #0.3 * nens * 1.0e-8 * la.norm(cmat,ord=2)
        cimat = cimat + beta*np.eye(cimat.shape[0])
        beta = 0.1 #0.3 * nens * 1.0e-8 * la.norm(cjmat,ord=2)
        cjmat = cjmat + beta*np.eye(cjmat.shape[0])
        ##
        # localized region
        floc0rp = np.roll(floc0r,-ishalo[l])[:lletlm[l]]
        #iloc = np.arange(lletlm[l])[np.abs(floc0rp)>1.0e-2]
        ##print(f"iloc={iloc}")
        #Xjl = floc0rp[iloc][:,None] * Xjp[iloc]
        #cjlmat = Xjp[iloc].T @ Xjl
        Xjl = floc0rp[:,None] * Xjp 
        cjlmat = Xjp.T @ Xjl
        cmat = la.pinv(cjmat, rcond=1e-4, hermitian=True) @ cjlmat @ la.pinv(cimat, rcond=1e-4, hermitian=True)
        ii=lhalo
        for i in range(ispatch[l],iepatch[l]+1):
            floc0r[i] = np.dot(Xip[ii],np.dot(cmat,Xip[ii]))
            ii+=1
    floc0r = floc0r / np.max(floc0r) # scaling
    ax.plot(ix,floc0r,c=cmap(icol*(ir+1)),ls='dashed',alpha=0.7)
ax.plot(ix,floc0r,c=cmap(icol*nr),ls='dashed')
ax.legend()
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s')
fig.savefig(figdir/f'prop_bc_recursive_ft{ft:.0f}_k{nens}ni{nint}_letlm_lp{lpatch}lh{lhalo}.png')
plt.show(block=False)
plt.close()

y = rng.normal(loc=0.0,scale=0.5,size=nx)
z = dtlm @ y
rhoy = floc0 * y
rhoy2 = floc0 * y * scale
mrhoy = dtlm @ rhoy
mrhoy2 = dtlm @ rhoy2
merhoy = np.zeros_like(mrhoy)
merhoy2 = np.zeros_like(mrhoy2)
for l in range(npatch):
    i0 = ispatch[l]
    i1 = iepatch[l]+1
    ii0 = ishalo[l]
    ii1 = iehalo[l]+1
    merhoy[i0:i1] = (letlm[l] @ np.roll(rhoy,-ii0)[:lletlm[l]])[lhalo:-lhalo]
    merhoy2[i0:i1] = (letlm[l] @ np.roll(rhoy2,-ii0)[:lletlm[l]])[lhalo:-lhalo]
rhoz = floct * z
fig, ax = plt.subplots()
cmap=plt.get_cmap('tab10')
ax.plot(ix,rhoz,label=r'$\rho_t \circ \mathbf{z}$')
ax.plot(ix,mrhoy,c=cmap(1),label=r'$\mathbf{M}_{0,t}(\rho_0\circ\mathbf{y})$')
ax.plot(ix,mrhoy2,c=cmap(1),ls='dotted',label=r'$\mathbf{M}_{0,t}(\rho_0^*\circ\mathbf{y})$')
ax.plot(ix,merhoy,c=cmap(2),label=r'$\mathbf{M}^\mathrm{ens}_{0,t}(\rho_0\circ\mathbf{y})$')
ax.plot(ix,merhoy2,c=cmap(2),ls='dotted',label=r'$\mathbf{M}^\mathrm{ens}_{0,t}(\rho_0^*\circ\mathbf{y})$')
ax.legend()
fig.suptitle(f'FT={ft:.0f}s K={nens} Tint={tint:.1f}s lpatch={lpatch} lhalo={lhalo}')
fig.savefig(figdir/f'test_prop_bc_ft{ft:.0f}_k{nens}ni{nint}_letlm_lp{lpatch}lh{lhalo}.png')
plt.show(block=False)
plt.close()
