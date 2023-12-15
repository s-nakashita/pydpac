from lorenz_nest import L05nest
import numpy as np
from numpy import random
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['font.size'] = 16
from pathlib import Path
import sys

nx_true = 960
nx_lam  = 480
nx_gm   = 240
intgm   = 4
nk_lam  = 32
nk_gm   = 8
ni = 12
b = 10.0
c = 0.6
dt = 0.05 / 36.0
F = 15.0
ist_lam = 240
nsp = 30
intrlx = 6
lamstep = 1
po = 1
if len(sys.argv)>1:
    po = int(sys.argv[1])
step = L05nest(nx_true, nx_gm, nx_lam, nk_gm, nk_lam, ni, b, c, dt, F,\
    intgm, ist_lam, nsp, lamstep=lamstep, intrlx=intrlx, po=po)
## ensemble size
nens = 720
outdir=Path(f'lorenz/ng{nx_gm}nl{nx_lam}kg{nk_gm}kl{nk_lam}nsp{nsp}p{po}F{int(F)}b{b:.1f}c{c:.1f}')
if not outdir.exists():
    outdir.mkdir()

x0c_gm = np.ones(nx_gm)*F
x0c_gm[nx_gm//2-1] += 0.001*F
for j in range(7200): # spin up
    x0c_gm = step.gm(x0c_gm)
#GM ensemble
x0_gm = random.normal(0, scale=0.1, size=(nx_gm,nens))
x0_gm = x0_gm + x0c_gm[:,None]
#print(x0.std(axis=1))
for j in range(5000): # spin up
    x0_gm = step.gm(x0_gm)
#print(x0.std(axis=1))
for m in range(nens):
    plt.plot(step.ix_gm,x0_gm[:,m],c='gray',alpha=0.5)
plt.plot(step.ix_gm,x0_gm.mean(axis=1),c='blue',lw=3)
plt.ylim(-15.,15.)
plt.show(block=False)
plt.close()

#LAM control & ensemble
gm2lam = interp1d(step.ix_gm, x0c_gm)
x0c_lam = gm2lam(step.ix_lam)
gm2lam = interp1d(step.ix_gm,x0_gm,axis=0)
x0_lam = gm2lam(step.ix_lam)
for m in range(nens):
    plt.plot(step.ix_lam,x0_lam[:,m],c='gray',alpha=0.5)
plt.plot(step.ix_lam,x0_lam.mean(axis=1),c='blue',lw=3)
plt.ylim(-15.,15.)
plt.show(block=False)
plt.close()

nt = 100 * 4
x_gm  = []
x_lam = []
time = []
for i in range(nt):
    x0_gm, x0_lam = step(x0_gm,x0_lam)
    time.append(i * 6) #[hours]
    x_lam.append(x0_lam)
    x_gm.append(x0_gm)
    if i%40==0:
        print(f"t={i/int(0.05 / 6 / dt)/24.0:.1f}d")
        for m in range(nens):
            plt.plot(step.ix_gm,x0_gm[:,m],c='gray',ls='dashed',alpha=0.5)
            plt.plot(step.ix_lam,x0_lam[:,m],c='k',alpha=0.5)
        plt.plot(step.ix_gm,x0_gm.mean(axis=1),c='blue',lw=3)
        plt.plot(step.ix_lam,x0_lam.mean(axis=1),c='green',lw=3)
        plt.ylim(-15.,15.)
        plt.show(block=False)
        plt.close()
x_gm = np.array(x_gm)
x_lam = np.array(x_lam)
print(x_gm.shape)
print(x_lam.shape)

time = np.array(time) / 24.0 #[days]
X_gm, T_gm = np.meshgrid(step.ix_gm,time)
X_lam, T_lam = np.meshgrid(step.ix_lam,time)
print(X_gm.shape)
print(X_lam.shape)
print("GM  mean={:.3f}".format(x_gm[:,:,0].mean()))
print("LAM mean={:.3f}".format(x_lam[:,:,0].mean()))
fig, axs = plt.subplots(ncols=2,figsize=[12,8],sharey=True,constrained_layout=True)
p0 = axs[0].pcolormesh(X_gm,T_gm,x_gm[:,:,0],shading='auto',cmap='coolwarm',\
    vmin=-12.,vmax=12.)
fig.colorbar(p0,ax=axs[0],pad=0.01,shrink=0.6)
p1 = axs[1].pcolormesh(X_lam,T_lam,x_lam[:,:,0],shading='auto',cmap='coolwarm',\
    vmin=-12.,vmax=12.)
fig.colorbar(p1,ax=axs[1],pad=0.01,shrink=0.6)
axs[0].vlines([step.ix_lam[0],step.ix_lam[-1]],0,1,\
    colors='k',linestyle='dashdot',transform=axs[0].get_xaxis_transform())
axs[1].vlines([step.ix_lam[nsp],step.ix_lam[-nsp]],0,1,\
    colors='white',linestyle='dashdot',transform=axs[1].get_xaxis_transform())
axs[0].set_xlabel("location")
axs[1].set_xlabel("location")
axs[0].set_ylabel("time(days)")
axs[0].set_title("GM")
axs[1].set_title("LAM")
fig.savefig(outdir/"hov_gm+lam.png",dpi=300)
plt.show(block=False)
plt.close()

fig, axs = plt.subplots(ncols=2,figsize=[12,8],sharey=True,constrained_layout=True)
p0 = axs[0].pcolormesh(X_gm,T_gm,x_gm.std(axis=2),shading='auto')
fig.colorbar(p0,ax=axs[0],pad=0.01,shrink=0.6)
p1 = axs[1].pcolormesh(X_lam,T_lam,x_lam.std(axis=2),shading='auto')
fig.colorbar(p1,ax=axs[1],pad=0.01,shrink=0.6)
axs[0].vlines([step.ix_lam[0],step.ix_lam[-1]],0,1,\
    colors='k',linestyle='dashdot',transform=axs[0].get_xaxis_transform())
axs[1].vlines([step.ix_lam[nsp],step.ix_lam[-nsp]],0,1,\
    colors='white',linestyle='dashdot',transform=axs[1].get_xaxis_transform())
axs[0].set_xlabel("location")
axs[1].set_xlabel("location")
axs[0].set_ylabel("time(days)")
axs[0].set_title("GM")
axs[1].set_title("LAM")
fig.savefig(outdir/"hov_sprd_gm+lam.png",dpi=300)
plt.show(block=False)
plt.close()

# covariance
x_gm  = x_gm - x_gm.mean(axis=2)[:,:,None]
x_lam = x_lam - x_lam.mean(axis=2)[:,:,None]
B_lam = np.zeros((nx_lam,nx_lam))
B_gmfull = np.zeros((nx_gm,nx_gm))
B_gm  = np.zeros((nx_lam,nx_lam))
E_gl  = np.zeros((nx_lam,nx_lam))
E_lg  = np.zeros((nx_lam,nx_lam))
nts = x_lam.shape[0] // 3
nte = x_lam.shape[0]
nsample = 0
for i in range(nts,nte):
    bmat = np.dot(x_lam[i,:,:],x_lam[i,:,:].T)/float(nens-1)
    B_lam = B_lam + bmat
    bmat = np.dot(x_gm[i,:,:],x_gm[i,:,:].T)/float(nens-1)
    B_gmfull = B_gmfull + bmat
    gm2lam = interp1d(step.ix_gm,x_gm[i,:,:],axis=0)
    xtmp = gm2lam(step.ix_lam)
    bmat = np.dot(xtmp,xtmp.T)/float(nens-1)
    B_gm = B_gm + bmat
    b_gl = np.dot(xtmp,x_lam[i,:,:].T)/float(nens-1)
    E_gl = E_gl + b_gl
    b_lg = np.dot(x_lam[i,:,:],xtmp.T)/float(nens-1)
    E_lg = E_lg + b_lg
    nsample += 1
B_lam = B_lam / nsample
B_gmfull = B_gmfull / nsample
B_gm = B_gm / nsample
E_gl = E_gl / nsample
E_lg = E_lg / nsample
fig = plt.figure(figsize=[12,12],constrained_layout=True)
gs = GridSpec(2,2,figure=fig)
axs = []
ax = fig.add_subplot(gs[0,0])
axs.append(ax)
ax = fig.add_subplot(gs[0,1])
axs.append(ax)
ax = fig.add_subplot(gs[1,0])
axs.append(ax)
ax = fig.add_subplot(gs[1,1])
axs.append(ax)
p00 = axs[0].pcolormesh(step.ix_lam,step.ix_lam,B_lam,shading='auto')
fig.colorbar(p00,ax=axs[0],pad=0.01,shrink=0.6)
p01 = axs[1].pcolormesh(step.ix_lam,step.ix_lam,E_lg,shading='auto')
fig.colorbar(p01,ax=axs[1],pad=0.01,shrink=0.6)
p10 = axs[2].pcolormesh(step.ix_lam,step.ix_lam,E_gl,shading='auto')
fig.colorbar(p10,ax=axs[2],pad=0.01,shrink=0.6)
p11 = axs[3].pcolormesh(step.ix_lam,step.ix_lam,B_gm,shading='auto')
fig.colorbar(p11,ax=axs[3],pad=0.01,shrink=0.6)
for ax in axs:
    ax.set_xlabel("location")
    ax.set_xlabel("location")
#axs[0].set_aspect("equal")
#axs[1].set_aspect(nx_lam/nx_gm)
#axs[2].set_aspect(nx_gm/nx_lam)
#axs[3].set_aspect("equal")
axs[0].set_title(r"$\mathbf{X}^\mathrm{b}_\mathrm{LAM}(\mathbf{X}^\mathrm{b}_\mathrm{LAM})^\mathrm{T}$")
axs[1].set_title(r"$\mathbf{X}^\mathrm{b}_\mathrm{LAM}(\mathbf{H}_1\mathbf{X}^\mathrm{b}_\mathrm{GM})^\mathrm{T}$")
axs[2].set_title(r"$\mathbf{H}_1\mathbf{X}^\mathrm{b}_\mathrm{GM}(\mathbf{X}^\mathrm{b}_\mathrm{LAM})^\mathrm{T}$")
axs[3].set_title(r"$\mathbf{H}_1\mathbf{X}^\mathrm{b}_\mathrm{GM}(\mathbf{H}_1\mathbf{X}^\mathrm{b}_\mathrm{GM})^\mathrm{T}$")
fig.savefig(outdir/"crosscov_gm+lam.png",dpi=300)
plt.show(block=False)
plt.close()
np.savetxt(outdir/"ix_true.txt",step.ix_true)
np.savetxt(outdir/"ix_gm.txt",step.ix_gm)
np.savetxt(outdir/"ix_lam.txt",step.ix_lam)
np.save(outdir/"B_lam.npy",B_lam)
np.save(outdir/"B_gm.npy",B_gm)
np.save(outdir/"B_gmfull.npy",B_gmfull)
np.save(outdir/"E_lg.npy",E_lg)
np.save(outdir/"E_gl.npy",E_gl)
