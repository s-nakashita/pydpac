from lorenz3 import L05III
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
plt.rcParams['font.size'] = 16
from pathlib import Path

nx = 960
nk = 32
ni = 12
b = 10.0
c = 0.6
dt = 0.05 / 6 / b
F = 15.0
step = L05III(nx, nk, ni, b, c, dt, F)
## ensemble size
nens = 50
outdir=Path(f'lorenz/n{nx}k{nk}i{ni}F{int(F)}b{b:.1f}c{c:.1f}')
if not outdir.exists():
    outdir.mkdir()

x0c = np.ones(nx)
x0c[nx//2-1] += 0.001
for j in range(5000): # spin up
    x0c = step(x0c)
#ensemble
x0 = random.normal(0, scale=0.1, size=(nx,nens))
x0 = x0 + x0c[:,None]
#print(x0.std(axis=1))
for j in range(5000): # spin up
    x0 = step(x0)
#print(x0.std(axis=1))
for m in range(nens):
    plt.plot(np.arange(nx),x0[:,m],c='gray',alpha=0.5)
plt.plot(np.arange(nx),x0.mean(axis=1),c='blue',lw=3)
plt.ylim(-15.,15.)
plt.show(block=False)
plt.close()

nt = 100 * 24 * int(b) #[hours]
x = []
x_lr = []
x_hr = []
for i in range(nt):
    x0 = step(x0)
    if i%(6*int(b))==0:
        x0_lr, x0_hr = step.decomp(x0)
        x.append(x0)
        x_hr.append(x0_hr)
        x_lr.append(x0_lr)
    if i%2000==0:
        print(f"t={i/24.0/b:.1f}d")
#        for m in range(nens):
#            plt.plot(np.arange(nx),x0[:,m],c='gray',alpha=0.5)
#        plt.plot(np.arange(nx),x0.mean(axis=1),c='blue',lw=3)
#        plt.ylim(-15.,15.)
#        plt.show()
#        plt.close()
x = np.array(x)
x_lr = np.array(x_lr)
x_hr = np.array(x_hr)
print(x_lr.shape)
print(x_hr.shape)

time = np.arange(0,nt,6*int(b)) / 24.0 / b
xaxis = np.arange(nx)
X, T = np.meshgrid(xaxis,time)
print(X.shape)
fig, axs = plt.subplots(ncols=2,figsize=[12,8],sharey=True,constrained_layout=True)
p0 = axs[0].pcolormesh(X,T,x_lr[:,:,0],shading='auto',cmap='coolwarm',\
    vmin=-12.,vmax=12.)
fig.colorbar(p0,ax=axs[0],pad=0.01,shrink=0.6)
p1 = axs[1].pcolormesh(X,T,x_hr[:,:,0]*20.0,shading='auto',cmap='coolwarm',\
    vmin=-12.,vmax=12.)
fig.colorbar(p1,ax=axs[1],pad=0.01,shrink=0.6)
axs[0].set_xlabel("location")
axs[1].set_xlabel("location")
axs[0].set_ylabel("time(days)")
axs[0].set_title("large-scale")
axs[1].set_title(r"small-scale ($\times$20)")
fig.savefig(outdir/f"hov_lr+hr.png",dpi=300)
plt.show()
plt.close()

fig, axs = plt.subplots(ncols=2,figsize=[12,8],sharey=True,constrained_layout=True)
p0 = axs[0].pcolormesh(X,T,x_lr.std(axis=2),shading='auto')
fig.colorbar(p0,ax=axs[0],pad=0.01,shrink=0.6)
p1 = axs[1].pcolormesh(X,T,x_hr.std(axis=2)*20.0,shading='auto')
fig.colorbar(p1,ax=axs[1],pad=0.01,shrink=0.6)
axs[0].set_xlabel("location")
axs[1].set_xlabel("location")
axs[0].set_ylabel("time(days)")
axs[0].set_title("large-scale")
axs[1].set_title(r"small-scale ($\times$20)")
fig.savefig(outdir/f"hov_sprd_lr+hr.png",dpi=300)
plt.show()
plt.close()

# covariance
x = x - x.mean(axis=2)[:,:,None]
x_hr = x_hr - x_hr.mean(axis=2)[:,:,None]
x_lr = x_lr - x_lr.mean(axis=2)[:,:,None]
Bful = np.zeros((nx,nx))
B_hr = np.zeros((nx,nx))
B_lr = np.zeros((nx,nx))
E_lh = np.zeros((nx,nx))
E_hl = np.zeros((nx,nx))
nts = x_hr.shape[0] // 3
nte = x_hr.shape[0]
nsample = 0
for i in range(nts,nte):
    bmat = np.dot(x[i,:,:],x[i,:,:].T)/float(nens-1)
    Bful = Bful + bmat
    bmat = np.dot(x_hr[i,:,:],x_hr[i,:,:].T)/float(nens-1)
    B_hr = B_hr + bmat
    bmat = np.dot(x_lr[i,:,:],x_lr[i,:,:].T)/float(nens-1)
    B_lr = B_lr + bmat
    b_lh = np.dot(x_lr[i,:,:],x_hr[i,:,:].T)/float(nens-1)
    E_lh = E_lh + b_lh
    b_hl = np.dot(x_hr[i,:,:],x_lr[i,:,:].T)/float(nens-1)
    E_hl = E_hl + b_hl
    nsample += 1
Bful = Bful / nsample
B_hr = B_hr / nsample
B_lr = B_lr / nsample
E_lh = E_lh / nsample
E_hl = E_hl / nsample
fig = plt.figure(figsize=[12,8],constrained_layout=True)
gs = GridSpec(2,3,figure=fig)
axs = []
ax = fig.add_subplot(gs[0,0])
axs.append(ax)
ax = fig.add_subplot(gs[0,1])
axs.append(ax)
ax = fig.add_subplot(gs[1,0])
axs.append(ax)
ax = fig.add_subplot(gs[1,1])
axs.append(ax)
ax = fig.add_subplot(gs[0,2])
axs.append(ax)
p00 = axs[0].pcolormesh(xaxis,xaxis,B_hr*400.0,shading='auto')
fig.colorbar(p00,ax=axs[0],pad=0.01,shrink=0.6)
p01 = axs[1].pcolormesh(xaxis,xaxis,E_hl*20.0,shading='auto')
fig.colorbar(p01,ax=axs[1],pad=0.01,shrink=0.6)
p10 = axs[2].pcolormesh(xaxis,xaxis,E_lh*20.0,shading='auto')
fig.colorbar(p10,ax=axs[2],pad=0.01,shrink=0.6)
p11 = axs[3].pcolormesh(xaxis,xaxis,B_lr,shading='auto')
fig.colorbar(p11,ax=axs[3],pad=0.01,shrink=0.6)
p20 = axs[4].pcolormesh(xaxis,xaxis,Bful,shading='auto')
fig.colorbar(p20,ax=axs[4],pad=0.01,shrink=0.6)
for ax in axs:
    ax.set_xlabel("location")
    ax.set_xlabel("location")
#axs[0].set_aspect("equal")
#axs[1].set_aspect(nx_hr/nx_lr)
#axs[2].set_aspect(nx_lr/nx_hr)
#axs[3].set_aspect("equal")
axs[0].set_title(r"small-scale $\times$ small-scale ($\times 20^2$)")
axs[1].set_title(r"small-scale $\times$ large-scale ($\times 20$)")
axs[2].set_title(r"large-scale $\times$ small-scale ($\times 20$)")
axs[3].set_title(r"large-scale $\times$ large-scale")
axs[4].set_title("full field")
fig.savefig(outdir/f"crosscov_lr+hr.png",dpi=300)
plt.show()

np.savetxt(outdir/"ix.txt",xaxis)
np.save(outdir/f"B.npy",Bful)
np.save(outdir/f"B_hr.npy",B_hr)
np.save(outdir/f"B_lr.npy",B_lr)
np.save(outdir/f"E_hl.npy",E_hl)
np.save(outdir/f"E_lh.npy",E_lh)
