import numpy as np 
from numpy.random import default_rng
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 16
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
import sys 
sys.path.append('../analysis')
from corrfunc import Corrfunc
from pathlib import Path

figdir = Path('errcov')
if not figdir.exists():
    figdir.mkdir(parents=True)

rng = default_rng()
nsample = 100000
nst = 50
nstl = 25
## l05nest geometry
nglb = 960
nst = 240
nstl = 24
intl = nst / nstl

mtype="gc5"
sigb = 0.8
lb = 28.77
a_b = -0.11
dist_b = np.eye(nst)
for i in range(nst):
    for j in range(nst):
        dist_b[i,j] = 2.0*np.pi*np.abs(i-j)/nglb
corrfunc_b = Corrfunc(np.deg2rad(lb),a=a_b)
sigv = 0.8
lv = 12.03
a_v = 0.12
dist_v = np.eye(nstl)
for i in range(nstl):
    for j in range(nstl):
        dist_v[i,j] = 2.0*np.pi*np.abs(i-j)*intl/nglb
corrfunc_v = Corrfunc(np.deg2rad(lv),a=a_v)
if mtype=="diag":
    B = np.eye(nst)*sigb*sigb
    V = np.eye(nstl)*sigv*sigv
    Bsqrt = np.sqrt(B)
    Vsqrt = np.sqrt(V)
    figtitle = f'{mtype}_sigb{sigb}v{sigv}'
else:
    B = np.eye(nst)
    V = np.eye(nstl)
    if mtype=="gauss":
        for i in range(nst):
            B[i,] = corrfunc_b(dist_b[i,],ftype=mtype)
        for i in range(nstl):
            V[i,] = corrfunc_v(dist_v[i,],ftype=mtype)
        figtitle = f'{mtype}_sigb{sigb}v{sigv}lb{lb}v{lv}'
    elif mtype=="gc5":
        for i in range(nst):
            if i < nst-i:
                ctmp = corrfunc_b(np.roll(dist_b[i,],-i)[:nst-i],ftype=mtype)
                ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:-1])])
                B[i,] = np.roll(ctmp2,i)[:nst]
            else:
                ctmp = corrfunc_b(np.flip(dist_b[i,:i+1]),ftype=mtype)
                ctmp2 = np.hstack([np.flip(ctmp),ctmp[1:-1]])
                B[i,] = ctmp2[:nst]
        for i in range(nstl):
            if i < nstl-i:
                ctmp = corrfunc_v(np.roll(dist_v[i,],-i)[:nstl-i],ftype=mtype)
                ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:])])
                V[i,] = np.roll(ctmp2,i)[:nstl]
            else:
                ctmp = corrfunc_v(np.flip(dist_v[i,:i+1]),ftype=mtype)
                ctmp2 = np.hstack([np.flip(ctmp),ctmp[1:]])
                V[i,] = ctmp2[:nstl]
        figtitle = f'l05nest_{mtype}ab{a_b}v{a_v}_sigb{sigb}v{sigv}lb{lb}v{lv}'
    B = B * sigb * sigb
    V = V * sigv * sigv
    eival, eivec = np.linalg.eigh(B)
    Bsqrt = np.dot(eivec,np.diag(np.sqrt(eival)))
    eival, eivec = np.linalg.eigh(V)
    Vsqrt = np.dot(eivec,np.diag(np.sqrt(eival)))

fig, axs = plt.subplots(nrows=2,figsize=[4,8],constrained_layout=True)
vlim = max(sigb*sigb,sigv*sigv)
mps = []
mp00 = axs[0].matshow(B,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp00)
axs[0].set_title(r'$\mathbf{B}$')
mp01 = axs[1].matshow(V,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp01)
axs[1].set_title(r'$\mathbf{V}$')
for mp, ax in zip(mps,axs):
    fig.colorbar(mp,ax=ax,pad=0.01,shrink=0.6)
fig.savefig(figdir/f'{figtitle}.png',dpi=300)
plt.show()

Bs = np.zeros_like(B)
Vs = np.zeros_like(V)
Ekb = np.zeros((nstl,nst))
Ebk = np.zeros((nst,nstl))
for i in range(nsample):
    epsb = np.dot(Bsqrt,rng.normal(0.0,scale=1.0,size=nst))
    epsv = np.dot(Vsqrt,rng.normal(0.0,scale=1.0,size=nstl))
    Bs = Bs + np.outer(epsb,epsb)
    Vs = Vs + np.outer(epsv,epsv)
    Ekb = Ekb + np.outer(epsv,epsb)
    Ebk = Ebk + np.outer(epsb,epsv)
Bs = Bs / float(nsample)
Vs = Vs / float(nsample)
Ekb = Ekb / float(nsample)
Ebk = Ebk / float(nsample)

Binv = np.linalg.inv(B)
Vinv = np.linalg.inv(V)
Bsinv = np.linalg.inv(Bs)
Vsinv = np.linalg.inv(Vs)
B1 = Bs - np.dot(Ebk,np.dot(Vsinv,Ekb))
V1 = Vs - np.dot(Ekb,np.dot(Bsinv,Ebk))
B1inv = np.linalg.inv(B1)
V1inv = np.linalg.inv(V1)
Ekb1 = -1.0 * np.dot(Vsinv,np.dot(Ekb,B1inv))
Ebk1 = -1.0 * np.dot(Bsinv,np.dot(Ebk,V1inv))

fig, axs = plt.subplots(nrows=2,ncols=2,figsize=[8,8],constrained_layout=True)
vlim = max(np.max(Binv),np.max(Bsinv))
mps = []
mp00 = axs[0,0].matshow(Binv,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp00)
axs[0,0].set_title(r'$\mathbf{B}^{-1}$')
mp01 = axs[0,1].matshow(Bsinv,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp01)
axs[0,1].set_title(r'$\langle \mathbf{\varepsilon}^\mathrm{b}\mathbf{\varepsilon}^\mathrm{bT}\rangle^{-1}$')
vlim = max(np.max(Vinv),np.max(Vsinv))
mp10 = axs[1,0].matshow(Vinv,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp10)
axs[1,0].set_title(r'$\mathbf{V}^{-1}$')
mp11 = axs[1,1].matshow(Vsinv,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp11)
axs[1,1].set_title(r'$\langle \mathbf{\varepsilon}^\mathrm{v}\mathbf{\varepsilon}^\mathrm{vT}\rangle^{-1}$')
for mp, ax in zip(mps,axs.flatten()):
    fig.colorbar(mp,ax=ax,pad=0.01,shrink=0.6)
fig.savefig(figdir/f'{figtitle}.inv.png',dpi=300)
plt.show()

fig = plt.figure(figsize=[10,6],constrained_layout=True)
vlim = max(sigb*sigb,sigv*sigv)
gs = GridSpec(nrows=3,ncols=4,figure=fig)
axs = []
ax = fig.add_subplot(gs[:2,:2])
axs.append(ax)
ax = fig.add_subplot(gs[2,:2])
axs.append(ax)
ax = fig.add_subplot(gs[:2,2])
axs.append(ax)
ax = fig.add_subplot(gs[2,2])
axs.append(ax)
mps = []
mp00 = axs[0].matshow(Bs,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp00)
axs[0].set_title(r'$\langle \mathbf{\varepsilon}^\mathrm{b}\mathbf{\varepsilon}^\mathrm{bT}\rangle$')
mp10 = axs[1].matshow(Ekb,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp10)
axs[1].set_title(r'$\langle \mathbf{\varepsilon}^\mathrm{v}\mathbf{\varepsilon}^\mathrm{bT}\rangle$')
mp01 = axs[2].matshow(Ebk,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp01)
axs[2].set_title(r'$\langle \mathbf{\varepsilon}^\mathrm{b}\mathbf{\varepsilon}^\mathrm{vT}\rangle$')
mp11 = axs[3].matshow(Vs,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp11)
axs[3].set_title(r'$\langle \mathbf{\varepsilon}^\mathrm{v}\mathbf{\varepsilon}^\mathrm{vT}\rangle$')
for mp, ax in zip(mps,axs):
    fig.colorbar(mp,ax=ax,pad=0.01,shrink=0.6)
axs[1].set_aspect(5.0)
axs[2].set_aspect(0.2)
fig.suptitle(f'nsample={nsample}')
fig.savefig(figdir/f'{figtitle}.sample.png',dpi=300)
plt.show()

fig = plt.figure(figsize=[10,6],constrained_layout=True)
gs = GridSpec(nrows=3,ncols=4,figure=fig)
axs = []
ax = fig.add_subplot(gs[:2,:2])
axs.append(ax)
ax = fig.add_subplot(gs[2,:2])
axs.append(ax)
ax = fig.add_subplot(gs[:2,2])
axs.append(ax)
ax = fig.add_subplot(gs[2,2])
axs.append(ax)
mps = []
vlim = max(np.max(B1inv),-np.min(B1inv))
mp00 = axs[0].matshow(B1inv,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp00)
axs[0].set_title(r'$\mathbf{B}_1^{-1}=(\mathbf{B}-\mathbf{E}_\mathrm{bk}\mathbf{V}^{-1}\mathbf{E}_\mathrm{kb})^{-1}$')
vlim = max(np.max(Ekb1),-np.min(Ekb1))
mp10 = axs[1].matshow(Ekb1,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp10)
axs[1].set_title(r'$-\mathbf{V}^{-1}\mathbf{E}_\mathrm{kb}\mathbf{B}_1^{-1}$')
vlim = max(np.max(Ebk1),-np.min(Ebk1))
mp01 = axs[2].matshow(Ebk1,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp01)
axs[2].set_title(r'$-\mathbf{B}^{-1}\mathbf{E}_\mathrm{bk}\mathbf{V}_1^{-1}$')
vlim = max(np.max(V1inv),-np.min(V1inv))
mp11 = axs[3].matshow(V1inv,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp11)
axs[3].set_title(r'$\mathbf{V}_1^{-1}=(\mathbf{V}-\mathbf{E}_\mathrm{kb}\mathbf{B}^{-1}\mathbf{E}_\mathrm{bk})^{-1}$')
for mp, ax in zip(mps,axs):
    fig.colorbar(mp,ax=ax,pad=0.01,shrink=0.6)
axs[1].set_aspect(5.0)
axs[2].set_aspect(0.2)
fig.suptitle(f'nsample={nsample}')
fig.savefig(figdir/f'{figtitle}.sample_inv.png',dpi=300)
plt.show()

I00 = np.dot(Bs,B1inv) + np.dot(Ebk,Ekb1)
I01 = np.dot(Bs,Ebk1) + np.dot(Ebk,V1inv)
I10 = np.dot(Ekb,B1inv) + np.dot(Vs,Ekb1)
I11 = np.dot(Ekb,Ebk1) + np.dot(Vs,V1inv)
fig = plt.figure(figsize=[10,6],constrained_layout=True)
gs = GridSpec(nrows=3,ncols=4,figure=fig)
axs = []
ax = fig.add_subplot(gs[:2,:2])
axs.append(ax)
ax = fig.add_subplot(gs[2,:2])
axs.append(ax)
ax = fig.add_subplot(gs[:2,2])
axs.append(ax)
ax = fig.add_subplot(gs[2,2])
axs.append(ax)
mps = []
vlim = 1.0
mp00 = axs[0].matshow(I00,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp00)
#axs[0].set_title(r'$\mathbf{B}_1^{-1}=(\mathbf{B}-\mathbf{E}_\mathrm{bk}\mathbf{V}^{-1}\mathbf{E}_\mathrm{kb})^{-1}$')
mp10 = axs[1].matshow(I10,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp10)
#axs[1].set_title(r'$-\mathbf{V}^{-1}\mathbf{E}_\mathrm{kb}\mathbf{B}_1^{-1}$')
mp01 = axs[2].matshow(I01,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp01)
#axs[2].set_title(r'$-\mathbf{B}^{-1}\mathbf{E}_\mathrm{bk}\mathbf{V}_1^{-1}$')
mp11 = axs[3].matshow(I11,cmap='bwr',norm=Normalize(vmin=-vlim,vmax=vlim))
mps.append(mp11)
#axs[3].set_title(r'$\mathbf{V}_1^{-1}=(\mathbf{V}-\mathbf{E}_\mathrm{kb}\mathbf{B}^{-1}\mathbf{E}_\mathrm{bk})^{-1}$')
for mp, ax in zip(mps,axs):
    fig.colorbar(mp,ax=ax,pad=0.01,shrink=0.6)
axs[1].set_aspect(5.0)
axs[2].set_aspect(0.2)
fig.suptitle(f'nsample={nsample}, '+r'$\mathbf{W}\mathbf{W}^{-1}$')
#fig.savefig(figdir/f'{figtitle}_inv.png',dpi=300)
plt.show()
