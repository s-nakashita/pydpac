import numpy as np 
import scipy.linalg as la
from numpy.random import default_rng
import matplotlib.pyplot as plt

rng = default_rng()

ntrial=10
nb=200
nv=100
ratio=np.arange(1,12)*0.1
reslist = []
for r in ratio:
    nm = int((nb+nv)*r)
    print(f"nm={nm}")
    resmean = 0.0
    resstd = 0.0
    for itrial in range(ntrial):
        print(f"trial {itrial+1}")
        # control vector
        chi = rng.normal(loc=0.0,scale=0.5,size=nm)
        # innovation vector
        d = rng.normal(loc=0.0,scale=0.7,size=(nb+nv))
        #d[:nb] = 0.0
        # square root of error cov.
        lbmat = rng.normal(loc=0.0,scale=2.0,size=(nb,nm))
        lvmat = rng.normal(loc=0.0,scale=1.0,size=(nv,nm))
        lmat = np.concatenate((lbmat,lvmat),axis=0)
        #print(lbmat.shape,lvmat.shape,lmat.shape)
        lpinv = la.pinv(lmat)
        #print(lpinv.shape)
        #print(lpinv@lmat)
        wmat = np.dot(lmat,lmat.T)
        winv = la.pinv(wmat)
        winv2=np.dot(lpinv.T,lpinv)
        fig, axs = plt.subplots(ncols=3,figsize=(12,5))
        p0=axs[0].matshow(wmat)
        fig.colorbar(p0,ax=axs[0],shrink=0.6,pad=0.001)
        axs[0].set_title(r"$\mathbf{W}=\mathbf{L}\mathbf{L}^\mathrm{T}$")
        p1=axs[1].matshow(winv2)
        fig.colorbar(p1,ax=axs[1],shrink=0.6,pad=0.001)
        axs[1].set_title(r"$(\mathbf{L}^{\dagger})^\mathrm{T}\mathbf{L}^{\dagger}$")
        diff=winv2@wmat-np.eye(wmat.shape[0])
        p2=axs[2].matshow(winv2@wmat)
        fig.colorbar(p2,ax=axs[2],shrink=0.6,pad=0.001)
        axs[2].set_title(r"$\|\{(\mathbf{L}^{\dagger})^\mathrm{T}\mathbf{L}^{\dagger}\}\mathbf{W}-\mathbf{I}\|=$"+f"{np.linalg.norm(diff,ord='fro'):.3e}")
        fig.suptitle(f"ndim={d.size} nens={chi.size}")
        plt.show(block=False)
        plt.close()
        # J1=[d-0.5*(I,I)^T@x]^T(W)^{-1}[d-0.5*(I,I)^T@x]
        x = lmat@chi
        J1=np.dot((d-x),np.dot(winv,(d-x)))
        # J2=[lpinv@d-chi]^T[lpinv@d-chi]
        J2=np.dot((lpinv@d-chi),(lpinv@d-chi))
        print(f"J1={J1:.4f} J2={J2:.4f} relative error={(J1-J2)/(J1+J2)/2.0*100.0:.3f}%")
        #
        fig, axs = plt.subplots(ncols=2,figsize=[8,4],constrained_layout=True)
        u, s, vt = la.svd(lmat)
        print(f"u.shape={u.shape} s.shape={s.shape} v.shape={vt.transpose().shape}")
        nsig = s.size
        rk = rng.normal(loc=0.0,scale=5.0,size=(nb+nv))
        pk = np.dot(np.dot(lmat,lmat.T),rk)
        axs[0].plot(rk,label=r'$\mathbf{r}^k$')
        axs[1].plot(pk,label=r'$\mathbf{p}^k=\mathbf{L}\mathbf{L}^\mathrm{T}\mathbf{r}^k$')
        laminv = 1.0 / s / s
        rk2 = np.dot(np.dot(np.dot(u[:,:nsig],np.diag(laminv)),u[:,:nsig].transpose()),pk)
        pk2 = np.dot(np.dot(lmat,lmat.T),rk2)
        axs[0].plot(rk2,label=r"$\tilde{\mathbf{r}^k}=(\mathbf{L}\mathbf{L}^\mathrm{T})^{\dagger}\mathbf{p}^k$")
        axs[1].plot(pk2,label=r"$\tilde{\mathbf{p}^k}=\mathbf{L}\mathbf{L}^\mathrm{T}\tilde{\mathbf{r}^k}$")
        axs[0].plot(rk-rk2,label="res")
        fig.suptitle(f"ndim={pk.size} nens={nsig}")
        axs[0].legend()
        axs[1].legend()
        #if itrial==0: plt.show()
        plt.close()
        res=np.linalg.norm(rk-rk2,ord=2)/np.linalg.norm(rk,ord=2)
        resmean+=res
        resstd +=res**2
    resmean /= ntrial
    resstd = np.sqrt(resstd/ntrial - resmean**2)
    reslist.append([resmean,resstd])

res = np.array(reslist) * 100
print(res.shape)
fig, ax = plt.subplots()
ax.errorbar(ratio*100,res[:,0],yerr=res[:,1])
ax.set_xlabel(f"nens/ndim(={nb+nv}) (%)")
ax.set_ylabel("residual(%)")
plt.show()