import numpy as np 
from numpy import linalg as la 
from numpy.random import default_rng

class SVSA():
    
    def __init__(self,n,tlm,adj,inorm=None,vnorm=None,seed=None,debug=False):
        self.n = n # state size
        self.tlm = tlm # Tangent linear model
        self.adj = adj # Adjoint model
        self.inorm = inorm # initial norm (square root)
        if self.inorm is None:
            self.inorm = np.eye(self.n)
        self.vnorm = vnorm # verification norm (square root)
        if self.vnorm is None:
            self.vnorm = np.eye(self.n)
        self.rng = default_rng(seed=seed)
        self.debug = debug

    def __call__(self,xb,nmode=1,analytical=False):
        if analytical:
            M = np.eye(self.n)
            alp=5.0e-3
            M = M * alp
            for i in range(self.n):
                for j in range(len(xb)):
                    mcol = self.tlm(xb[j],M[:,i])
                    M[:,i] = mcol
            M = M / alp
            u,s,vt = la.svd(self.vnorm@M)
            sv = la.solve(self.inorm,vt[:nmode,:].T)
            sv = sv.T
        else:
            sv = self.lanczos(xb,nmode=nmode)
        return sv
    
    def lanczos(self,xb,nmode=1,maxiter=200,thres=1.0e-6):
        sv = []
        for imode in range(nmode):
            v0 = self.rng.normal(0.0,scale=1.0,size=self.n)
            scale = 5.0e-3 / la.norm(v0,ord=2)
            v0 = v0 * scale
            niter=0
            while niter<maxiter:
                niter += 1
                if imode>0:
                    for jmode in range(imode):
                        smag = la.norm(sv[jmode],ord=2)
                        v0 = v0 - np.dot(v0,sv[jmode])/smag/smag*sv[jmode]
                v = la.solve(self.inorm,v0)
                # forward
                for j in range(len(xb)):
                    v = self.tlm(xb[j],v)
                v = self.vnorm @ v
                # backward
                v = self.vnorm.T @ v
                for j in range(len(xb)):
                    v = self.adj(xb[len(xb)-j-2],v)
                v = la.solve(self.inorm.T,v)
                scale = 5.0e-3 / la.norm(v,ord=2)
                v = v * scale
                d = la.norm(v-v0,ord=2)
                if d/5.0e-3 < thres:
                    print(f"lanczos converged at {niter}th iteration")
                    break
                if self.debug:
                    print(f"{niter}th iteration d={d:.3e}")
                v0 = v.copy()
            sv.append(v0/la.norm(v0,ord=2))
        return np.array(sv)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__),'../model'))
    from lorenz import L96

    rng = default_rng(515)
    nx = 40
    dt = 0.05
    F = 8.0
    model = L96(nx,dt,F)

    vnorm = np.zeros((nx,nx))
    for j in range(17,22):
        vnorm[j,j] = np.sqrt(0.5)
    svsa = SVSA(nx,model.step_t,model.step_adj,vnorm=vnorm,seed=509,debug=True)

    # base trajectory
    x0 = rng.normal(0.0,scale=1.0,size=nx)
    for i in range(500):
        x0 = model(x0)
    xb = [x0]
    for i in range(4):
        x0 = model(x0)
        xb.append(x0)
    
    # SV calculation
    nmode = 3
    sv_a = svsa(xb,nmode=nmode,analytical=True)
    sv_l = svsa(xb,nmode=nmode,analytical=False)

    fig, axs = plt.subplots(ncols=nmode,constrained_layout=True)
    for imode in range(nmode):
        sv_a1 = sv_a[imode] / la.norm(sv_a[imode],ord=2)
        sv_l1 = sv_l[imode] / la.norm(sv_l[imode],ord=2)
        if imode==2:
            sv_l1 = sv_l1*(-1.0)
        axs[imode].plot(sv_a1,label='analytical')
        axs[imode].plot(sv_l1,ls='dashed',label='lanczos')
        axs[imode].set_title(f'{imode+1} mode')
    axs[0].legend()
    plt.show()