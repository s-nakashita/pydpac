#cloned from tenomoto/DEnKF(https://github.com/tenomoto/DEnKF.git) then modified
from math import tau
import jax
import jax.numpy as jnp
from jax import jit, config, vjp, jvp
from functools import partial
try:
    from .qg.fd import laplacian, jacobian
    from .qg.mg import v_cycle
    from .qg.ode import rk4
except ImportError:
    from qg.fd import laplacian, jacobian
    from qg.mg import v_cycle
    from qg.ode import rk4
import matplotlib.pyplot as plt
import time
config.update("jax_enable_x64",True)

@jit
def lbc(pin):
    ni, nj = pin.shape
    pout = jnp.vstack((
        jnp.zeros(nj+2),
        jnp.hstack((jnp.zeros(ni).reshape(-1,1),pin,jnp.zeros(ni).reshape(-1,1))),
        jnp.zeros(nj+2)))
    return pout

#@partial(jit,static_argnames=['calc_time'])
def step(q, psi, d, f, beta, eps, a, tau0, y, itermax, tol): #, calc_time):
    #if calc_time: start = time.perf_counter()
    d2 = d*2
    dd = d*d
    psinew, _ = v_cycle(psi, q, d, f, itermax, tol) #, calc_time=calc_time)
    #if calc_time: 
    #    end = time.perf_counter()
    #    print(f"mg cycle: {(end - start)*1e3:.3f}ms")
    #if calc_time: start = time.perf_counter()
    psix = ((jnp.roll(psinew,-1,axis=0) - jnp.roll(psinew,1,axis=0)) / d2)[1:-1,1:-1]
    psix = lbc(psix)
    lap3psi = laplacian(laplacian(q + f * psinew) / dd) / dd 
    jac = jacobian(psinew, q) / dd
    dqdt = (-beta * psix - eps * jac - a * lap3psi + tau0 * jnp.sin(tau * y[None,]))[1:-1,1:-1]
    dqdt = lbc(dqdt)
    #if calc_time: 
    #    end = time.perf_counter()
    #    print(f"dqdt: {(end - start)*1e3:.3f}ms")
    return dqdt

def fwd(q,dt,psi,*params):
    d, f, beta, eps, a, tau0, y, itermax, tol = params
    #if calc_time: start = time.perf_counter()
    dqdt = rk4(step, q, dt, psi, *params)
    qnew = q + lbc(dqdt[1:-1,1:-1])
    psinew, _ = v_cycle(psi,qnew, d, f, itermax, tol) #, calc_time=calc_time)
    #if calc_time: 
    #    end = time.perf_counter()
    #    print(f"rk4 total: {(end - start)*1e3:.3f}ms")
    return qnew, psinew

def tlm(q,dq,dt,psi,*params):
    f = lambda q: fwd(q,dt,psi,*params)
    qnew, dfq, psinew = jvp(f, (q,), (dq,), has_aux=True)
    return dfq

def adj(q,dq,dt,psi,*params):
    f = lambda q: fwd(q,dt,psi,*params)
    qnew, f_vjp, psinew = vjp(f,q,has_aux=True)
    return f_vjp(dq)[0]

class QG():
    def __init__(self, ni, nj, dt, y, beta, f, eps, a, tau0, itermax, tol):
        self.ni = ni
        self.nj = nj
        self.dt = dt
        self.y = y
        self.d = self.y[1] - self.y[0]
        self.d2 = self.d * 2
        self.dd = self.d ** 2
        self.beta = beta
        self.f = f
        self.eps = eps
        self.a = a
        self.tau0 = tau0
        self.itermax = itermax
        self.tol = tol

    def get_params(self):
        return self.ni,self.nj,self.dt,self.d,self.beta,self.f,self.eps,\
            self.a,self.tau0,self.itermax,self.tol

    def __call__(self, qa, psia):
        inputs = (self.d, self.f, self.beta, self.eps, self.a, self.tau0, self.y, self.itermax, self.tol)
        return fwd(qa, self.dt, psia, *inputs)
    
    def step_t(self, q, dq, psi):
        inputs = (self.d, self.f, self.beta, self.eps, self.a, self.tau0, self.y, self.itermax, self.tol)
        return tlm(q,dq,self.dt,psi,*inputs)
    
    def step_adj(self, q, dq, psi):
        inputs = (self.d, self.f, self.beta, self.eps, self.a, self.tau0, self.y, self.itermax, self.tol)
        return adj(q,dq,self.dt,psi,*inputs)

    def calc_dist(self, rij):
        dist = jnp.zeros(self.ni*self.nj)
        ij = int(rij)
        iloc1 = ij1 // self.nj
        jloc1 = ij1 - iloc1*self.nj
        #print(f"ij={ij} i={iloc} j={jloc}")
        k=0
        for i in range(self.ni):
            for j in range(self.nj):
                dist[k] = jnp.sqrt(abs(i-iloc)**2+abs(j-jloc)**2)
                k+=1
        dist*=self.d
        return dist

    def calc_dist1(self, rij1, rloc):
        ij1 = int(rij1)
        iloc1 = ij1 // self.nj
        jloc1 = ij1 - iloc1*self.nj
        #print(f"ij1={ij1} i={iloc1} j={jloc1}")
        #print(f"rloc={rloc}")
        dist = jnp.sqrt(abs(iloc1-rloc[1])**2+abs(jloc1-rloc[2])**2)*self.d
        return dist

if __name__ == "__main__":
    from jax import random
    import numpy as np
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
    from model.qgmain import QG as QGn

    n = 129 
    dt = 1.25
    tsave = 0 
    nstep = 10000
    nsave = 100
    itermax = 1, 1, 100
    
    x = jnp.linspace(0.0, 1.0, n)
    y = jnp.linspace(0.0, 1.0, n)
    seed = 514
    key = random.key(seed)
    key, subkey = random.split(key)
    #q = jnp.zeros([n, n])
    #q = q.at[1:-1, 1:-1].set(2.0 * random.normal(subkey,shape=[n-2, n-2]) - 1.0)
    q0 = np.load('../model/qg/test/q000000.npy')
    q = jnp.asarray(q0)
    psi = jnp.zeros([n, n])
#    beta, f, eps, a, tau0 = 0.0, 0.0, 1.0, 2.0e-12, 0.0
    beta, f, eps, a, tau0 = 1.0, 1600, 1.0e-5, 2.0e-12, -tau
    tol = 1.0e-4
    params = beta, f, eps, a, tau0, itermax, tol
    #qn = q0.copy()
    #psin = np.zeros([n, n])
    #qgn = QGn(n,n,dt,y,*params)
    qg = QG(n,n,dt,y,*params)
    # jit precompile
    qtmp, ptmp = qg(q, psi)

    datadir="qg/test"
    #datadir="../data/qg"
    if not os.path.isdir(datadir):
        os.makedirs(datadir)
    for i in range(nstep+1): 
        if i >= tsave and i % nsave == 0:
            np.save(f"{datadir}/q{i:06d}.npy", jax.device_get(q))
            np.save(f"{datadir}/p{i:06d}.npy", jax.device_get(psi))
        print(f"step {i} p: min={psi.min():5.2e} max={psi.max():5.2e} q: min={q.min():5.2e} max={q.max():5.2e}")
        #dq = np.zeros([n, n])
        #dq[1:-1, 1:-1] = rk4(step, q, dt, psi, y, *params)[1:-1, 1:-1]
        #np.save(f"dq.npy", dq)
        if i<0:
            print("\n==JAX==")
            q, psi = qg(q, psi, calc_time=True)
            print("\n==NumPy==")
            qn[1:-1,1:-1] = qgn(qn, psin, calc_time=True)[1:-1,1:-1]

            qj = jax.device_get(q)
            psij = jax.device_get(psi)
            fig, axs = plt.subplots(2,3,figsize=[10,6],constrained_layout=True)
            # psi
            c = axs[0,0].pcolormesh(x,y,psin)
            fig.colorbar(c, ax=axs[0,0], shrink=0.6, pad=0.01)
            axs[0,0].set_title('NumPy')
            axs[0,0].set_ylabel(r'$\psi$')
            c = axs[0,1].pcolormesh(x,y,psij)
            fig.colorbar(c,ax=axs[0,1],shrink=0.6,pad=0.01)
            axs[0,1].set_title('JAX')
            pd = (psij - psin) #/np.sqrt(np.mean(psin**2))*100
            c = axs[0,2].pcolormesh(x,y,pd,cmap='RdBu')
            fig.colorbar(c,ax=axs[0,2],shrink=0.6,pad=0.01)
            axs[0,2].set_title('JAX - NumPy')
            # q
            c = axs[1,0].pcolormesh(x,y,qn)
            fig.colorbar(c,ax=axs[1,0],shrink=0.6,pad=0.01)
            axs[1,0].set_ylabel(r'$q$')
            c = axs[1,1].pcolormesh(x,y,qj)
            fig.colorbar(c,ax=axs[1,1],shrink=0.6,pad=0.01)
            qd = (qj - qn) #/np.sqrt(np.mean(qn**2))*100
            c = axs[1,2].pcolormesh(x,y,qd,cmap='RdBu')
            fig.colorbar(c,ax=axs[1,2],shrink=0.6,pad=0.01)
            for ax in axs.flatten():
                ax.set_aspect(1.0)
            fig.suptitle(r"$t=$"+f"{i}")
            plt.show()
            plt.close()
        else:
            q, psi = qg(q, psi)
