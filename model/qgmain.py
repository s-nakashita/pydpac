#cloned from tenomoto/DEnKF(https://github.com/tenomoto/DEnKF.git) then modified
from math import tau
import numpy as np
from .qg.fd import laplacian, jacobian
from .qg.mg import v_cycle
from .qg.ode import rk4
import sys

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
        return self.ni,self.nj,self.dt,self.y,self.beta,self.f,self.eps,\
            self.a,self.tau0,self.itermax,self.tol

    def step(self, q, psi):
        psi[:,:], _ = v_cycle(psi, q, self.d, self.f, self.itermax, self.tol)
        psix = np.zeros_like(psi)
        dqdt = np.zeros_like(psi)
        psix[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / self.d2
        lap3psi = laplacian(laplacian(q + self.f * psi) / self.dd) / self.dd 
        jac = jacobian(psi, q) / self.dd
        dqdt[1:-1, 1:-1] = (-self.beta * psix - self.eps * jac - self.a * lap3psi + self.tau0 * np.sin(tau * self.y[None,]))[1:-1, 1:-1]
        return dqdt

    def __call__(self, qa, psia):
        qa[1:-1, 1:-1] += rk4(self.step,qa,self.dt,psia)[1:-1, 1:-1]
        return qa

    def calc_dist(self, ij):
        dist = np.zeros(self.ni*self.nj)
        jloc = ij // self.ni
        iloc = ij - jloc*self.ni
        print(f"ij={ij} i={iloc} j={jloc}")
        k=0
        for i in range(self.ni):
            for j in range(self.nj):
                dist[k] = np.sqrt(abs(i-iloc)**2+abs(j-jloc)**2)
                k+=1
        dist*=self.d
        return dist

    def calc_dist1(self, ij1, ij2):
        jloc1 = ij1 // self.ni
        iloc1 = ij1 - jloc1*self.ni
        print(f"ij1={ij1} i={iloc1} j={jloc1}")
        jloc2 = ij2 // self.ni
        iloc2 = ij2 - jloc2*self.ni
        print(f"ij2={ij2} i={iloc2} j={jloc2}")
        dist = np.sqrt(abs(iloc1-iloc2)**2+abs(jloc1-jloc2)**2)*self.d
        return dist

if __name__ == "__main__":
    n = 129 
    dt = 1.25
    tsave = 0 
    nstep = 10000
    nsave = 100
    itermax = 1, 1, 100
    datadir="free"
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    q = np.zeros([n, n])
    seed = 514
    rng = np.random.default_rng(514)
    q[1:-1, 1:-1] = 2.0 * rng.random([n-2, n-2]) - 1.0
    psi = np.zeros([n, n])
#    beta, f, eps, a, tau0 = 0.0, 0.0, 1.0, 2.0e-12, 0.0
    beta, f, eps, a, tau0 = 1.0, 1600, 1.0e-5, 2.0e-12, -tau
    tol = 1.0e-4
    params = beta, f, eps, a, tau0, itermax, tol
    qg = QG(n,n,dt,y,*params)

    for i in range(nstep+1): 
        if i >= tsave and i % nsave == 0:
            #np.save(f"qg/test/q{i:06d}.npy", q)
            #np.save(f"qg/test/p{i:06d}.npy", psi)
            np.save(f"../data/qg/q{i:06d}.npy", q)
            np.save(f"../data/qg/p{i:06d}.npy", psi)
        print(f"step {i} p: min={psi.min():5.2e} max={psi.max():5.2e} q: min={q.min():5.2e} max={q.max():5.2e}")
        #dq = np.zeros([n, n])
        #dq[1:-1, 1:-1] = rk4(step, q, dt, psi, y, *params)[1:-1, 1:-1]
        #np.save(f"dq.npy", dq)
        q[1:-1, 1:-1] = qg(q, psi)[1:-1, 1:-1]
