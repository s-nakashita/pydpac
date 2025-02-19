from qgmain import QG 
from math import tau
import numpy as np
import jax.numpy as jnp
import jax

datadir = "qg/test"

n = 129
dt = 1.25
x = np.linspace(0.0,1.0,n)
y = np.linspace(0.0,1.0,n)
itermax = 1, 1, 100
beta, f, eps, a, tau0 = 1.0, 1600, 1.0e-5, 2.0e-12, -tau
tol = 1.0e-4
params = beta, f, eps, a, tau0, itermax, tol 
qg = QG(n,n,dt,y,*params)

t = 1000
qtmp = np.load(f"{datadir}/q{t:06d}.npy")
ptmp = np.load(f"{datadir}/p{t:06d}.npy")
qb = jnp.asarray(qtmp)
psib = jnp.asarray(ptmp)

alp = 1.0e-5
seed = 514
key = jax.random.key(seed)
niter = 5
for i in range(niter):
    key, subkey = jax.random.split(key)
    dq = jax.random.normal(subkey,qb.shape)
    #key, subkey = jax.random.split(key)
    #dpsi = jax.random.normal(subkey,psib.shape)
    adq, adp = qg(qb+alp*dq,psib) #+alp*dpsi)
    aq, ap = qg(qb,psib)
    aqj = qg.step_t(qb,dq,psib)
    #print(aqj)
    d1 = jnp.sqrt(jnp.sum((adq-aq)**2))
    d2 = alp * (jnp.sqrt(jnp.sum(aqj**2)))
    d=d1/d2
    print("iter{}:     TLM check = {:.6e},{:.6e} ratio-1={:.3e}".format(i+1,d1,d2,d-1.0))

    ataqj = qg.step_adj(qb,aqj,psib)
#    print(ataqj)
    d1 = jnp.dot(aqj.flatten(),aqj.flatten()) #+ jnp.dot(apj.flatten(),apj.flatten())
    d2 = jnp.dot(dq.flatten(),ataqj.flatten()) #+ jnp.dot(dpsi.flatten(),atapj.flatten())
    d=d1-d2
    print("iter{}:     ADM check = {:.6e},{:.6e} diff={:.3e}".format(i+1,d1,d2,d))
