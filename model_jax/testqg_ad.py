from qgmain import fwd 
from math import tau
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

datadir = "qg/test"

n = 129
dt = 1.25
x = np.linspace(0.0,1.0,n)
y = np.linspace(0.0,1.0,n)
d = x[1] - x[0]
itermax = 1, 1, 100
beta, f, eps, a, tau0 = 1.0, 1600, 1.0e-5, 2.0e-12, -tau
tol = 1.0e-4
params = d, f, beta, eps, a, tau0, y, itermax, tol

ts = 3500
nstep = 100
qtmp = np.load(f"{datadir}/q{ts:06d}.npy")
ptmp = np.load(f"{datadir}/p{ts:06d}.npy")
q = jnp.asarray(qtmp)
psi = jnp.asarray(ptmp)
qblist = [q]
psiblist = [psi]
f_vjp_list = []
for i in range(nstep):
    f = lambda q: fwd(q,dt,psi,*params)
    q, f_vjp, psi = jax.vjp(f, q, has_aux=True)
    f_vjp_list.append(f_vjp)
    qblist.append(q)
    psiblist.append(psi)

dq = jnp.zeros_like(q)
dq = dq.at[62:67,62:67].set(jnp.full((5,5),1e2))
dqlist = [dq]
for i in range(nstep):
    f_vjp = f_vjp_list[nstep-i-1]
    dq = f_vjp(dq)[0]
    print(f"step {i}: dq max={jnp.max(dq)} min={jnp.min(dq)}")
    dqlist.append(dq)

ncols = nstep // 10 + 1
nrows = 3
fig = plt.figure(figsize=(ncols,nrows),constrained_layout=True)
plim = jnp.max(jnp.abs(psi))
qlim = jnp.max(jnp.abs(q))
for icol in range(ncols):
    t = nstep - 10*icol
    psib = psiblist[t]
    qb   = qblist[t]
    dq   = dqlist[nstep-t]
    axp = fig.add_subplot(nrows,ncols,icol+1)
    axq = fig.add_subplot(nrows,ncols,ncols+icol+1)
    axd = fig.add_subplot(nrows,ncols,2*ncols+icol+1)
    if icol==0:
        axp.set_ylabel(r'$\psi_b$')
        axq.set_ylabel(r'$q_b$')
        axd.set_ylabel(r'$\delta q$')
    axp.set_title(t)
    c0 = axp.pcolormesh(x,y,psib.T,vmin=-plim,vmax=plim)
    c1 = axq.pcolormesh(x,y,qb.T,vmin=-qlim,vmax=qlim)
    vlim = 1e2 #jnp.max(jnp.abs(dq))
    c2 = axd.pcolormesh(x,y,dq.T,vmin=-vlim,vmax=vlim,cmap='RdBu_r')
    fig.colorbar(c2,ax=axd,shrink=0.6,pad=0.01,orientation='horizontal')
    for ax in [axp,axq,axd]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.0)
#fig.colorbar(c0,ax=axp,shrink=0.5,orientation='horizontal')
#fig.colorbar(c1,ax=axq,shrink=0.5,orientation='horizontal')
plt.show()