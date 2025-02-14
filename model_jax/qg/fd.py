import jax.numpy as jnp
from jax import jit, config
config.update("jax_enable_x64",True)

@jit
def laplacian(p):
    n = p.shape[0]
    lpin = (jnp.roll(p,-1,axis=0) + jnp.roll(p,1,axis=0) + jnp.roll(p,-1,axis=1) + jnp.roll(p,1,axis=1) - 4 * p)[1:-1,1:-1]
    lp = jnp.vstack((jnp.zeros(n),
         jnp.hstack((jnp.zeros(n-2).reshape(-1,1),lpin,jnp.zeros(n-2).reshape(-1,1))),
         jnp.zeros(n)))
    #lp = jnp.zeros_like(p)
    #lpin = p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] \
    #               - 4 * p[1:-1, 1:-1]
    #lp = lp.at[1:-1, 1:-1].set(lpin)
    return lp

@jit
def jacobian(p, q):
    n = p.shape[0]
    ja = jnp.zeros_like(p)
    j1 = (jnp.roll(q,-1,axis=1) - jnp.roll(q,1,axis=1)) * (jnp.roll(p,-1,axis=0) - jnp.roll(p,1,axis=0)) - \
         (jnp.roll(q,-1,axis=0) - jnp.roll(q,1,axis=0)) * (jnp.roll(p,-1,axis=1) - jnp.roll(p,1,axis=1))
    #j1 = (q[1:-1, 2:] - q[1:-1, :-2]) * (p[2:, 1:-1] - p[:-2, 1:-1]) - \
    #     (q[2:, 1:-1] - q[:-2, 1:-1]) * (p[1:-1, 2:] - p[1:-1, :-2])
    j2 = (jnp.roll(jnp.roll(q,-1,axis=1),-1,axis=0) - jnp.roll(jnp.roll(q,1,axis=1),-1,axis=0)) * jnp.roll(p,-1,axis=0) - \
         (jnp.roll(jnp.roll(q,-1,axis=1),1,axis=0) - jnp.roll(jnp.roll(q,1,axis=1),1,axis=0)) * jnp.roll(p,1,axis=0) - \
         (jnp.roll(jnp.roll(q,-1,axis=1),-1,axis=0) - jnp.roll(jnp.roll(q,-1,axis=1),1,axis=0)) * jnp.roll(p,-1,axis=1) + \
         (jnp.roll(jnp.roll(q,1,axis=1),-1,axis=0) - jnp.roll(jnp.roll(q,1,axis=1),1,axis=0)) * jnp.roll(p,1,axis=1)
    #j2 = (q[2:, 2:]  - q[2:, :-2])  * p[2:, 1:-1] - \
    #     (q[:-2, 2:] - q[:-2, :-2]) * p[:-2, 1:-1] - \
    #     (q[2:, 2:]  - q[:-2, 2:])  * p[1:-1, 2:] + \
    #     (q[2:,:-2]  - q[:-2, :-2]) * p[1:-1, :-2]
    j3 = jnp.roll(q,-1,axis=1) * (jnp.roll(jnp.roll(p,-1,axis=0),-1,axis=1) - jnp.roll(jnp.roll(p,1,axis=0),-1,axis=1)) - \
         jnp.roll(q,1,axis=1) * (jnp.roll(jnp.roll(p,-1,axis=0),1,axis=1) - jnp.roll(jnp.roll(p,1,axis=0),1,axis=1)) - \
         jnp.roll(q,-1,axis=0) * (jnp.roll(jnp.roll(p,-1,axis=0),-1,axis=1) - jnp.roll(jnp.roll(p,-1,axis=0),1,axis=1)) + \
         jnp.roll(q,1,axis=0) * (jnp.roll(jnp.roll(p,1,axis=0),-1,axis=1) - jnp.roll(jnp.roll(p,1,axis=0),1,axis=1))
    #j3 = q[1:-1, 2:]  * (p[2:, 2:]  - p[:-2, 2:]) - \
    #     q[1:-1, :-2] * (p[2:, :-2] - p[:-2, :-2]) - \
    #     q[2:, 1:-1]  * (p[2:, 2:]  - p[2:, :-2]) + \
    #     q[:-2, 1:-1] * (p[:-2, 2:] - p[:-2, :-2])
    #ja = ja.at[1:-1, 1:-1].add((j1 + j2 + j3) / 12)
    jain = ((j1 + j2 + j3) / 12)[1:-1,1:-1]
    ja = jnp.vstack((jnp.zeros(n),
         jnp.hstack((jnp.zeros(n-2).reshape(-1,1),jain,jnp.zeros(n-2).reshape(-1,1))),
         jnp.zeros(n)))
    return ja

@jit
def l2norm(u, h):
    return jnp.sqrt(h ** u.ndim * (u ** 2).sum())


