import jax.numpy as jnp
from jax import config, jit, vjp, jvp
from functools import partial
config.update('jax_enable_x64',True)

class L96j():
    def __init__(self, nx, dt, F):
        self.nx = nx
        self.dt = dt
        self.F = F
        print(f"nx={self.nx} F={self.F} dt={self.dt:.3e}")

    def get_params(self):
        return self.nx, self.dt, self.F

    @partial(jit, static_argnums=(0))
    def l96(self, x):
        l = (jnp.roll(x, -1, axis=0) - jnp.roll(x, 2, axis=0)) * jnp.roll(x, 1, axis=0) - x + self.F
        return l

    @partial(jit, static_argnums=(0))
    def __call__(self, xa):
        k1 = self.dt * self.l96(xa)
    
        k2 = self.dt * self.l96(xa+k1/2)
    
        k3 = self.dt * self.l96(xa+k2/2)
    
        k4 = self.dt * self.l96(xa+k3)
    
        return xa + (0.5*k1 + k2 + k3 + 0.5*k4)/3.0

    def step_t(self, x, dx):
        _, dxnew = jvp(self.__call__, (x,), (dx,))
        return dxnew

    def step_adj(self, x, dx):
        _, vjp_fun = vjp(self.__call__, x)
        dxa = vjp_fun(dx)[0]
        return dxa

    def calc_dist(self, iloc):
        dist = jnp.zeros(self.nx)
        for j in range(self.nx):
            dist = dist.at[j].set([abs(self.nx / jnp.pi * jnp.sin(jnp.pi * (iloc - float(j)) / self.nx))])
        return dist
    
    def calc_dist1(self, iloc, jloc):
        dist = abs(self.nx / jnp.pi * jnp.sin(jnp.pi * (iloc - jloc) / self.nx))
        return dist

if __name__ == "__main__":
    import numpy as np
    import jax
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 16
    from lorenz import L96
    n = 40
    F = 8.0
    h = 0.05

    l96 = L96(n, h, F)
    l96j = L96j(n, h, F)

    x0 = np.ones(n)*F
    x0[19] += 0.001*F
    tmax = 2.0
    nt = int(tmax/h)
    x = [x0]
    for k in range(nt):
        x0 = l96(x0)
        x.append(x0)
    x = np.array(x)

    x0j = jnp.ones(n)*F
    x0j = x0j.at[19].set(1.001*F)
    xj = [jax.device_get(x0j)]
    for k in range(nt):
        x0j = l96j(x0j)
        xj.append(jax.device_get(x0j))
    xj = np.array(xj)
    print(f"initial diff={jnp.dot((x[0]-xj[0]),(x[0]-xj[0])):.4e}")

    fig, axs = plt.subplots(figsize=[8,6],ncols=3,constrained_layout=True)
    xaxis = np.arange(n)
    taxis = np.arange(nt+1)
    p0 = axs[0].pcolormesh(xaxis,taxis,x,shading='auto',cmap='coolwarm',vmin=-15.0,vmax=15.0)
    p1 = axs[1].pcolormesh(xaxis,taxis,xj,shading='auto',cmap='coolwarm',vmin=-15.0,vmax=15.0)
    fig.colorbar(p1,ax=axs[1],pad=0.01,shrink=0.6)
    diff = x - xj 
    p2 = axs[2].pcolormesh(xaxis,taxis,diff,shading='auto',cmap='coolwarm')
    fig.colorbar(p2,ax=axs[2],pad=0.01,shrink=0.6)
    axs[0].set_title('numpy')
    axs[1].set_title('jax.numpy')
    axs[2].set_title('diff')
    fig.savefig('lorenz/test_l96_jnp.png')
    plt.show()
    #exit()

    a = 1e-5
    x0 = jnp.ones(n)
    seed = 514
    key = jax.random.key(seed)
    niter = 10
    for i in range(niter):
        key, subkey = jax.random.split(key)
        dx = jax.random.normal(subkey,x0.shape)
        key = subkey
        adx = l96j(x0+a*dx)
        ax = l96j(x0)
        axj = l96j.step_t(x0, dx)
        d = jnp.sqrt(jnp.sum((adx-ax)**2)) / a / (jnp.sqrt(jnp.sum(axj**2)))
        print("iter{}: TLM (jax) check diff.={}".format(i+1,d-1.0))

        ax = l96j.step_t(x0, dx)
        atax = l96j.step_adj(x0, ax)
        d = (ax.T @ ax) - (dx.T @ atax)
        print("iter{}: ADJ (jax) check diff.={}".format(i+1,d))
