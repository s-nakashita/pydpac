import jax.numpy as jnp
from jax import config, jit, vjp, jvp
from functools import partial
config.update('jax_enable_x64',True)

def fwd(f,x,dt,params):
    k1 = dt*f(x,*params)
    k2 = dt*f(x+k1/2,*params)
    k3 = dt*f(x+k2/2,*params)
    k4 = dt*f(x+k3,*params)
    return x + (k1+2*k2+2*k3+k4)/6

# type I (L96)
@partial(jit,static_argnums=(1))
def tend1(x,*params):
    F = params[0]
    dxdt = (jnp.roll(x,-1,axis=0) - jnp.roll(x,2,axis=0))*jnp.roll(x,1,axis=0) - x + F 
    return dxdt

# type II (L05)
@partial(jit,static_argnums=(1))
def tend2(x,*params):
    K, F = params
    dxdt = adv(x,K) - x + F
    return dxdt

@partial(jit,static_argnames=['K'])
def adv(x,K):
#    sumdiff = K%2==0
    Kmod = 1 - K%2
    J = (K - K%2)//2
    w = jnp.zeros_like(x)
    for j in range(-J,J+1):
        w = w + jnp.roll(x,-j,axis=0)
    w = w - Kmod*(0.5*jnp.roll(x,-J,axis=0) + 0.5*jnp.roll(x,J,axis=0))
    w = w / K 
    ladv = jnp.zeros_like(x)
    for j in range(-J,J+1):
        ladv = ladv + jnp.roll(w,K-j,axis=0)*jnp.roll(x,-K-j,axis=0)
    ladv = ladv - Kmod*(0.5*jnp.roll(w,K+J,axis=0)*jnp.roll(x,-K+J,axis=0) + 0.5*jnp.roll(w,K-J,axis=0)*jnp.roll(x,-K-J,axis=0))
    ladv = ladv / K 
    ladv = ladv - jnp.roll(w,2*K,axis=0)*jnp.roll(w,K,axis=0)
    return ladv

@partial(jit,static_argnums=(1))
def tend3(z,*params):
    K,filmat,b,c,F = params
    x, y = decomp(z,filmat)
    adv1 = adv(x,K)
    adv2 = adv(y,1)
    adv3 = -jnp.roll(y,2,axis=0)*jnp.roll(x,1,axis=0) + jnp.roll(y,1,axis=0)*jnp.roll(x,-1,axis=0)
    dxdt = adv1 + b*b*adv2 + c*adv3 - x - b*y + F
    return dxdt

@jit
def decomp(z,filmat):
    x = jnp.dot(filmat,z)
    y = z - x 
    return x,y 

def set_filmat(nx,ni):
    i2 = ni*ni
    i3 = i2*ni
    i4 = i3*ni
    alp = (3.0*i2+3.0)/(2.0*i3+4.0*ni)
    bet = (2.0*i2+1.0)/(i4+2.0*i2)
    filmat = jnp.zeros((nx,nx))
    for i in range(filmat.shape[0]):
        js = i - ni 
        je = i + ni + 1
        if js < 0: js+=filmat.shape[1]
        if je>filmat.shape[1]: je-=filmat.shape[1]
        tmplist = []
        for j in range(filmat.shape[1]):
            tmp = 0.0
            jj = j 
            if js<je:
                if j>=js and j<je:
                    tmp = alp - bet*abs(jj-i)
            else:
                if j<je or j>=js:
                    tmp = alp - bet*min(abs(jj-i),filmat.shape[1]-abs(jj-i))
            if j==js or j==(je-1): tmp*=0.5
            tmplist.append(tmp)
        filmat = filmat.at[i,:].set(tmplist) 
    return filmat

def tlm(tend,x,dx,dt,*params):
    f = lambda x: fwd(tend,x,dt,*params)
    xnew, df = jax.jvp(f, (x,), (dx,))
    return df

def adj(tend,x,dxa,dt,*params):
    f = lambda x: fwd(tend,x,dt,*params)
    xnew, vjp_fun = jax.vjp(f,x)
    return vjp_fun(dxa)[0]

class L05j():
    def __init__(self, nx, dt, *params, ltype=1):
        self.nx = nx
        self.dt = dt
        self.params = params
        self.ltype = ltype
        self.tend = {1:tend1,2:tend2,3:tend3}

    def get_params(self):
        return self.nx, self.dt, self.params

    def __call__(self, xa):
        return fwd(self.tend[self.ltype],xa,self.dt,*self.params)

    def step_t(self, x, dx):
        return tlm(self.tend[self.ltype],x,dx,self.dt,*self.params)

    def step_adj(self, x, dx):
        return adj(self.tend[self.ltype],x,dx,self.dt,*self.params)

    def calc_dist(self, iloc):
        dist = []
        for j in range(self.nx):
            dist.append([abs(self.nx / jnp.pi * jnp.sin(jnp.pi * (iloc - float(j)) / self.nx))])
        return jnp.asarray(dist)
    
    def calc_dist1(self, iloc, jloc):
        dist = abs(self.nx / jnp.pi * jnp.sin(jnp.pi * (iloc - jloc) / self.nx))
        return dist

if __name__ == "__main__":
    import sys
    import os
    import numpy as np
    import jax
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 16
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
    from model.lorenz import L96
    from model.lorenz2 import L05II
    from model.lorenz3 import L05III
    
    ltype = 1
    if len(sys.argv)>1:
        ltype = int(sys.argv[1])

    if ltype == 1:
        n = 40
        F = 8.0
        dt = 0.05
        model = L96(n, dt, F)
        model_j = L05j(n, dt, (F,), ltype=ltype)
    elif ltype == 2:
        n = 240
        k = 8
        F = 10.0
        dt = 0.05
        model = L05II(n,k,dt,F)
        model_j = L05j(n, dt, (k,F), ltype=ltype)
    elif ltype == 3:
        n = 960
        k = 32
        i = 12
        F = 15.0
        b = 10.0
        c = 0.6
        dt = 0.05 / b
        model = L05III(n,k,i,b,c,dt,F)
        model_j = L05j(n,dt,(k,set_filmat(n,i),b,c,F),ltype=ltype)

    x0 = np.ones(n)*F
    x0[n//2-1] += 0.001*F
    x0j = jnp.asarray(x0)
    
    tmax = 2.0
    nt = int(tmax/dt)
    x = [x0]
    for k in range(nt):
        x0 = model(x0)
        x.append(x0)
    x = np.array(x)

    xj = [jax.device_get(x0j)]
    for k in range(nt):
        x0j = model_j(x0j)
        xj.append(jax.device_get(x0j))
    xj = np.array(xj)
    print(f"initial diff={jnp.dot((x[0]-xj[0]),(x[0]-xj[0])):.4e}")

    if not os.path.isdir('lorenz'):
        os.mkdir('lorenz')
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
    fig.suptitle(f'Type {ltype}')
    fig.savefig(f'lorenz/test_l05_{ltype}_jax.png')
    plt.show()
    #exit()

    a = 1e-5
    x0 = jnp.ones(n)
    seed = 514
    key = jax.random.key(seed)
    niter = 10
    print(f"\n== Type {ltype} ==")
    for i in range(niter):
        key, subkey = jax.random.split(key)
        dx = jax.random.normal(subkey,x0.shape)
        adx = model_j(x0+a*dx)
        ax = model_j(x0)
        axj = model_j.step_t(x0, dx)
        d = jnp.sqrt(jnp.sum((adx-ax)**2)) / a / (jnp.sqrt(jnp.sum(axj**2)))
        print("iter{}: TLM (jax) check diff.={}".format(i+1,d-1.0))

        ax = model_j.step_t(x0, dx)
        atax = model_j.step_adj(x0, ax)
        d = (ax.T @ ax) - (dx.T @ atax)
        print("iter{}: ADM (jax) check diff.={}".format(i+1,d))
