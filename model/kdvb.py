import numpy as np
from .fft import derivative
from numpy.fft import rfft
from math import tau

class KdVB():
    def __init__(self, nx, dt, dx, nu, x=None, alpha=6.0, beta=1.0, period=tau, fd=False):
        self.nx = nx
        self.dt = dt
        self.dx = dx
        self.nu = nu
        self.alpha = alpha
        self.beta = beta
        self.period = period
        self.fd = fd
        if not self.fd:
            self.ntrunc = int(self.nx / 3 - 1)
        if x is None:
            xmax = (self.nx-1) * self.dx
            self.x = np.linspace(-xmax/2, xmax/2, self.nx, endpoint=True)
        else:
            self.x = x
        print(f"nx={self.nx} dt={self.dt:.4e} dx={self.dx:.4e} \n x={self.x}")
        print(f"alpha={self.alpha:.4e} beta={self.beta:.4e} nu={self.nu:.4e} period={self.period:.4e} fd={self.fd}")

    def get_params(self):
        return self.nx, self.dt, self.dx, self.nu

    def get_x(self):
        return self.x

    def sech(self,x):
        return 2.0 / (np.exp(x) + np.exp(-x))
    
    def csch(self,x):
        return 2.0 / (np.exp(x) - np.exp(-x))
    
    def tanh(self,x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def coth(self,x):
        return (np.exp(x) + np.exp(-x)) / (np.exp(x) - np.exp(-x))

    def soliton1(self, t, k):
        c = 4.0*k*k
        return 2.0*k*k*np.square(self.sech(k*(self.x - c*t)))

    def soliton2(self, t, k1, k2):
        u = np.zeros_like(self.x)
        k1k1 = k1*k1
        k2k2 = k2*k2
        t1 = k1 * (self.x - 4.0 * k1k1 * t)
        t2 = k2 * (self.x - 4.0 * k2k2 * t)
        u = k2k2 - k1k1 + k2k2 * np.cosh(2.0 * t1) + k1k1 * np.cosh(2.0 * t2)
        u /= ((k2 - k1) * np.cosh(t1 + t2) + (k2 + k1) * np.cosh(t1 - t2)) ** 2
        u *= 4.0 * (k2k2 - k1k1)
        return u

    ### FFT ###
    def tendency(self, u):
        u, du, du3 = derivative(u, [0,1,3], period=self.period, trunc=self.ntrunc)
        udu = derivative(u*du, [0], trunc=self.ntrunc)[0]
        l = -(self.alpha*udu + self.beta*du3)
        return l

    def diffuse(self,u):
        du2 = derivative(u, [2], period=self.period)[0]
        l = self.nu + du2
        return l

    ### Finite difference ###
    def tendency_fd(self, u):
        l = -0.5 * (self.alpha * u * (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) \
                  + self.beta * (np.roll(u, -2, axis=0) - 2.0*np.roll(u, -1, axis=0)\
                         + 2.0*np.roll(u, 1, axis=0) - np.roll(u, 2, axis=0)) / self.dx / self.dx ) / self.dx 
        return l

    def diffuse_fd(self, u):
        l = self.nu * (np.roll(u, 1, axis=0) - 2.0*u + np.roll(u, -1, axis=0)) / self.dx / self.dx
        return l

    def l_operator(self, u):
        if self.fd:
            return self.tendency_fd(u)
        else:
            return self.tendency(u)

    def __call__(self, ua):
        k1 = self.dt * self.l_operator(ua)
        #print(np.max(k1), np.min(k1))
    
        k2 = self.dt * self.l_operator(ua+k1*0.5)
        #print(np.max(k2), np.min(k2))
    
        k3 = self.dt * self.l_operator(ua+k2*0.5)
        #print(np.max(k3), np.min(k3))
    
        k4 = self.dt * self.l_operator(ua+k3)
        #print(np.max(k4), np.min(k4))
        uf = ua + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
        if self.nu != 0.0:
            if self.fd:
                uf += self.diffuse_fd(ua)
            else:
                uf += self.diffuse(ua)
        return uf

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nx = 101
    dt = 0.01
    dx = 0.5
    nu = 0.07
    print(dt/(np.power(dx,3)))
    
    kdvb = KdVB(nx, dt, dx, nu)
    x = kdvb.get_x()
    b1 = 0.5
    b2 = 1.0
    k1 = np.sqrt(0.5*b1)
    k2 = np.sqrt(0.5*b2)
    fig, ax = plt.subplots(figsize=(10,5))
    for t in range(-4, 9, 4):
        u = kdvb.soliton2(t,k1,k2)
        ax.plot(x, u, label=f"$t={t}$")
    ax.legend()
    fig.savefig("two_solitons.png", bbox_inches="tight", dpi=300)
