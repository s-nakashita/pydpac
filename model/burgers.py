import numpy as np
import matplotlib.pyplot as plt

class Bg():
    def __init__(self, nx, dx, dt, nu):
        self.nx = nx
        self.dx = dx
        self.dt = dt
        self.nu = nu

    def get_params(self):
        return self.nx, self.dx, self.dt, self.nu

    def l_operator(self, f, u):
# NB u is defined at integral index
# f[i-1/2] = (f[i-1] + f[i]) / 2
# f[i+1/2] = (f[i] + f[i+1]) / 2
# f[i+1/2] - f[i-1/2] = (f[i+1] - f[i-1]) / 2
        l = np.zeros_like(u)
        l[1:-1] = -0.5 * (f[2:] - f[0:-2]) / self.dx \
            + self.nu / self.dx**2 * (u[2:] - 2 * u[1:-1] + u[0:-2])
        return l

    def __call__(self, u):
        f = 0.5 * u**2
        u1 = u + self.dt * self.l_operator(f, u)
        return 0.5 * (u + u1 + self.dt * self.l_operator(f, u1))

    def tlm(self, f, du):
        l = np.zeros_like(du)
        l[1:-1] = -0.5 * (f[2:] - f[0:-2]) / self.dx \
            + self.nu / self.dx**2 * (du[2:] - 2 * du[1:-1] + du[0:-2])
        return l

    def step_t(self, u, du):
        f = u * du
        u1 = du + self.dt * self.tlm(f, du)
        return 0.5 * (du + u1 + self.dt * self.tlm(f, u1))

    def adj(self, du):
        l = np.zeros(du.shape[0]*2)
        l[0:2] = 0.5 * du[1:3] / self.dx
        l[2:self.nx-2] = 0.5 * (du[3:-1] - du[1:-3]) / self.dx 
        l[self.nx-2:self.nx] = -0.5 * du[-3:-1] / self.dx
        l[self.nx] = self.nu / self.dx**2 * du[1]
        l[self.nx+1] = self.nu / self.dx**2 * (-2 * du[1] + du[2])
        l[self.nx+2:-2] = self.nu / self.dx**2 * (du[3:-1] - 2 * du[2:-2] + du[1:-3])
        l[-2] = self.nu / self.dx**2 * (du[-3] - 2 * du[-2])
        l[-1] = self.nu / self.dx**2 * du[-2]
        return l 

    def step_adj(self, u, du):
        u1 = self.dt * self.adj(du)
        u2 = self.dt * self.adj(u1[self.nx:])
        du[1:-1] = du[1:-1] + u[1:-1] * u1[1:self.nx-1] + u1[self.nx+1:-1] \
            + 0.5 * (u[1:-1] * u2[1:self.nx-1] + u2[self.nx+1:-1])
        return du

    def calc_dist(self,iloc):
        dist = np.zeros(self.nx)
        for j in range(self.nx):
            dist[j] = abs(iloc - j)*dx
        return dist

    def calc_dist1(self,iloc,jloc):
        return abs(iloc - jloc)*dx

if __name__ == "__main__":
    n = 81
    nu = 1.0
    dt = 0.002
    tmax = 2.0
    tsave = 0.5
    reynolds_number = 1.0

    x = np.linspace(-np.pi, np.pi, n)
    dx = x[1] - x[0]
    u = -reynolds_number * np.sin(x)
    umax = np.amax(np.abs(u))
    c = umax * dt / dx
    nt = int(tmax / dt) + 1
    
    burgers = Bg(n, dx, dt, nu)

    print("n={} nu={} dt={:7.3e} tmax={} tsave={}".format(n, nu, dt, tmax, tsave))
    print("R={} dx={:7.3e} umax={} c={} nt={}".format(reynolds_number, dx, umax, c, nt))

    np.savetxt("x.txt".format(0), x)
    np.savetxt("u{:05d}.txt".format(0), u)
    fig, ax = plt.subplots()
    for k in range(nt):
        print("step {:05d}".format(k))
        if k * dt % tsave == 0:
                np.savetxt("u{:05d}.txt".format(k), u)
                ax.plot(x, u)
        u = burgers(u)
    fig.savefig("u.png")

    a = 1e-5
    u = -reynolds_number * np.sin(x)
    du = np.random.randn(u.size)
    adx = burgers(u+a*du)
    ax = burgers(u)
    jax = burgers.step_t(u, du)
    d = np.sqrt(np.sum((adx-ax)**2)) / a / (np.sqrt(np.sum(jax**2)))
    print("TLM check diff.={}".format(d-1))

    ax = burgers.step_t(u, du)
    atax = burgers.step_adj(u, ax)
    d = (ax.T @ ax) - (du.T @ atax)
    print("ADJ check diff.={}".format(d))