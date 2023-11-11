import numpy as np

class L96():
    def __init__(self, nx, dt, F):
        self.nx = nx
        self.dt = dt
        self.F = F
        print(f"nx={self.nx} F={self.F} dt={self.dt}")

    def get_params(self):
        return self.nx, self.dt, self.F

    def l96(self, x):
        l = np.zeros_like(x)
        l = (np.roll(x, -1, axis=0) - np.roll(x, 2, axis=0)) * np.roll(x, 1, axis=0) - x + self.F
        return l

    def __call__(self, xa):
        k1 = self.dt * self.l96(xa)
    
        k2 = self.dt * self.l96(xa+k1/2)
    
        k3 = self.dt * self.l96(xa+k2/2)
    
        k4 = self.dt * self.l96(xa+k3)
    
        return xa + (0.5*k1 + k2 + k3 + 0.5*k4)/3.0

    def l96_t(self, x, dx):
        l = np.zeros_like(x)
        l = (np.roll(x, -1, axis=0) - np.roll(x, 2, axis=0)) * np.roll(dx, 1, axis=0) + \
            (np.roll(dx, -1, axis=0) - np.roll(dx, 2, axis=0)) * np.roll(x, 1, axis=0) - dx
        return l

    def step_t(self, x, dx):
        k1 = self.dt * self.l96(x)
        dk1 = self.dt * self.l96_t(x, dx)
    
        k2 = self.dt * self.l96(x+k1/2)
        dk2 = self.dt * self.l96_t(x+k1/2, dx+dk1/2)
    
        k3 = self.dt * self.l96(x+k2/2)
        dk3 = self.dt * self.l96_t(x+k2/2, dx+dk2/2)
    
        k4 = self.dt * self.l96(x+k3)
        dk4 = self.dt * self.l96_t(x+k3, dx+dk3)
    
        return dx + (0.5*dk1 + dk2 + dk3 + 0.5*dk4)/3.0

    def l96_adj(self, x, dx):
        l = np.zeros_like(x)
        l = np.roll(x, 2, axis=0) * np.roll(dx, 1, axis=0) + \
            (np.roll(x, -2, axis=0) - np.roll(x, 1, axis=0)) * np.roll(dx, -1, axis=0) - \
            np.roll(x, -1, axis=0) * np.roll(dx, -2, axis=0) - dx
        return l

    def step_adj(self, x, dx):
        k1 = self.dt * self.l96(x)
        x2 = x + 0.5*k1
        k2 = self.dt * self.l96(x2)
        x3 = x + 0.5*k2
        k3 = self.dt * self.l96(x3)
        x4 = x + k3
        k4 = self.dt * self.l96(x4)

        dxa = dx
        dk1 = dx / 6
        dk2 = dx / 3
        dk3 = dx / 3
        dk4 = dx / 6

        dxa = dxa + self.dt * self.l96_adj(x4, dk4)
        dk3 = dk3 + self.dt * self.l96_adj(x4, dk4)

        dxa = dxa + self.dt * self.l96_adj(x3, dk3)
        dk2 = dk2 + 0.5 * self.dt * self.l96_adj(x3, dk3)

        dxa = dxa + self.dt * self.l96_adj(x2, dk2)
        dk1 = dk1 + 0.5 * self.dt * self.l96_adj(x2, dk2)

        dxa = dxa + self.dt * self.l96_adj(x, dk1)

        return dxa

    def calc_dist(self, iloc):
        dist = np.zeros(self.nx)
        for j in range(self.nx):
            dist[j] = abs(self.nx / np.pi * np.sin(np.pi * (iloc - float(j)) / self.nx))
        return dist
    
    def calc_dist1(self, iloc, jloc):
        dist = abs(self.nx / np.pi * np.sin(np.pi * (iloc - jloc) / self.nx))
        return dist

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 16
    n = 40
    F = 8.0
    h = 0.05

    l96 = L96(n, h, F)

    x0 = np.ones(n)*F
    x0[19] += 0.001*F
    tmax = 2.0
    nt = int(tmax/h) + 1

    for k in range(nt):
        x0 = l96(x0)
    fig, ax = plt.subplots(figsize=[6,12],constrained_layout=True)
    cmap = plt.get_cmap('tab10')
    xaxis = np.arange(n)
    ydiff = 100.0
    nt5d = 5 * 4
    icol=0
    for k in range(nt5d):
        x0 = l96(x0)
        if k%2==0:
            ax.plot(xaxis,x0+ydiff,lw=2.0,c=cmap(icol))
            ydiff-=10.0
            icol += 1
    ax.set_xlim(0.0,n-1)
    ax.set_title(f"Lorenz I, N={n}, F={F}")
    fig.savefig(f"l96_n{n}F{int(F)}.png",dpi=300)
    plt.show()

    a = 1e-5
    x0 = np.ones(n)
    dx = np.random.randn(x0.size)
    adx = l96(x0+a*dx)
    ax = l96(x0)
    jax = l96.step_t(x0, dx)
    d = np.sqrt(np.sum((adx-ax)**2)) / a / (np.sqrt(np.sum(jax**2)))
    print("TLM check diff.={}".format(d-1))

    ax = l96.step_t(x0, dx)
    atax = l96.step_adj(x0, ax)
    d = (ax.T @ ax) - (dx.T @ atax)
    print("ADJ check diff.={}".format(d))