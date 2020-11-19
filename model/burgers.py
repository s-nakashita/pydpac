import numpy as np
import matplotlib.pyplot as plt

class Bg():
    def __init__(self, dx, dt, nu):
        self.dx = dx
        self.dt = dt
        self.nu = nu

    def l_operator(self, f, u):
# NB u is defined at integral index
# f[i-1/2] = (f[i-1] + f[i]) / 2
# f[i+1/2] = (f[i] + f[i+1]) / 2
# f[i+1/2] - f[i-1/2] = (f[i+1] - f[i-1]) / 2
        l = np.zeros_like(u)
        l[1:-1] = -0.5 * (f[2:] - f[0:-2]) / self.dx \
            + self.nu / self.dx**2 * (u[2:] - 2 * u[1:-1] + u[0:-2])
        return l


    def step(self, u):
        f = 0.5 * u**2
        #print("f={}".format(f))
        u1 = u + self.dt * self.l_operator(f, u)
        return 0.5 * (u + u1 + self.dt * self.l_operator(f, u1))


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
    
    burgers = Bg(dx, dt, nu)

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
        u = burgers.step(u)
    fig.savefig("u.png")
