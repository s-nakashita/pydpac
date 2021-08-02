import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft

class Bve():
    def __init__(self, nx, dt, omega):
        self.nx = nx
        self.lam = np.array([2.0*np.pi / self.nx * i for i in range(self.nx)])
        self.dt = dt
        self.omega = omega
        self.tn = int((nx - 1) / 3)
        self.imag = 1.0j
        self.freq = fft.fftfreq(self.nx, d=1.0/self.nx)

    def get_params(self):
        return self.nx, self.dt, self.omega
    
    def get_lam(self):
        return self.lam

    def f_operator(self, phin):
        fn = np.zeros_like(phin)
        #print(type(fn[0]))
        for i in range(1, self.freq.size):
            fn[i] = self.imag * 2.0 * self.omega * phin[i] / self.freq[i]
            #a = -2.0*self.omega*phin[i].imag / freq[i]
            #b = 2.0*self.omega*phin[i].real / freq[i]
            #fn[i] = complex(a, b)
        return fn
    
    def __call__(self, phi):
        phin = fft.fft(phi,axis=0)
        #print(phin.shape)
        
        k1 = self.dt * self.f_operator(phin)
        k1[0] = 0.0
        k2 = self.dt * self.f_operator(phin+k1/2)
        k2[0] = 0.0
        k3 = self.dt * self.f_operator(phin+k2/2)
        k3[0] = 0.0
        k4 = self.dt * self.f_operator(phin+k3)
        k4[0] = 0.0
        phin = phin + (0.5*k1 + k2 + k3 + 0.5*k4)/3.0
        return fft.ifft(phin,axis=0).real

    def step_t(self, phib, phi):
        return self.__call__(phi)

    def step_adj(self, phib, phi):
        phin = fft.fft(phi,axis=0)
        
        dphi = phin 
        dk1 = dphi / 6.0
        dk2 = dphi / 3.0
        dk3 = dphi / 3.0
        dk4 = dphi / 6.0
        
        f = -self.f_operator(dk4)
        f[0] = 0.0
        dphi = dphi + self.dt * f
        dk3 = dk3 + self.dt * f
        
        f = -self.f_operator(dk3)
        f[0] = 0.0
        dphi = dphi + self.dt * f
        dk2 = dk2 + 0.5 * self.dt * f
        
        f = -self.f_operator(dk2)
        f[0] = 0.0
        dphi = dphi + self.dt * f
        dk1 = dk1 + 0.5 * self.dt * f
       
        f = -self.f_operator(dk1)
        f[0] = 0.0
        dphi = dphi + self.dt * f

        return fft.ifft(dphi,axis=0).real

    def rh(self, phi0, nm):
        phi = np.zeros_like(self.lam)
        for i in range(nm.size):
            #phi += phi0[i] * np.exp(-self.imag*nm[i]*lam)
            #phi += phi0[i] * np.exp(self.imag*nm[i]*lam)
            if nm[i] != 0:
                phi += 2.0 * phi0[i] * np.cos(nm[i]*self.lam)
            else:
                phi += phi0[i] * np.ones_like(self.lam)
        return phi 

    def analytical(self, phi0, nm, time):
        phit = np.zeros((time.size,self.lam.size))
        for t in range(time.size):
            for i in range(nm.size):
                if nm[i] != 0:
                    crh = - 2.0 * self.omega / nm[i] / nm[i]
                    #phit[t] += phi0[i] * np.exp(-self.imag * nm[i] * (lam - crh * time[t]))
                    #phit[t] += phi0[i] * np.exp(self.imag * nm[i] * (lam - crh * time[t]))
                    phit[t] += 2.0 * phi0[i] * np.cos(nm[i] * (self.lam - crh * time[t]))
                else:
                    phit[t] += phi0[i] * np.ones_like(self.lam)
        return phit 

if __name__ == "__main__":
    nx = 64
    dt = 0.1
    omega = 2.0 * np.pi 
    tmax = 2.0
    tplot = 0.4
    ntmax = int(tmax / dt) + 1

    nm = np.array([0, 3, 4, 6])
    phi0 = np.array([1.0, 0.5, 0.8, 0.2])

    bve = Bve(nx, dt, omega)
    lam = bve.get_lam()
    lon = lam * 180.0 / np.pi

    time = np.linspace(0.0, tmax, ntmax)
    phit = bve.analytical(phi0, nm, time)
    phi = bve.rh(phi0, nm)
    
    fig, ax = plt.subplots()
    ax.plot(lon, phi, label="t=0, exp")
    ax.plot(lon, phit[0], linestyle="dashed", label="t=0, true")
    for i in range(1,ntmax):
        t = time[i]
        phi = bve(phi)
        if t % tplot == 0:
            ax.plot(lon, phi, label="t={}, exp".format(t))
            ax.plot(lon, phit[i], linestyle="dashed", label="t={}, true".format(t))
    ax.legend()
    fig.savefig("BVE.png")

    a = 1e-5
    phi = bve.rh(phi0, nm)
    dphi = np.random.randn(phi.size)
    adx = bve(phi+a*dphi)
    ax = bve(phi)
    jax = bve.step_t(phi,dphi)
    d = np.sqrt(np.sum((adx-ax)**2)) / a / (np.sqrt(np.sum(jax**2)))
    print("TLM check diff.={}".format(d-1))

    ax = bve.step_t(phi,dphi)
    atax = bve.step_adj(phi,ax)
    d = (ax.T @ ax) - (dphi.T @ atax)
    print("ADJ check diff.={}".format(d))