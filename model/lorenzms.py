import numpy as np 
# Multi-scale Lorenz96
# Reference : Lorenz (1996), Pathiraja and van Leeuwen (2022)
class L96ms():
    def __init__(self,nx,nz,hx,hz,xi,F,dt):
        self.nx = nx
        self.nz = nz
        self.hx = hx
        self.hz = hz
        self.xi = xi
        self.F = F
        self.dt = dt
        print(f"nx={self.nx} nz={self.nz}")
        print(f"hx={self.hx} hz={self.hz}")
        print(f"xi={self.xi} F={self.F} dt={self.dt}")

    def get_params(self):
        return self.nx,self.nz,self.hx,self.hz,self.xi,self.F,self.dt

    def tend(self,x,v):
        v2d = v.reshape((self.nx,self.nz))
        # subgrid-scale tendency
        u = np.mean(v2d,axis=1)*self.hx
        # large-scale
        dxdt = np.zeros_like(x)
        dxdt = (np.roll(x, -1, axis=0) - np.roll(x, 2, axis=0)) * np.roll(x, 1, axis=0) - x + self.F + u
        # small-scale
        cpl = np.zeros_like(v)
        i=0
        for k in range(self.nx):
            for l in range(self.nz):
                cpl[i] = x[k]
                i+=1
        dvdt = np.zeros_like(v)
        dvdt = (np.roll(v, 1, axis=0) - np.roll(v, -2, axis=0)) * np.roll(v, -1, axis=0) - v + self.hz * cpl
        return dxdt, dvdt

    def __call__(self,x,v):
        dxdt, dvdt = self.tend(x,v)
        k1x = dxdt * self.dt
        k1v = dvdt * self.dt / self.xi
    
        dxdt, dvdt = self.tend(x+k1x/2,v+k1v/2)
        k2x = dxdt * self.dt
        k2v = dvdt * self.dt / self.xi
    
        dxdt, dvdt = self.tend(x+k2x/2,v+k2v/2)
        k3x = dxdt * self.dt
        k3v = dvdt * self.dt / self.xi
    
        dxdt, dvdt = self.tend(x+k3x,v+k3v)
        k4x = dxdt * self.dt
        k4v = dvdt * self.dt / self.xi
    
        xf = x + (0.5*k1x + k2x + k3x + 0.5*k4x)/3.0
        vf = v + (0.5*k1v + k2v + k3v + 0.5*k4v)/3.0
        return xf, vf

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
    from numpy import random
    plt.rcParams['font.size'] = 16
    ## Case 1
    #nx = 9
    #nz = 128
    #hx = -0.8
    #hz = 1.0
    #xi = 1.0 / 128.0
    #F = 10.0
    # Case 2
    nx = 9
    nz = 20
    hx = -2.0
    hz = 1.0
    xi = 0.7
    F = 14.0
    dt = 8.0e-4

    l96ms = L96ms(nx, nz, hx, hz, xi, F, dt)

    x0 = np.ones(nx)*F
    x0[nx//2] += 0.001*F
    v0 = random.randn(nx*nz)*0.0001
    tmax = 2.0
    nt = int(tmax/dt) + 1

    for k in range(nt):
        x0, v0 = l96ms(x0, v0)
    print(x0);print(v0.reshape(nx,nz))

    fig, ax = plt.subplots(figsize=[6,12],constrained_layout=True)
    cmap = plt.get_cmap('tab10')
    xaxis = np.arange(nx)
    vaxis = np.arange(nz*nx)/float(nz)
    ydiff = 100.0
    nt5d = 5 * 4 * int(0.05 / dt)
    nt12h = 2 * int(0.05 / dt)
    icol=0
    xlist = []
    vlist = []
    for k in range(nt5d):
        x0, v0 = l96ms(x0, v0)
        xlist.append(x0)
        vlist.append(v0)
        if k%nt12h==0:
            ax.plot(xaxis,x0+ydiff,lw=2.0,c=cmap(icol))
            ax.plot(vaxis,v0+ydiff,lw=1.0,c=cmap(icol))
            ydiff-=10.0
            icol += 1
    ax.set_xlim(0.0,nx-1)
    ax.set_title(f"Multi-scale Lorenz96, Nx={nx}, Nz={nz}\n F={F}, "\
        +r"$\xi$="+f"{xi:.3f}")
    fig.savefig(f"l96ms_nx{nx}nz{nz}F{int(F)}.png",dpi=300)
    plt.show()
    plt.close()

    x = np.array(xlist)
    v = np.array(vlist).reshape(-1,nx,nz)
    u = np.mean(v,axis=2)*hx
    fig, ax = plt.subplots(figsize=[8,8],constrained_layout=True)
    ax.scatter(x.flatten(),u.flatten(),s=1.0)
    ax.grid(True)
    ax.set_xlabel('X[k]')
    ax.set_ylabel('Sub-grid tendency U[k]')
    ax.set_title(f"Nx={nx}, Nz={nz}, F={F}, "\
        +r"$\xi$="+f"{xi:.3f}")
    fig.savefig(f"l96ms_sgtend_nx{nx}nz{nz}F{int(F)}.png",dpi=300)
    plt.show()
    plt.close()