import numpy as np 
from lorenz2 import L05II
# Lorenz III model
# Reference : Lorenz (2005, JAS) Section 4
class L05III():
    def __init__(self, nx, nk, ni, b, c, dt, F):
        self.nx = nx
        self.nk = nk
        self.ni = ni
        i2 = self.ni*self.ni
        i3 = i2*self.ni
        i4 = i3*self.ni
        self.al = (3.0*i2+3.0)/(2.0*i3+4.0*self.ni)
        self.be = (2.0*i2+1.0)/(i4+2.0*i2)
        #self.al = 1.0 / self.ni
        #self.be = 1.0 / i2
        print(f"ni={self.ni} alpha={self.al:.3f} beta={self.be:.3f}")
        #filtering matrix
        self.filmat = np.zeros((self.nx,self.nx))
        for i in range(self.nx):
            js = i - self.ni
            if js<0: js+=self.nx
            je = i + self.ni + 1
            if je>self.nx: je-=self.nx
            for j in range(self.nx):
                tmp = 0.0
                if js<je:
                    if j>=js and j<je:
                        tmp = self.al - self.be*np.abs(j-i)
                else:
                    if j<je or j>=js:
                        tmp = self.al - self.be*min(np.abs(j-i),self.nx-np.abs(j-i))
                if j==js or j==(je-1):
                    tmp*=0.5
                self.filmat[i,j] = tmp
        ## debug
        print(self.filmat.max(),self.filmat.min())
        #import matplotlib.pyplot as plt
        #plt.plot(self.filmat[self.nx//2,:])
        ##plt.matshow(self.filmat)
        ##plt.colorbar()
        #plt.show()
        #plt.close()
        #ztmp = 0.1*(np.arange(self.nx)-self.nx//2)**2 + 1.0
        #plt.plot(ztmp)
        #plt.plot(np.dot(self.filmat,ztmp))
        #plt.show()
        #plt.close()
        ## debug
        self.b = b 
        self.c = c
        self.dt = dt
        self.F = F
        self.l2 = L05II(self.nx,self.nk,self.dt,self.F)
        print(f"b={self.b} c={self.c}")

    def get_params(self):
        return self.nx, self.nk, self.ni, self.b, self.c, self.dt, self.F

    def decomp(self,z):
        # decompose Z to X and Y
        x = np.dot(self.filmat,z)
        y = z - x
        return x, y

    def tend(self,z):
        x, y = self.decomp(z)
        adv1 = self.l2.adv(self.nk, x)
        #adv2 = self.l2.adv(1, y)
        adv2 = (np.roll(y,-1,axis=0) - np.roll(y,2,axis=0))*np.roll(y,1,axis=0)
        #adv3 = self.l2.adv(1, y, y=x)
        adv3 = np.roll(y,-1,axis=0)*np.roll(x,1,axis=0) - np.roll(y,1,axis=0)*np.roll(x,2,axis=0)
        l = adv1 + self.b*self.b*adv2 + self.c*adv3 - x - self.b*y + self.F
        return l

    def __call__(self,x):
        k1 = self.dt * self.tend(x)
    
        k2 = self.dt * self.tend(x+k1/2)
    
        k3 = self.dt * self.tend(x+k2/2)
    
        k4 = self.dt * self.tend(x+k3)
    
        return x + (0.5*k1 + k2 + k3 + 0.5*k4)/3.0

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
    nx = 960
    nk = 32
    ni = 12
    F = 15.0
    b = 10.0
    c = 2.5
    h = 0.05 / b
    #for ni in [20,40,80]:
    l3 = L05III(nx,nk,ni,b,c,h,F)
    #exit()

    #z0 = np.sin(np.arange(nx)*2.0*np.pi/30.0)
    tmax = 5.0
    nt = int(tmax/h) + 1
    z0 = np.random.rand(nx)
    for k in range(nt):
        z0 = l3(z0)
    x0, y0 = l3.decomp(z0)
    plt.plot(z0)
    plt.plot(x0)
    plt.plot(y0)
    plt.show()
    #exit()

    fig, axs = plt.subplots(ncols=2,figsize=[12,12],sharey=True,constrained_layout=True)
    cmap = plt.get_cmap('tab10')
    xaxis = np.arange(nx)
    ydiff = 100.0
    
    nt = 5 * 4 * int(b)
    icol=0
    for k in range(nt):
        z0 = l3(z0)
        if k%(2*b)==0:
            x0, y0 = l3.decomp(z0)
            axs[0].plot(xaxis,z0+ydiff,lw=2.0,c=cmap(icol))
            axs[1].plot(xaxis,x0+ydiff,lw=2.0,c=cmap(icol))
            axs[1].plot(xaxis,y0*10.0+ydiff,c=cmap(icol))
            ydiff-=10.0
            icol += 1
    for ax in axs:
        ax.set_xlim(0.0,nx-1)
    axs[0].set_title("Z")
    axs[1].set_title(r"X+Y($\times$10)")
    fig.suptitle(f"Lorenz III, N={nx}, K={nk}, I={ni}, F={F}")
    fig.savefig(f"l05III_n{nx}k{nk}i{ni}F{int(F)}.png",dpi=300)
    plt.show()