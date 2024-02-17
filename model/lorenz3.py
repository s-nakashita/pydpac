import numpy as np 
import logging
try:
    from lorenz2 import L05II
except ImportError:
    from .lorenz2 import L05II
# Lorenz III model
# Reference : Lorenz (2005, JAS) Section 4
class L05III():
    def __init__(self, nx, nk, ni, b, c, dt, F, cyclic=True, ghost=0, debug=False):
        self.nx = nx
        self.ghost = ghost
        self.nx_gho = self.nx + 2*self.ghost
        self.nk = nk
        self.ni = ni
        i2 = self.ni*self.ni
        i3 = i2*self.ni
        i4 = i3*self.ni
        self.al = (3.0*i2+3.0)/(2.0*i3+4.0*self.ni)
        self.be = (2.0*i2+1.0)/(i4+2.0*i2)
        #self.al = 1.0 / self.ni
        #self.be = 1.0 / i2
        logging.info(f"ni={self.ni} alpha={self.al:.3f} beta={self.be:.3f}")
        #filtering matrix
        self.cyclic = cyclic
        if self.cyclic:
            self.filmat = np.zeros((self.nx,self.nx))
        else:
            self.filmat = np.zeros((self.nx_gho,self.nx_gho+2*self.ni))
        nifil = self.filmat.shape[0]
        njfil = self.filmat.shape[1]
        for i in range(nifil):
            lefthalf=True
            righthalf=True
            js = i - self.ni
            je = i + self.ni + 1
            if not self.cyclic:
                js += self.ni
                je += self.ni
            if js<0:
                #if self.cyclic:
                    js+=njfil
                #else:
                #    js = 0
                #    lefthalf=False
            if je>njfil:
                #if self.cyclic:
                    je-=njfil
                #else:
                #    je=njfil
                #    righthalf=False
            for j in range(njfil):
                tmp = 0.0
                if self.cyclic:
                    jj = j
                else:
                    jj = j - self.ni
                if js<je:
                    if j>=js and j<je:
                        tmp = self.al - self.be*np.abs(jj-i)
                else:
                    if j<je or j>=js:
                        tmp = self.al - self.be*min(np.abs(jj-i),njfil-np.abs(jj-i))
                if (lefthalf and j==js) or (righthalf and j==(je-1)):
                    tmp*=0.5
                self.filmat[i,j] = tmp
        self.b = b 
        self.c = c
        self.dt = dt
        self.F = F
        self.l2 = L05II(self.nx_gho,self.nk,self.dt,self.F)
        logging.info(f"b={self.b} c={self.c}")
        if debug:
            logging.debug(self.filmat.max(),self.filmat.min())
            import matplotlib.pyplot as plt
            #plt.plot(self.filmat[self.nx_gho//2,:])
            plt.matshow(self.filmat)
            plt.colorbar()
            plt.show()
            plt.close()
            #plt.plot(self.filmat[0,:])
            #plt.plot(self.filmat[self.nx_gho//2,:])
            #plt.plot(self.filmat[-1,:])
            #plt.plot([self.filmat[i-1,i] for i in range(1,self.nx_gho)],lw=0.0,marker='o')
            #plt.plot([self.filmat[i+1,i] for i in range(self.nx_gho-1)],lw=0.0,marker='x')
            #plt.show()
            #plt.close()
            #ztmp = 0.1*(np.arange(self.nx)-self.nx//2)**2 + 1.0
            #plt.plot(ztmp)
            #plt.plot(np.dot(self.filmat,ztmp))
            #plt.show()
            #plt.close()

    def get_params(self):
        return self.nx, self.nk, self.ni, self.b, self.c, self.dt, self.F

    def decomp(self,z):
        # decompose Z to X and Y
        if not self.cyclic:
            if z.ndim == 2:
                nz = z.shape[0]
                ztmp = np.zeros((nz+2*self.ni,z.shape[1]))
            else:
                nz = z.size
                ztmp = np.zeros(nz+2*self.ni)
            ztmp[self.ni:self.ni+nz] = z[:]
            for i in range(1,self.ni+1):
                ztmp[self.ni-i] = z[i]
                ztmp[self.ni+nz+i-1] = z[-i]
        else:
            ztmp = z
        x = np.dot(self.filmat,ztmp)
        y = z - x
        return x, y

    def tend(self,z):
        x, y = self.decomp(z)
        adv1 = self.l2.adv(self.nk, x)
        #adv2 = self.l2.adv(1, y)
        adv2 = (np.roll(y,-1,axis=0) - np.roll(y,2,axis=0))*np.roll(y,1,axis=0)
        #adv3 = self.l2.adv(1, y, y=x)
        adv3 = - np.roll(y,2,axis=0)*np.roll(x,1,axis=0) + np.roll(y,1,axis=0)*np.roll(x,-1,axis=0)
        l = adv1 + self.b*self.b*adv2 + self.c*adv3 - x - self.b*y + self.F
        return l

    def __call__(self,z):
        k1 = self.dt * self.tend(z)
    
        k2 = self.dt * self.tend(z+k1/2)
    
        k3 = self.dt * self.tend(z+k2/2)
    
        k4 = self.dt * self.tend(z+k3)
    
        zf = z + (0.5*k1 + k2 + k3 + 0.5*k4)/3.0
        return zf

    def calc_dist(self, iloc):
        dist = np.zeros(self.nx)
        for j in range(self.nx):
            dist[j] = 2.0*np.pi*abs(iloc - float(j))/self.nx
            dist[j] = min(dist[j],2.0*np.pi-dist[j])
            #dist[j] = abs(self.nx*np.sin(np.pi*(iloc - float(j))/self.nx)/np.pi)
        return dist
    
    def calc_dist1(self, iloc, jloc):
        dist = 2.0*np.pi*abs(iloc - jloc)/self.nx
        dist = min(dist,2.0*np.pi-dist)
        #dist = abs(self.nx*np.sin(np.pi*(iloc - jloc)/self.nx)/np.pi)
        return dist

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 16
    from pathlib import Path
    nx = 960
    nk = 32
    ni = 12
    F = 15.0
    b = 10.0
    c = 0.6
    h = 0.05 / b
    #for ni in [20,40,80]:
    l3 = L05III(nx,nk,ni,b,c,h,F,debug=True)
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
            axs[1].plot(xaxis,y0*4.0+ydiff,c=cmap(icol))
            ydiff-=10.0
            icol += 1
    for ax in axs:
        ax.set_xlim(0.0,nx-1)
    axs[0].set_title("Z")
    axs[1].set_title(r"X+Y($\times$4)")
    fig.suptitle(f"Lorenz III, N={nx}, K={nk}, I={ni}, F={F}, b={b}, c={c}")
    fig.savefig(f"lorenz/l05III_n{nx}k{nk}i{ni}F{int(F)}c{c:.1f}.png",dpi=300)
    plt.show()

    nt1yr = int(0.05 / h) * 4 * 365 # 1 year
    ksave = int(0.05 / h) # 6 hours
    zsave = []
    for k in range(nt1yr):
        z0 = l3(z0)
        if k%ksave==0:
            zsave.append(z0)
    datadir = Path('../data/l05III')
    if not datadir.exists():
        datadir.mkdir(parents=True)
    np.save(datadir/'truth.npy',np.array(zsave))