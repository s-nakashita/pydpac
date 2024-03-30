import numpy as np 
import logging
# Lorenz II model
# Reference : Lorenz (2005, JAS) Section 3
class L05II():
    def __init__(self, nx, nk, dt, F):
        self.nx = nx
        self.nk = nk
        self.dt = dt
        self.F = F
        logging.info(f"nx={self.nx} nk={self.nk}")
        logging.info(f"F={self.F} dt={self.dt:.3e}")

    def get_params(self):
        return self.nx, self.nk, self.dt, self.F

    def adv(self,K,x,y=None):
        ## calcurate [X,Y] terms
        if K%2==0:
            sumdiff = True
            J = K // 2
        else:
            sumdiff = False
            J = (K-1)//2
        w = np.zeros_like(x)
        for j in range(-J, J+1):
            w = w + np.roll(x,-j,axis=0)
        if sumdiff:
            w = w - 0.5*np.roll(x,-J,axis=0) - 0.5*np.roll(x,J,axis=0)
        w = w / K
        if y is not None:
            w2 = np.zeros_like(y)
            for j in range(-J, J+1):
                w2 = w2 + np.roll(y,-j,axis=0)
            if sumdiff:
                w2 = w2 - 0.5*np.roll(y,-J,axis=0) - 0.5*np.roll(y,J,axis=0)
            w2 = w2 / K
        else:
            y = x
            w2 = w
        ladv = np.zeros_like(x)
        for j in range(-J, J+1):
            ladv = ladv + np.roll(w,K-j,axis=0)*np.roll(y,-K-j,axis=0)
        if sumdiff:
            ladv = ladv - 0.5*np.roll(w,K+J,axis=0)*np.roll(y,-K+J,axis=0) \
                - 0.5*np.roll(w,K-J,axis=0)*np.roll(y,-K-J,axis=0)
        ladv = ladv / K
        ladv = ladv - np.roll(w,2*K,axis=0)*np.roll(w2,K,axis=0)
        return ladv

    def tend(self,x):
        l = self.adv(self.nk,x) - x + self.F
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
            dist[j] = abs(iloc - float(j))
            dist[j] = min(dist[j],self.nx-dist[j])
        return dist
    
    def calc_dist1(self, iloc, jloc):
        dist = abs(iloc - jloc)
        dist = min(dist,self.nx-dist)
        return dist

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    plt.rcParams['font.size'] = 16
    nx = 240
    nk = 8
    F = 15.0
    h = 0.05
    tmax = 20.0
    nt = int(tmax/h)
    xaxis = np.arange(nx)
    figdir = Path('lorenz/l05II')
    if not figdir.exists():
        figdir.mkdir(parents=True)

    l2 = L05II(nx, nk, h, F)
    x0 = np.ones(nx)*F
    x0[nx//2-1] += 0.0001*F
    print(x0)
    x=[x0]
    for k in range(nt):
        x0 = l2(x0)
        x.append(x0)
        if k==0: print(x0)
    print(x0)
    x = np.array(x)
    print(x.shape)
    np.save(figdir/f"n{nx}k{nk}F{int(F)}.npy",x)
    exit()
    fig, ax = plt.subplots(figsize=[6,12],constrained_layout=True)
    cmap = plt.get_cmap('tab10')
    xaxis = np.arange(nx)
    ydiff = 100.0
    nt5d = 5 * 4
    icol=0
    for k in range(nt5d):
        x0 = l2(x0)
        if k%2==0:
            ax.plot(xaxis,x0+ydiff,lw=2.0,c=cmap(icol))
            ydiff-=10.0
            icol += 1
    ax.set_xlim(0.0,nx-1)
    ax.set_title(f"Lorenz II, N={nx}, K={nk}, F={F}")
    fig.savefig(figdir/f"n{nx}k{nk}F{int(F)}.png",dpi=300)

    fig, ax = plt.subplots(figsize=[8,8],constrained_layout=True)
    ydiff = 100.0
    for nk in [2,4,8,16,32,64]:
        l2 = L05II(nx, nk, h, F)

        x0 = np.ones(nx)*F
        x0[nx//2-1] += 0.001*F

        for k in range(nt):
            x0 = l2(x0)
        ax.plot(xaxis,x0+ydiff,label=f'K={nk}')
        ydiff-=20.0
    ax.set_xlim(0.0,nx-1)
    ax.set_title(f"Lorenz II, N={nx}, F={F}")
    ax.legend(loc='upper left',bbox_to_anchor=(1.0,0.8))
    #print(x0)
    fig.savefig(figdir/f"n{nx}F{int(F)}.png",dpi=300)
    plt.show()
