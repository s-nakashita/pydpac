import numpy as np 
import logging
# Lorenz II model with multiple wave lengths
# Reference : Lorenz (2005, JAS) Section 3
class L05IIm():
    def __init__(self, nx, nks, dt, F):
        self.nx = nx
        self.nks = nks
        self.dt = dt
        self.F = F
        logging.info(f"nx={self.nx} F={self.F} dt={self.dt:.3e}")
        logging.info(f"nks={self.nks}")

    def get_params(self):
        return self.nx, self.nks, self.dt, self.F

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
        l = np.zeros_like(x)
        for nk in self.nks:
            l = l + self.adv(nk,x)
        l = l - x + self.F
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
    from matplotlib.ticker import FixedLocator, FixedFormatter
    plt.rcParams['font.size'] = 16
    import sys
    sys.path.append('../plot')
    from nmc_tools import NMC_tools, wnum2wlen, wlen2wnum
    from pathlib import Path

    nx = 240
    nks = [8,16,32,64]
    F = 15.0
    h = 0.05 / 36
    nt6h = int(0.05 / h)
    nt = 100 * 4 * nt6h
    xaxis = np.arange(nx)
    ix_rad = 2.0 * np.pi * xaxis / nx
    nmc = NMC_tools(ix_rad, ttype='c')

    figdir = Path('lorenz/l05IIm')
    if not figdir.exists():
        figdir.mkdir(parents=True)

    l2 = L05IIm(nx, nks, h, F)
    x0 = np.ones(nx)*F
    x0[nx//2-1] += 0.001*F
#    t = []
#    x = [x0]
#    en = []
#    sp = []
#    for k in range(nt):
#        print(f'{k/nt*100:.2f}%')
#        x0 = l2(x0)
#        if (k+1)%nt6h == 0:
#            x.append(x0)
#        if k>nt//10:
#            t.append(k*h)
#            en.append(np.mean(x0**2)/2.)
#            wnum, sp1 = nmc.psd(x0)
#            sp.append(sp1)
#    x = np.array(x)
#    print(x.shape)
#    #np.save(figdir/f"n{nx}k{'+'.join([str(n) for n in nks])}F{int(F)}.npy",x)
#    #exit()

    fig2, axs = plt.subplots(nrows=2,figsize=[6,12],constrained_layout=True)
    days = np.array(t) / 0.05 / 4
    axs[0].plot(days,en)
    axs[0].set_xlabel('days')
    axs[0].set_title(r'$\overline{X^2}/2$')
    axs[1].semilogy(wnum, np.array(sp).mean(axis=0))
    axs[1].set(xlabel=r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)",title='variance power spectra')
    #axs[1].set_xscale('log')
    #axs[1].xaxis.set_major_locator(FixedLocator([240./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi]))
    #axs[1].xaxis.set_major_formatter(FixedFormatter([r'$\frac{240}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$']))
    axs[1].xaxis.set_major_locator(FixedLocator([480,240,120,60,30,1]))
    axs[1].xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','1']))
    #axs[1].set_xlim(0.5/np.pi,wnum[-1])
    secax = axs[1].secondary_xaxis('top',functions=(wnum2wlen, wlen2wnum))
    secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
    secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
    secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
    fig2.savefig(figdir/f"en+psd_n{nx}k{'+'.join([str(n) for n in nks])}F{int(F)}.png",dpi=300)
    
    fig, ax = plt.subplots(figsize=[6,12],constrained_layout=True)
    cmap = plt.get_cmap('tab10')
    xaxis = np.arange(nx)
    ydiff = 100.0
    nt5d = 5 * 4 * nt6h
    icol=0
    for k in range(nt5d):
        x0 = l2(x0)
        if k%(2*nt6h)==0:
            ax.plot(xaxis,x0+ydiff,lw=2.0,c=cmap(icol))
            ydiff-=10.0
            icol += 1
    ax.set_xlim(0.0,nx-1)
    ax.set_title(f"Lorenz IIm, N={nx}\nK={'+'.join([str(n) for n in nks])}, F={F}")
    fig.savefig(figdir/f"n{nx}k{'+'.join([str(n) for n in nks])}F{int(F)}.png",dpi=300)
    plt.show()
    #exit()

    nt1yr = nt6h * 4 * 365 # 1 year
    ksave = nt6h # 6 hours
    zsave = []
    for k in range(nt1yr):
        print(f'{k/nt1yr*100:.2f}%')
        x0 = l2(x0)
        if k%ksave==0:
            zsave.append(x0)
    datadir = Path('../data/l05IIm')
    if not datadir.exists():
        datadir.mkdir(parents=True)
    np.save(datadir/'truth.npy',np.array(zsave))
