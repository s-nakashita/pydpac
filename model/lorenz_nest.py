import numpy as np 
from lorenz2 import L05II
from lorenz3 import L05III
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# nesting Lorenz system
# Reference : Kretschmer et al. (2015, Tellus A)
class L05nest():
    def __init__(self, nx_true, nx_gm, nx_lam, nk_gm, nk_lam, \
        ni, b, c, dt, F,\
        intgm, ist_lam, nsp):
        # Actual grid
        self.nx_true = nx_true
        self.xaxis_true = self.nx_true / np.pi * np.sin(np.pi * np.arange(self.nx_true) / self.nx_true)
        # Limited-area model (LAM)
        print("LAM")
        self.nx_lam = nx_lam
        self.nk_lam = nk_lam
        self.ni = ni
        self.b = b
        self.c = c
        self.dt = dt
        self.F = F
        self.ix_lam = np.arange(ist_lam,ist_lam+self.nx_lam)
        self.xaxis_lam = self.nx_true / np.pi * np.sin(np.pi * self.ix_lam / self.nx_true)
        self.ix_lam_ext = np.arange(ist_lam-self.nk_lam,ist_lam+self.nx_lam+self.nk_lam) # including xaxiseral boundaries
        self.xaxis_lam_ext = self.nx_true / np.pi * np.sin(np.pi * self.ix_lam_ext / self.nx_true)
        self.lam = L05III(self.xaxis_lam_ext.size, self.nk_lam, self.ni, self.b, self.c, self.dt, self.F)
        # Global model (GM)
        print("GM")
        self.intgm = intgm # grid interval of GM rexaxisive to LAM
        self.nx_gm = nx_gm
        self.nk_gm = nk_gm
        self.ix_gm = np.arange(0,self.nx_gm*self.intgm,self.intgm)
        self.xaxis_gm = self.nx_true / np.pi * np.sin(np.pi * self.ix_gm / self.nx_true)
        self.gm = L05II(self.nx_gm, self.nk_gm, self.dt, self.F)
        # Boundary condition
        self.nsp = nsp # sponge region width
        self.rlx = np.arange(1,self.nsp+1)/(self.nsp+1) # relaxation factor
        print(f"sponge width={self.nsp}")
        ## debug
        #plt.plot(self.rlx)
        #plt.show()
        #plt.close()
        #print(f"xaxis_true={self.xaxis_true}")
        #plt.plot(np.arange(self.nx_true),self.xaxis_true,lw=4.0,label='Actual')
        #print(f"xaxis_gm={self.xaxis_gm}")
        #plt.plot(self.ix_gm,self.xaxis_gm,lw=0.0,marker='.',label='GM')
        #print(f"xaxis_lam={self.xaxis_lam}")
        #plt.plot(self.ix_lam,self.xaxis_lam,lw=2.0,ls='dashed',label='LAM')
        #print(f"xaxis_lam_ext={self.xaxis_lam_ext}")
        #plt.plot(self.ix_lam_ext,self.xaxis_lam_ext,lw=2.0,ls='dashdot',label='LAM_ext')
        #plt.legend()
        #plt.show()
        #plt.close()

    def get_params(self):
        return self.gm.get_params(), self.lam.get_params()

    def __call__(self,x_gm,x_lam):
        #GM
        xf_gm = self.gm(x_gm)
        #LAM
        gm2lam = interp1d(self.ix_gm, x_gm)
        x_lam_ext = gm2lam(self.ix_lam_ext)
        x_lam_ext[self.nk_lam:self.nk_lam+self.nx_lam] = x_lam[:]
        xf_lam_ext = self.lam(x_lam_ext)
        xf_lam = xf_lam_ext[self.nk_lam:self.nk_lam+self.nx_lam]
        # Davies relaxation
        gm2lam = interp1d(self.ix_gm, xf_gm)
        xf_gm2lam = gm2lam(self.ix_lam)
        xf_lam[:self.nsp]  = xf_lam[:self.nsp]*(1.0-self.rlx[::-1]) + xf_gm2lam[:self.nsp]*self.rlx[::-1]
        xf_lam[-self.nsp:] = xf_lam[-self.nsp:]*(1.0-self.rlx[:]) + xf_gm2lam[-self.nsp:]*self.rlx[::-1]
        return xf_gm, xf_lam

if __name__ == "__main__":
    from matplotlib.gridspec import GridSpec
    plt.rcParams['font.size'] = 16
    nx_true = 960
    nx_lam  = 240
    nx_gm   = 240
    intgm   = 4
    nk_lam  = 32
    nk_gm   = 8
    ni = 12
    b = 10.0
    c = 0.6
    dt = 0.05 / 36.0
    F = 15.0
    ist_lam = 60
    nsp = 10
    step = L05nest(nx_true, nx_gm, nx_lam, nk_gm, nk_lam, ni, b, c, dt, F, intgm, ist_lam, nsp)
    
    x0_gm = np.ones(nx_gm)*F
    x0_gm[nx_gm//2-1] += 0.001*F
    nt = 36*200
    for k in range(nt):
        x0_gm = step.gm(x0_gm)
    gm2lam = interp1d(step.ix_gm,x0_gm)
    x0_lam = gm2lam(step.ix_lam)
    fig, ax = plt.subplots()
    ax.plot(step.ix_gm,x0_gm,lw=2.0)
    ax.plot(step.ix_lam,x0_lam,lw=1.0)
    plt.show()
    plt.close()

    fig = plt.figure(figsize=[12,10],constrained_layout=True)
    gs = GridSpec(1,3,figure=fig)
    axs = []
    ax = fig.add_subplot(gs[:2])
    axs.append(ax)
    ax = fig.add_subplot(gs[2])
    axs.append(ax)
    cmap = plt.get_cmap('tab10')
    ydiff = 100.0
    nt5d = int( 5 * 4 * 0.05 / dt )
    icol=0
    for k in range(nt5d):
        x0_gm, x0_lam = step(x0_gm,x0_lam)
        if k%int(2*0.05/dt)==0:
            axs[0].plot(step.ix_gm,x0_gm+ydiff,lw=2.0,c=cmap(icol))
            axs[1].plot(step.ix_lam,x0_lam+ydiff,lw=2.0,c=cmap(icol))
            ydiff-=10.0
            icol += 1
    axs[0].vlines([step.ix_lam[0],step.ix_lam[-1]],0,1,\
        colors='k',linestyle='dashdot',transform=axs[0].get_xaxis_transform())
    axs[0].set_xlim(step.ix_gm[0],step.ix_gm[-1])
    axs[1].set_xlim(step.ix_lam[0],step.ix_lam[-1])
    fig.suptitle(f"Nesting Lorenz, N_gm={nx_gm}, K_gm={nk_gm}"\
        +f"\n N_lam={nx_lam}, K_lam={nk_lam}, F={F}")
    fig.savefig(f"l05nest_ng{nx_gm}nl{nx_lam}kg{nk_gm}kl{nk_lam}F{int(F)}.png",dpi=300)
    plt.show()