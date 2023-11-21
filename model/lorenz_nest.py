import numpy as np 
try:
    from .lorenz2 import L05II
    from .lorenz3 import L05III
except ImportError:
    from lorenz2 import L05II
    from lorenz3 import L05III
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# nesting Lorenz system
# Reference : Kretschmer et al. (2015, Tellus A)
class L05nest():
    def __init__(self, nx_true, nx_gm, nx_lam, nk_gm, nk_lam, \
        ni, b, c, dt, F,\
        intgm, ist_lam, nsp, lamstep=4):
        # Actual grid
        self.nx_true = nx_true
        self.ix_true = np.arange(self.nx_true,dtype=np.int32)
        self.xaxis_true = self.nx_true / np.pi * np.sin(np.pi * np.arange(self.nx_true) / self.nx_true)
        # Limited-area model (LAM)
        print("LAM")
        self.nx_lam = nx_lam
        self.nk_lam = nk_lam
        self.ghost = int(np.ceil(5*self.nk_lam/2))
        print(f"ghost point={self.ghost}")
        self.ni = ni
        self.b = b
        self.c = c
        self.lamstep = lamstep
        self.dt_gm = dt
        self.dt_lam = self.dt_gm / lamstep
        self.F = F
        self.ix_lam = np.arange(ist_lam,ist_lam+self.nx_lam,dtype=np.int32)
        self.xaxis_lam = self.nx_true / np.pi * np.sin(np.pi * self.ix_lam / self.nx_true)
        self.ix_lam_ext = np.arange(ist_lam-self.ghost,ist_lam+self.nx_lam+self.ghost,dtype=np.int32) # including xaxis lateral boundaries
        self.xaxis_lam_ext = self.nx_true / np.pi * np.sin(np.pi * self.ix_lam_ext / self.nx_true)
        self.lam = L05III(self.nx_lam, self.nk_lam, self.ni, \
            self.b, self.c, self.dt_lam, self.F,\
            cyclic=False, ghost=self.ghost)
        # Global model (GM)
        print("GM")
        self.intgm = intgm # grid interval of GM relative to LAM
        self.nx_gm = nx_gm
        self.nk_gm = nk_gm
        self.ix_gm = np.arange(0,self.nx_gm*self.intgm,self.intgm,dtype=np.int32)
        self.xaxis_gm = self.nx_true / np.pi * np.sin(np.pi * self.ix_gm / self.nx_true)
        self.gm = L05II(self.nx_gm, self.nk_gm, self.dt_gm, self.F)
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
        return self.gm.get_params(), self.lam.get_params(), self.lamstep

    def __call__(self,x_gm,x_lam):
        #GM
        xf_gm = self.gm(x_gm)
        #LAM
        ## boundary conditions from previous step
        gm2lam0 = interp1d(self.ix_gm, x_gm, axis=0)
        ## boundary conditions from nest step
        gm2lam1 = interp1d(self.ix_gm, xf_gm, axis=0)
        x_lam_ext = gm2lam0(self.ix_lam_ext)
        x_lam_ext[self.ghost:self.ghost+self.nx_lam] = x_lam
        #print(x_lam_ext.shape)
        for i in range(self.lamstep):
            x_lam_ext = self.lam(x_lam_ext)
            # boundary conditions
            t_wgt = (self.lamstep-i-1)/self.lamstep
            x0_gm2lamext = gm2lam0(self.ix_lam_ext)
            x1_gm2lamext = gm2lam1(self.ix_lam_ext)
            x_lam_ext[:self.ghost] = t_wgt*x0_gm2lamext[:self.ghost] + (1.0-t_wgt)*x1_gm2lamext[:self.ghost]
            x_lam_ext[-self.ghost:] = t_wgt*x0_gm2lamext[-self.ghost:] + (1.0-t_wgt)*x1_gm2lamext[-self.ghost:]
            # Davies relaxation
            #x0_gm2lam = gm2lam0(self.ix_lam)
            #x1_gm2lam = gm2lam1(self.ix_lam)
            x0_gm2lam = x0_gm2lamext[self.ghost:self.ghost+self.nx_lam]
            x1_gm2lam = x1_gm2lamext[self.ghost:self.ghost+self.nx_lam]
            if x_lam_ext.ndim==2:
                x_lam_ext[self.ghost:self.ghost+self.nsp]  = x_lam_ext[self.ghost:self.ghost+self.nsp]*(1.0-self.rlx[::-1,None]) + (t_wgt*x0_gm2lam[:self.nsp]+(1.0-t_wgt)*x1_gm2lam[:self.nsp])*self.rlx[::-1,None]
                x_lam_ext[self.nx_lam+self.ghost-self.nsp:self.nx_lam+self.ghost] = x_lam_ext[self.nx_lam+self.ghost-self.nsp:self.nx_lam+self.ghost]*(1.0-self.rlx[:,None]) + (t_wgt*x0_gm2lam[-self.nsp:]+(1.0-t_wgt)*x1_gm2lam[-self.nsp:])*self.rlx[:,None]
            else:
                x_lam_ext[self.ghost:self.ghost+self.nsp]  = x_lam_ext[self.ghost:self.ghost+self.nsp]*(1.0-self.rlx[::-1]) + (t_wgt*x0_gm2lam[:self.nsp]+(1.0-t_wgt)*x1_gm2lam[:self.nsp])*self.rlx[::-1]
                x_lam_ext[self.nx_lam+self.ghost-self.nsp:self.nx_lam+self.ghost] = x_lam_ext[self.nx_lam+self.ghost-self.nsp:self.nx_lam+self.ghost]*(1.0-self.rlx[:]) + (t_wgt*x0_gm2lam[-self.nsp:]+(1.0-t_wgt)*x1_gm2lam[-self.nsp:])*self.rlx[:]
        xf_lam = x_lam_ext[self.ghost:self.ghost+self.nx_lam]
        return xf_gm, xf_lam

    def calc_dist_gm(self, iloc):
        dist = np.zeros(self.nx_gm)
        for j in range(self.nx_gm):
            dist[j] = abs(self.nx_gm / np.pi * np.sin(np.pi * (self.ix_gm[int(iloc)] - float(self.ix_gm[j])) / self.nx_gm))
        return dist
    
    def calc_dist1_gm(self, iloc, jloc):
        dist = abs(self.nx_gm / np.pi * np.sin(np.pi * (self.ix_gm[int(iloc)] - jloc) / self.nx_gm))
        return dist

    def calc_dist_lam(self, iloc):
        dist = np.zeros(self.nx_lam)
        for j in range(self.nx_lam):
            dist[j] = abs(self.nx_lam / np.pi * np.sin(np.pi * (self.ix_lam[int(iloc)] - float(self.ix_lam[j])) / self.nx_lam))
        return dist
    
    def calc_dist1_lam(self, iloc, jloc):
        dist = abs(self.nx_lam / np.pi * np.sin(np.pi * (self.ix_lam[int(iloc)] - jloc) / self.nx_lam))
        return dist

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
    ist_lam = 240
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
    gm2lam = interp1d(step.ix_gm, x0_gm)
    x0_lam_ext = gm2lam(step.ix_lam_ext)
    x0_lam_ext[step.nk_lam:step.nk_lam+step.nx_lam] = x0_lam[:]
    plt.plot(step.ix_lam_ext,x0_lam_ext,ls='dashed')
    plt.show(block=False)
    plt.close()
    x0l_lam, x0s_lam = step.lam.decomp(x0_lam_ext)
    plt.plot(x0l_lam)
    plt.plot(x0s_lam)
    plt.show(block=False)
    plt.close()
    x0_gm2lam = gm2lam(step.ix_lam)
    plt.plot(x0_lam)
    x0_lam[:step.nsp]  = x0_lam[:step.nsp]*(1.0-step.rlx[::-1]) + x0_gm2lam[:step.nsp]*step.rlx[::-1]
    x0_lam[-step.nsp:] = x0_lam[-step.nsp:]*(1.0-step.rlx[:]) + x0_gm2lam[-step.nsp:]*step.rlx[:]
    plt.plot(x0_lam,ls='dashed')
    plt.show(block=False)
    plt.close()

    fig = plt.figure(figsize=[12,10],constrained_layout=True)
    gs = GridSpec(1,3,figure=fig)
    axs = []
    ax0 = fig.add_subplot(gs[:2])
    axs.append(ax0)
    ax1 = fig.add_subplot(gs[2],sharey=ax0)
    plt.setp(ax1.get_yticklabels(), visible=False)
    axs.append(ax1)
    cmap = plt.get_cmap('tab10')
    ydiff = 200.0
    yticks = []
    nt5d = int( 10 * 4 * 0.05 / dt )
    icol=0
    for k in range(nt5d):
        x0_gm, x0_lam = step(x0_gm,x0_lam)
        if k%int(4*0.05/dt)==0:
            axs[0].plot(step.ix_gm,x0_gm+ydiff,lw=2.0,c=cmap(icol))
            axs[1].plot(step.ix_lam,x0_lam+ydiff,lw=2.0,c=cmap(icol))
            gm2lam = interp1d(step.ix_gm, x0_gm)
            x0_lam_ext = gm2lam(step.ix_lam_ext)
            x0_lam_ext[step.nk_lam:step.nk_lam+step.nx_lam] = x0_lam[:]
            x0l_lam, x0s_lam = step.lam.decomp(x0_lam_ext)
            axs[1].plot(step.ix_lam_ext,x0s_lam+ydiff,lw=1.0,c=cmap(icol))
            yticks.append(ydiff)
            ydiff-=20.0
            icol += 1
    axs[0].vlines([step.ix_lam[0],step.ix_lam[-1]],0,1,\
        colors='k',linestyle='dashdot',transform=axs[0].get_xaxis_transform())
    axs[1].fill_between(step.ix_lam, 0, 1, where=step.ix_lam < step.ix_lam[nsp],
                color='gray', alpha=0.3, transform=axs[1].get_xaxis_transform())
    axs[1].fill_between(step.ix_lam, 0, 1, where=step.ix_lam > step.ix_lam[-nsp],
                color='gray', alpha=0.3, transform=axs[1].get_xaxis_transform())
    axs[0].set_xlim(step.ix_gm[0],step.ix_gm[-1])
    axs[1].set_xlim(step.ix_lam[0],step.ix_lam[-1])
    axs[0].set_yticks(yticks)
    axs[1].set_yticks(yticks)
    axs[0].set_yticklabels([])
    axs[1].set_yticklabels([])
    axs[0].grid(True)
    axs[1].grid(True)
    axs[0].set_title('GM')
    axs[1].set_title('LAM')
    fig.suptitle(f"Nesting Lorenz, N_gm={nx_gm}, K_gm={nk_gm}"\
        +f"\n N_lam={nx_lam}, K_lam={nk_lam}, F={F}, c={c}")
    fig.savefig(f"lorenz/l05nest_ng{nx_gm}nl{nx_lam}kg{nk_gm}kl{nk_lam}F{int(F)}c{c:.1f}.png",dpi=300)
    plt.show()
    plt.close()