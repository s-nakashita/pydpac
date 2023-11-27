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
        intgm, ist_lam, nsp, lamstep=4, po=1, debug=False):
        self.dt6h = 0.05
        # Actual grid
        self.nx_true = nx_true
        self.ix_true = np.arange(self.nx_true,dtype=np.int32)
        self.xaxis_true = self.nx_true / np.pi * np.sin(np.pi * np.arange(self.nx_true) / self.nx_true)
        # Limited-area model (LAM)
        print("LAM")
        self.nx_lam = nx_lam
        self.nk_lam = nk_lam
        self.ni = ni
        self.ghost = max(int(np.ceil(5*self.nk_lam/2)),2*self.ni)
        print(f"ghost point={self.ghost}")
        self.b = b
        self.c = c
        self.lamstep = lamstep
        self.dt_gm = dt
        self.nt6h_gm = int(self.dt6h/self.dt_gm)
        self.dt_lam = self.dt_gm / lamstep
        self.nt6h_lam = int(self.dt6h/self.dt_lam)
        self.F = F
        self.ix_lam = np.arange(ist_lam,ist_lam+self.nx_lam,dtype=np.int32)
        self.xaxis_lam = self.nx_true / np.pi * np.sin(np.pi * self.ix_lam / self.nx_true)
        self.ix_lam_ext = np.arange(ist_lam-self.ghost,ist_lam+self.nx_lam+self.ghost,dtype=np.int32) # including xaxis lateral boundaries
        self.xaxis_lam_ext = self.nx_true / np.pi * np.sin(np.pi * self.ix_lam_ext / self.nx_true)
        self.lam = L05III(self.nx_lam, self.nk_lam, self.ni, \
            self.b, self.c, self.dt_lam, self.F,\
            ghost=self.ghost, debug=debug, cyclic=False)
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
        # relaxation factor
        self.po = po
        self.rlx = 1.0 - (np.abs(np.arange(self.nsp)-self.nsp)/(self.nsp))**self.po
        print(f"sponge width={self.nsp}")
        ## debug
        if debug:
            plt.plot(np.arange(self.nsp),self.rlx)
            plt.plot(np.arange(self.nsp),1.0-self.rlx)
            plt.plot(np.arange(self.nsp),self.rlx[::-1])
            plt.plot(np.arange(self.nsp),1.0-self.rlx[::-1])
            plt.ylim(0,1)
            plt.xlim(0,self.nsp-1)
            plt.grid()
            plt.show()
            plt.close()
#            print(f"xaxis_true={self.xaxis_true}")
#            plt.plot(np.arange(self.nx_true),self.xaxis_true,lw=4.0,label='nature')
#            print(f"xaxis_gm={self.xaxis_gm}")
#            plt.plot(self.ix_gm,self.xaxis_gm,lw=0.0,marker='.',label='GM')
#            print(f"xaxis_lam={self.xaxis_lam}")
#            plt.plot(self.ix_lam,self.xaxis_lam,lw=2.0,ls='dashed',label='LAM')
#            print(f"xaxis_lam_ext={self.xaxis_lam_ext}")
#            plt.plot(self.ix_lam_ext,self.xaxis_lam_ext,lw=2.0,ls='dashdot',label='LAM_ext')
#            plt.legend()
#            plt.show()
#            plt.close()

    def get_params(self):
        return self.gm.get_params(), self.lam.get_params(), self.lamstep, self.nsp

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
            x_lam = self.lam(x_lam_ext)
            t_wgt = float(i+1)/float(self.lamstep)
            #x_lam_ext = gm2lam0(self.ix_lam_ext)
            #t_wgt = float(i)/float(self.lamstep)
            x0_gm2lamext = gm2lam0(self.ix_lam_ext)
            x1_gm2lamext = gm2lam1(self.ix_lam_ext)
            x_lam_ext[self.ghost:self.ghost+self.nx_lam] = x_lam
            if self.nsp>0:
                # Davies relaxation
                #x0_gm2lam = gm2lam0(self.ix_lam)
                #x1_gm2lam = gm2lam1(self.ix_lam)
                self.bound_rlx(x_lam_ext,t_wgt,x0_gm2lamext,x1_gm2lamext)
            # boundary conditions
            x_lam_ext[:self.ghost] = (1.0-t_wgt)*x0_gm2lamext[:self.ghost] + t_wgt*x1_gm2lamext[:self.ghost]
            x_lam_ext[self.ghost+self.nx_lam:] = (1.0-t_wgt)*x0_gm2lamext[self.ghost+self.nx_lam:] + t_wgt*x1_gm2lamext[self.ghost+self.nx_lam:]
            #xl_lam_ext, xs_lam_ext = self.lam.decomp(x_lam_ext)
            #x0l_gm2lamext, x0s_gm2lamext = self.lam.decomp(x0_gm2lamext)
            #x1l_gm2lamext, x1s_gm2lamext = self.lam.decomp(x1_gm2lamext)
            #xl_lam_ext[:self.ghost] = t_wgt*x0l_gm2lamext[:self.ghost] + (1.0-t_wgt)*x1l_gm2lamext[:self.ghost]
            #xs_lam_ext[:self.ghost] = 0.0
            #xl_lam_ext[-self.ghost:] = t_wgt*x0l_gm2lamext[-self.ghost:] + (1.0-t_wgt)*x1l_gm2lamext[-self.ghost:]
            #xs_lam_ext[-self.ghost:] = 0.0
            #x_lam_ext = xl_lam_ext + xs_lam_ext
            #x_lam = self.lam(x_lam_ext)
        xf_lam = x_lam_ext[self.ghost:self.ghost+self.nx_lam]
        return xf_gm, xf_lam

    # Davies relaxation
    def bound_rlx(self,x_lam_ext,t_wgt,x0_gm2lamext,x1_gm2lamext):
        #xl_lam_ext, xs_lam_ext = self.lam.decomp(x_lam_ext)
        #xl_lam = xl_lam_ext[self.ghost:self.ghost+self.nx_lam]
        #xs_lam = xs_lam_ext[self.ghost:self.ghost+self.nx_lam]
        #x0l_gm2lamext, x0s_gm2lamext = self.lam.decomp(x0_gm2lamext)
        #x1l_gm2lamext, x1s_gm2lamext = self.lam.decomp(x1_gm2lamext)
        #x0l_gm2lam = x0l_gm2lamext[self.ghost:self.ghost+self.nx_lam]
        #x1l_gm2lam = x1l_gm2lamext[self.ghost:self.ghost+self.nx_lam]
        if x_lam_ext.ndim==2:
            x_lam = x_lam_ext[self.ghost:self.ghost+self.nx_lam,:].copy()
            x0_gm2lam = x0_gm2lamext[self.ghost:self.ghost+self.nx_lam,:].copy()
            x1_gm2lam = x1_gm2lamext[self.ghost:self.ghost+self.nx_lam,:].copy()
            x_lam_ext[self.ghost:self.ghost+self.nsp,:] = x_lam[:self.nsp,:]*self.rlx[:,None] + ((1.0-t_wgt)*x0_gm2lam[:self.nsp,:]+t_wgt*x1_gm2lam[:self.nsp,:])*(1.0-self.rlx[:,None])
            #xl_lam[:self.nsp] = xl_lam[:self.nsp]*self.rlx[:,None] + ((1.0-t_wgt)*x0l_gm2lam[:self.nsp]+t_wgt*x1l_gm2lam[:self.nsp])*(1.0-self.rlx[:,None])
            #xs_lam[:self.nsp] = xs_lam[:self.nsp]*self.rlx[:,None]
            x_lam[self.ghost+self.nx_lam-self.nsp:self.ghost+self.nx_lam,:] = x_lam[-self.nsp:,:]*self.rlx[::-1,None] + ((1.0-t_wgt)*x0_gm2lam[-self.nsp:,:]+t_wgt*x1_gm2lam[-self.nsp:,:])*(1.0-self.rlx[::-1,None])
            #xl_lam[-self.nsp:] = xl_lam[-self.nsp:]*self.rlx[::-1,None] + ((1.0-t_wgt)*x0l_gm2lam[-self.nsp:]+t_wgt*x1l_gm2lam[-self.nsp:])*(1.0-self.rlx[::-1,None])
            #xs_lam[-self.nsp:] = xs_lam[-self.nsp:]*self.rlx[::-1,None]
            x_lam_ext[self.ghost:self.ghost+self.nx_lam,:] = x_lam[:,:]
        else:
            x_lam = x_lam_ext[self.ghost:self.ghost+self.nx_lam].copy()
            x0_gm2lam = x0_gm2lamext[self.ghost:self.ghost+self.nx_lam].copy()
            x1_gm2lam = x1_gm2lamext[self.ghost:self.ghost+self.nx_lam].copy()
            x_lam_ext[self.ghost:self.ghost+self.nsp] = x_lam[:self.nsp]*self.rlx[:] + ((1.0-t_wgt)*x0_gm2lam[:self.nsp]+t_wgt*x1_gm2lam[:self.nsp])*(1.0-self.rlx[:])
            #xl_lam[:self.nsp] = xl_lam[:self.nsp]*self.rlx[:] + ((1.0-t_wgt)*x0l_gm2lam[:self.nsp]+t_wgt*x1l_gm2lam[:self.nsp])*(1.0-self.rlx[:])
            #xs_lam[:self.nsp] = xs_lam[:self.nsp]*self.rlx[:]
            x_lam_ext[self.ghost+self.nx_lam-self.nsp:self.ghost+self.nx_lam] = x_lam[-self.nsp:]*self.rlx[::-1] + ((1.0-t_wgt)*x0_gm2lam[-self.nsp:]+t_wgt*x1_gm2lam[-self.nsp:])*(1.0-self.rlx[::-1])
            #xl_lam[-self.nsp:] = xl_lam[-self.nsp:]*self.rlx[::-1] + ((1.0-t_wgt)*x0l_gm2lam[-self.nsp:]+t_wgt*x1l_gm2lam[-self.nsp:])*(1.0-self.rlx[::-1])
            #xs_lam[-self.nsp:] = xs_lam[-self.nsp:]*self.rlx[::-1]
        #x_lam_ext[self.ghost:self.ghost+self.nx_lam] = x_lam[:]
        #xl_lam_ext[self.ghost:self.ghost+self.nx_lam] = xl_lam[:]
        #xs_lam_ext[self.ghost:self.ghost+self.nx_lam] = xs_lam[:]
        #x_lam_ext = xl_lam_ext + xs_lam_ext

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
    nx_lam  = 480
    nx_gm   = 240
    intgm   = 4
    nk_lam  = 32
    nk_gm   = 8
    ni = 1
    b = 10.0
    c = 0.6
    dt = 0.05 / 36.0
    F = 15.0
    ist_lam = 240
    nsp = 100
    po = 1
    step = L05nest(nx_true, nx_gm, nx_lam, nk_gm, nk_lam, ni, b, c, dt, F, intgm, ist_lam, nsp, po=po, debug=True)

    lamtest = np.arange(step.ix_lam_ext.size)/step.ix_lam_ext.size
    gmtest = np.ones(step.ix_lam_ext.size)*0.5
    plt.plot(step.ix_lam_ext,lamtest)
    plt.plot(step.ix_lam_ext,gmtest)
    step.bound_rlx(lamtest,0.0,gmtest,gmtest)
    plt.plot(step.ix_lam_ext,lamtest,ls='dashed')
    plt.vlines([step.ix_lam[0],step.ix_lam[-1]],0,1,colors='k',ls='dotted')
    plt.show()
    plt.close()
    #exit()

    x0_gm = np.ones(nx_gm)*F
    x0_gm[nx_gm//2-1] += 0.001*F
    nt = 36*200
    for k in range(nt):
        x0_gm = step.gm(x0_gm)
    gm2lam = interp1d(step.ix_gm,x0_gm)
    x0_lam = gm2lam(step.ix_lam)
    for k in range(36*20):
        x0_gm, x0_lam = step(x0_gm,x0_lam)
        ## boundary
        if k%36==0:
            figb,axsb=plt.subplots(ncols=2,figsize=[12,6],constrained_layout=True)
            gm2lam = interp1d(step.ix_gm,x0_gm)
            x0_lam_ext = gm2lam(step.ix_lam_ext)
            x0_lam_ext[step.ghost:step.ghost+step.nx_lam] = x0_lam[:]
            x0l_lam_ext, x0s_lam_ext = step.lam.decomp(x0_lam_ext)
            x0_gm2lamext = gm2lam(step.ix_lam_ext)
            for ax in axsb:
                    ax.plot(step.ix_lam_ext,x0_lam_ext,c='magenta',lw=3.0,label='LAM')
                    ax.plot(step.ix_lam_ext,x0l_lam_ext,c='tab:blue',label='LAM, large')
                    ax.plot(step.ix_lam_ext,x0s_lam_ext,c='tab:orange',label='LAM, small')
                    ax.plot(step.ix_lam_ext,x0_gm2lamext,c='k',label='GM')
                    if nsp>0:
                        step.bound_rlx(x0_lam_ext,0.0,x0_gm2lamext,x0_gm2lamext)
                        x0l_lam_ext, x0s_lam_ext = step.lam.decomp(x0_lam_ext)
                        ax.plot(step.ix_lam_ext,x0l_lam_ext,c='tab:green',ls='dashed',label='LAM, large, relaxed')
                        ax.plot(step.ix_lam_ext,x0s_lam_ext,c='tab:red',ls='dashed',label='LAM, small, relaxed')
                    ax.vlines([step.ix_lam[0],step.ix_lam[-1]],0,1,ls='dashdot',colors='k',transform=ax.get_xaxis_transform())
                    ax.hlines([0],0,1,colors='gray',alpha=0.7,transform=ax.get_yaxis_transform())
                    ax.set_xticks(step.ix_lam_ext[::10])
            axsb[0].set_xlim(step.ix_lam_ext[0],step.ix_lam_ext[step.ghost+step.nsp*2])
            axsb[0].fill_between(step.ix_lam_ext, 0, 1, where=step.ix_lam_ext < step.ix_lam[nsp],
                            color='gray', alpha=0.3, transform=axsb[0].get_xaxis_transform())
            axsb[1].set_xlim(step.ix_lam_ext[-step.ghost-step.nsp*2-1],step.ix_lam_ext[-1])
            axsb[1].fill_between(step.ix_lam_ext, 0, 1, where=step.ix_lam_ext >= step.ix_lam[-nsp],
                            color='gray', alpha=0.3, transform=axsb[1].get_xaxis_transform())
            axsb[0].legend()
            figb.suptitle(f"t={k*dt/0.05*6:.1f}h")
            #figb.savefig(f"lorenz/l05nest_bounds{k}_ng{nx_gm}nl{nx_lam}kg{nk_gm}kl{nk_lam}i{ni}nsp{nsp}p{step.po}F{int(F)}c{c:.1f}.png",dpi=300)
            plt.show(block=False)
            plt.close(fig=figb)
    fig, ax = plt.subplots()
    ax.plot(step.ix_gm,x0_gm,lw=2.0)
    ax.plot(step.ix_lam,x0_lam,lw=1.0)
    gm2lam = interp1d(step.ix_gm,x0_gm)
    x0_lam_ext = gm2lam(step.ix_lam_ext)
    x0_lam_ext[step.ghost:step.ghost+step.nx_lam] = x0_lam[:]
    plt.plot(step.ix_lam_ext,x0_lam_ext,ls='dashed')
    plt.show() #block=False)
    plt.close()
    #exit()

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
    nt20d = int( 20 * 4 * 0.05 / dt )
    icol=0
    for k in range(nt20d):
        x0_gm, x0_lam = step(x0_gm,x0_lam)
        if (k+1)%int(2*4*0.05/dt)==0:
            axs[0].plot(step.ix_gm,x0_gm+ydiff,lw=2.0,c=cmap(icol))
            axs[1].plot(step.ix_lam,x0_lam+ydiff,lw=2.0,alpha=0.6,c=cmap(icol))
            gm2lam = interp1d(step.ix_gm, x0_gm)
            x0_lam_ext = gm2lam(step.ix_lam_ext)
            x0_lam_ext[step.ghost:step.ghost+step.nx_lam] = x0_lam[:]
            x0l_lam_ext, x0s_lam_ext = step.lam.decomp(x0_lam_ext)
            x0l_lam = x0l_lam_ext[step.ghost:step.ghost+step.nx_lam]
            x0s_lam = x0s_lam_ext[step.ghost:step.ghost+step.nx_lam]
            axs[1].plot(step.ix_lam,x0l_lam+ydiff,lw=1.0,ls='dashed',c=cmap(icol))
            axs[1].plot(step.ix_lam,x0s_lam+ydiff,lw=1.0,c=cmap(icol))
            yticks.append(ydiff)
            ydiff-=20.0
            icol += 1
            ## boundary
            figb,axsb=plt.subplots(ncols=2,figsize=[12,6],constrained_layout=True)
            x0_gm2lamext = gm2lam(step.ix_lam_ext)
            for ax in axsb:
                ax.plot(step.ix_lam_ext,x0_lam_ext,c='magenta',lw=3.0,label='LAM')
                ax.plot(step.ix_lam_ext,x0l_lam_ext,c='tab:blue',label='LAM, large')
                ax.plot(step.ix_lam_ext,x0s_lam_ext,c='tab:orange',label='LAM, small')
                ax.plot(step.ix_lam_ext,x0_gm2lamext,c='k',label='GM')
                if nsp>0:
                    step.bound_rlx(x0_lam_ext,0.0,x0_gm2lamext,x0_gm2lamext)
                    x0l_lam_ext, x0s_lam_ext = step.lam.decomp(x0_lam_ext)
                    ax.plot(step.ix_lam_ext,x0l_lam_ext,c='tab:green',ls='dashed',label='LAM, large, relaxed')
                    ax.plot(step.ix_lam_ext,x0s_lam_ext,c='tab:red',ls='dashed',label='LAM, small, relaxed')
                ax.vlines([step.ix_lam[0],step.ix_lam[-1]],0,1,ls='dashdot',colors='k',transform=ax.get_xaxis_transform())
                ax.hlines([0],0,1,colors='gray',alpha=0.7,transform=ax.get_yaxis_transform())
                ax.set_xticks(step.ix_lam_ext[::10])
            axsb[0].set_xlim(step.ix_lam_ext[0],step.ix_lam_ext[step.ghost+step.nsp*2])
            axsb[0].fill_between(step.ix_lam_ext, 0, 1, where=step.ix_lam_ext < step.ix_lam[nsp],
                        color='gray', alpha=0.3, transform=axsb[0].get_xaxis_transform())
            axsb[1].set_xlim(step.ix_lam_ext[-step.ghost-step.nsp*2-1],step.ix_lam_ext[-1])
            axsb[1].fill_between(step.ix_lam_ext, 0, 1, where=step.ix_lam_ext >= step.ix_lam[-nsp],
                        color='gray', alpha=0.3, transform=axsb[1].get_xaxis_transform())
            axsb[0].legend()
            figb.suptitle(f"t={k*dt/0.05*6:.1f}h")
            figb.savefig(f"lorenz/l05nest_bounds{k}_ng{nx_gm}nl{nx_lam}kg{nk_gm}kl{nk_lam}i{ni}nsp{nsp}p{step.po}F{int(F)}c{c:.1f}.png",dpi=300)
            figb.show()
            plt.close(fig=figb)
    axs[0].vlines([step.ix_lam[0],step.ix_lam[-1]],0,1,\
        colors='k',linestyle='dashdot',transform=axs[0].get_xaxis_transform())
    if nsp>0:
        axs[1].fill_between(step.ix_lam, 0, 1, where=step.ix_lam < step.ix_lam[nsp],
                color='gray', alpha=0.3, transform=axs[1].get_xaxis_transform())
        axs[1].fill_between(step.ix_lam, 0, 1, where=step.ix_lam >= step.ix_lam[-nsp],
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
        +f"\n N_lam={nx_lam}, K_lam={nk_lam}, I={ni}, F={F}, c={c}")
    fig.savefig(f"lorenz/l05nest_ng{nx_gm}nl{nx_lam}kg{nk_gm}kl{nk_lam}i{ni}nsp{nsp}p{step.po}F{int(F)}c{c:.1f}.png",dpi=300)
    plt.show()
    plt.close()