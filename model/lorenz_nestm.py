import numpy as np 
import logging
try:
    from .lorenz2m import L05IIm
    from .lorenz3m import L05IIIm
except ImportError:
    from lorenz2m import L05IIm
    from lorenz3m import L05IIIm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
# nesting Lorenz system with multiple advection length scales
# Reference : Lorenz (2005, JAS), Yoon et al. (2012, Tellus A), Kretschmer et al. (2015, Tellus A)
class L05nestm():
    def __init__(self, nx_true, nx_gm, nx_lam, nks_gm, nks_lam, \
        ni, b, c, dt, F, intgm, ist_lam, nsp, \
        lamstep=1, po=1, intrlx=None, gm_same_with_nature=False, debug=False):
        self.model = "l05nestm"
        self.dt6h = 0.05
        # Actual grid
        self.nx_true = nx_true
        self.ix_true = np.arange(self.nx_true,dtype=np.int32)
        self.xaxis_true = self.nx_true / np.pi * np.sin(np.pi * np.arange(self.nx_true) / self.nx_true)
        # Limited-area model (LAM)
        logging.info("LAM")
        self.nx_lam = nx_lam
        self.nks_lam = nks_lam
        self.ni = ni
        self.ghost = max(int(np.ceil(5*np.max(self.nks_lam)/2)),2*self.ni)
        logging.info(f"ghost point={self.ghost}")
        self.b = b
        self.c = c
        self.lamstep = lamstep
        self.dt_gm = dt
        self.nt6h_gm = int(self.dt6h/self.dt_gm)
        self.dt_lam = self.dt_gm / self.lamstep
        self.nt6h_lam = int(self.dt6h/self.dt_lam)
        self.F = F
        self.ix_lam = np.arange(ist_lam,ist_lam+self.nx_lam,dtype=np.int32)
        self.xaxis_lam = self.nx_true / np.pi * np.sin(np.pi * self.ix_lam / self.nx_true)
        if self.ghost > ist_lam:
            self.lghost = ist_lam
            self.rghost = self.nx_true - (ist_lam+self.nx_lam)
            self.ix_lam_ext = self.ix_true.copy()
        else:
            self.lghost = self.ghost
            self.rghost = self.ghost
            self.ix_lam_ext = np.arange(ist_lam-self.ghost,ist_lam+self.nx_lam+self.ghost,dtype=np.int32) # including xaxis lateral boundaries
        self.xaxis_lam_ext = self.nx_true / np.pi * np.sin(np.pi * self.ix_lam_ext / self.nx_true)
        cyclic = False
        self.lam = L05IIIm(self.nx_lam, self.nks_lam, self.ni, \
            self.b, self.c, self.dt_lam, self.F,\
            lghost=self.lghost, rghost=self.rghost, debug=debug, cyclic=cyclic)
        # Global model (GM)
        logging.info("GM")
        self.intgm = intgm # grid interval of GM relative to LAM
        self.nx_gm = nx_gm
        self.nks_gm = nks_gm
        self.ix_gm = np.arange(0,self.nx_gm*self.intgm,self.intgm,dtype=np.int32)
        self.xaxis_gm = self.nx_true / np.pi * np.sin(np.pi * self.ix_gm / self.nx_true)
        self.gm_same_with_nature = gm_same_with_nature
        if self.gm_same_with_nature:
            self.ni_gm = self.ni // intgm
            self.gm = L05IIIm(self.nx_gm, self.nks_gm, self.ni_gm, \
                self.b, self.c, self.dt_lam, self.F)
        else:
            self.gm = L05IIm(self.nx_gm, self.nks_gm, self.dt_gm, self.F)
        # Interpolation matrix
        self.itplmat()
        # Boundary condition
        self.nsp = nsp # sponge region width
        self.intrlx = intrlx # time interval for boundary relaxation
        if self.intrlx is None:
            self.intrlx = self.nt6h_gm
        # relaxation factor
        self.po = po
        self.rlx = np.ones(self.nsp) - (np.abs(np.arange(self.nsp)-self.nsp)/(self.nsp))**self.po
        self.rrlx = self.rlx[::-1]
        logging.info(f"sponge width={self.nsp}")
        ## debug
        if debug:
            plt.plot(np.arange(self.nsp),self.rlx,label='rlx')
            plt.plot(np.arange(self.nsp),1.0-self.rlx,label='1.0-rlx')
            plt.plot(np.arange(self.nsp),self.rrlx,label='rlx[::-1]')
            plt.plot(np.arange(self.nsp),1.0-self.rrlx,label='1.0-rlx[::-1]')
            plt.ylim(0,1)
            plt.xlim(0,self.nsp-1)
            plt.grid()
            plt.legend()
            #plt.savefig(f'lorenz/l05nestm/rlx_nsp{self.nsp}p{self.po}.png',dpi=300)
            plt.show()
            plt.close()
#            logging.debug(f"xaxis_true={self.xaxis_true}")
#            plt.plot(np.arange(self.nx_true),self.xaxis_true,lw=4.0,label='nature')
#            logging.debug(f"xaxis_gm={self.xaxis_gm}")
#            plt.plot(self.ix_gm,self.xaxis_gm,lw=0.0,marker='.',label='GM')
#            logging.debug(f"xaxis_lam={self.xaxis_lam}")
#            plt.plot(self.ix_lam,self.xaxis_lam,lw=2.0,ls='dashed',label='LAM')
#            logging.debug(f"xaxis_lam_ext={self.xaxis_lam_ext}")
#            plt.plot(self.ix_lam_ext,self.xaxis_lam_ext,lw=2.0,ls='dashdot',label='LAM_ext')
#            plt.legend()
#            plt.show()
#            plt.close()

    def get_params(self):
        return self.gm.get_params(), self.lam.get_params(), self.lamstep, self.nsp

    def itplmat(self):
        from math import floor
        self.gm2lamext = np.zeros((self.ix_lam_ext.size,self.ix_gm.size))
        dx_gm = float(self.ix_gm[1] - self.ix_gm[0])
        for i in range(self.ix_lam_ext.size):
            ri = float(self.ix_lam_ext[i])
            ii = floor(ri)
            ig = np.argmin(np.abs(self.ix_gm - ii))
            if self.ix_gm[ig] > ii: ig -= 1
            ai = (ri - float(self.ix_gm[ig]))/dx_gm
            if ig < self.ix_gm.size - 1:
                self.gm2lamext[i,ig] = 1.0 - ai
                self.gm2lamext[i,ig+1] = ai
            else:
                self.gm2lamext[i,ig] = 1.0 - ai
                self.gm2lamext[i,0] = ai

    def __call__(self,x_gm,x_lam):
        ## boundary conditions from previous step
        #gm2lam0 = interp1d(self.ix_gm, x_gm, axis=0)
        x0_gm2lamext = np.dot(self.gm2lamext,x_gm)
        x_lam_ext = np.dot(self.gm2lamext,x_gm)
        x_lam_ext[self.lghost:-self.rghost] = x_lam
        #logging.debug(x_lam_ext.shape)
        #if self.nsp>0:
        #    # Davies relaxation
        #    t_wgt = 0.0
        #    self.bound_rlx(x_lam_ext,t_wgt,x0_gm2lamext,x0_gm2lamext)
        ## integration for 6 hours
        for k in range(self.nt6h_gm):
            #GM
            xf_gm = self.gm(x_gm)
            ## boundary conditions from previous step
            #gm2lam0 = interp1d(self.ix_gm, x_gm, axis=0)
            x0_gm2lamext = np.dot(self.gm2lamext,x_gm)
            ## boundary conditions from next step
            #gm2lam1 = interp1d(self.ix_gm, xf_gm, axis=0)
            x1_gm2lamext = np.dot(self.gm2lamext,xf_gm)
            #LAM
            if self.nsp>0 and k%self.intrlx==0:
                # Davies relaxation
                t_wgt = 0.0
                self.bound_rlx(x_lam_ext,t_wgt,x0_gm2lamext,x0_gm2lamext)
            for i in range(self.lamstep):
                x_lam_ext = self.lam(x_lam_ext)
                t_wgt = float(i+1)/float(self.lamstep)
                #x_lam_ext = gm2lam0(self.ix_lam_ext)
                #t_wgt = float(i)/float(self.lamstep)
                #if self.nsp>0:
                #    # Davies relaxation
                #    self.bound_rlx(x_lam_ext,t_wgt,x0_gm2lamext,x1_gm2lamext)
                ## boundary conditions
                x_lam_ext[:self.lghost] = (1.0-t_wgt)*x0_gm2lamext[:self.lghost] + t_wgt*x1_gm2lamext[:self.lghost]
                x_lam_ext[-self.rghost:] = (1.0-t_wgt)*x0_gm2lamext[-self.rghost:] + t_wgt*x1_gm2lamext[-self.rghost:]
            x_gm = xf_gm.copy()
            #xl_lam_ext, xs_lam_ext = self.lam.decomp(x_lam_ext)
            #x0l_gm2lamext, x0s_gm2lamext = self.lam.decomp(x0_gm2lamext)
            #x1l_gm2lamext, x1s_gm2lamext = self.lam.decomp(x1_gm2lamext)
            #xl_lam_ext[:self.ghost] = t_wgt*x0l_gm2lamext[:self.ghost] + (1.0-t_wgt)*x1l_gm2lamext[:self.ghost]
            #xs_lam_ext[:self.ghost] = 0.0
            #xl_lam_ext[-self.ghost:] = t_wgt*x0l_gm2lamext[-self.ghost:] + (1.0-t_wgt)*x1l_gm2lamext[-self.ghost:]
            #xs_lam_ext[-self.ghost:] = 0.0
            #x_lam_ext = xl_lam_ext + xs_lam_ext
            #x_lam = self.lam(x_lam_ext)
        xf_lam = x_lam_ext[self.lghost:-self.rghost]
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
            x_lam = x_lam_ext[self.lghost:-self.rghost,:].copy()
            x0_gm2lam = x0_gm2lamext[self.lghost:-self.rghost,:].copy()
            x1_gm2lam = x1_gm2lamext[self.lghost:-self.rghost,:].copy()
            x_lam_ext[self.lghost:self.lghost+self.nsp,:] = x_lam[:self.nsp,:]*self.rlx[:,None] + ((1.0-t_wgt)*x0_gm2lam[:self.nsp,:]+t_wgt*x1_gm2lam[:self.nsp,:])*(1.0-self.rlx[:,None])
            #xl_lam[:self.nsp] = xl_lam[:self.nsp]*self.rlx[:,None] + ((1.0-t_wgt)*x0l_gm2lam[:self.nsp]+t_wgt*x1l_gm2lam[:self.nsp])*(1.0-self.rlx[:,None])
            #xs_lam[:self.nsp] = xs_lam[:self.nsp]*self.rlx[:,None]
            x_lam_ext[-self.rghost-self.nsp:-self.rghost,:] = x_lam[-self.nsp:,:]*self.rrlx[:,None] + ((1.0-t_wgt)*x0_gm2lam[-self.nsp:,:]+t_wgt*x1_gm2lam[-self.nsp:,:])*(1.0-self.rrlx[:,None])
            #xl_lam[-self.nsp:] = xl_lam[-self.nsp:]*self.rlx[::-1,None] + ((1.0-t_wgt)*x0l_gm2lam[-self.nsp:]+t_wgt*x1l_gm2lam[-self.nsp:])*(1.0-self.rlx[::-1,None])
            #xs_lam[-self.nsp:] = xs_lam[-self.nsp:]*self.rlx[::-1,None]
        else:
            x_lam = x_lam_ext[self.lghost:-self.rghost].copy()
            x0_gm2lam = x0_gm2lamext[self.lghost:-self.rghost].copy()
            x1_gm2lam = x1_gm2lamext[self.lghost:-self.rghost].copy()
            x_gm2lam=(1.0-t_wgt)*x0_gm2lam+t_wgt*x1_gm2lam
            #logging.debug(self.rlx)
            x_lam_ext[self.lghost:self.lghost+self.nsp] = x_lam[:self.nsp]*self.rlx + x_gm2lam[:self.nsp]*(1.0-self.rlx)
            #xl_lam[:self.nsp] = xl_lam[:self.nsp]*self.rlx[:] + ((1.0-t_wgt)*x0l_gm2lam[:self.nsp]+t_wgt*x1l_gm2lam[:self.nsp])*(1.0-self.rlx[:])
            #xs_lam[:self.nsp] = xs_lam[:self.nsp]*self.rlx[:]
            #logging.debug(f'rlx={self.rrlx}')
            #logging.debug(f'LAM={x_lam_ext[self.ghost+self.nx_lam-self.nsp:self.ghost+self.nx_lam]}')
            #logging.debug(f'GM={x_gm2lam[self.nx_lam-self.nsp:]}')
            x_lam_ext[-self.rghost-self.nsp:-self.rghost] = x_lam[self.nx_lam-self.nsp:]*self.rrlx + x_gm2lam[self.nx_lam-self.nsp:]*(1.0-self.rrlx)
            #logging.debug(f'LAM={x_lam_ext[self.ghost+self.nx_lam-self.nsp:self.ghost+self.nx_lam]}')
            #xl_lam[-self.nsp:] = xl_lam[-self.nsp:]*self.rlx[::-1] + ((1.0-t_wgt)*x0l_gm2lam[-self.nsp:]+t_wgt*x1l_gm2lam[-self.nsp:])*(1.0-self.rlx[::-1])
            #xs_lam[-self.nsp:] = xs_lam[-self.nsp:]*self.rlx[::-1]
        #x_lam_ext[self.ghost:self.ghost+self.nx_lam] = x_lam[:]
        #xl_lam_ext[self.ghost:self.ghost+self.nx_lam] = xl_lam[:]
        #xs_lam_ext[self.ghost:self.ghost+self.nx_lam] = xs_lam[:]
        #x_lam_ext = xl_lam_ext + xs_lam_ext

    def calc_dist_gm(self, iloc):
        dist = np.zeros(self.nx_gm)
        for j in range(self.nx_gm):
            d = 2.0*np.pi*abs(float(self.ix_gm[int(iloc)] - self.ix_gm[j]))/self.nx_true
            dist[j] = min(d,2.0*np.pi-d)
            #dist[j] = abs(self.nx_true*np.sin(np.pi*(self.ix_gm[int(iloc)] - self.ix_gm[j])/self.nx_true)/np.pi)
        return dist
    
    def calc_dist1_gm(self, iloc, jloc):
        dist = 2.0*np.pi*abs(self.ix_gm[int(iloc)] - jloc)/self.nx_true
        dist = min(dist,2.0*np.pi-dist)
        #dist = abs(self.nx_true*np.sin(np.pi*(self.ix_gm[int(iloc)] - jloc)/self.nx_true)/np.pi)
        return dist

    def calc_dist_lam(self, iloc):
        dist = np.zeros(self.nx_lam)
        for j in range(self.nx_lam):
            dist[j] = 2.0*np.pi*abs(float(self.ix_lam[int(iloc)] - self.ix_lam[j]))/self.nx_true
            #dist[j] = abs(self.nx_true*np.sin(np.pi*(self.ix_lam[int(iloc)] - self.ix_lam[j])/self.nx_true)/np.pi)
        return dist
    
    def calc_dist1_lam(self, iloc, jloc):
        dist = 2.0*np.pi*abs(self.ix_lam[int(iloc)] - jloc)/self.nx_true
        #dist = abs(self.nx_true*np.sin(np.pi*(self.ix_lam[int(iloc)] - jloc)/self.nx_true)/np.pi)
        return dist

if __name__ == "__main__":
    from matplotlib.gridspec import GridSpec
    plt.rcParams['font.size'] = 16
    from pathlib import Path
    nx_true = 960
    nx_lam  = 240
    nx_gm   = 240
    intgm   = 4
    nks_lam = [256,128, 64, 32]
    nks_gm  = [ 64, 32, 16,  8]
    ni = 12
    b = 10.0
    c = 0.6
    dt = 0.05 / 36.0
    F = 15.0
    ist_lam = 240
    lamstep = 1
    nsp = 10
    po = 1
    intrlx = 1
    step = L05nestm(nx_true, nx_gm, nx_lam, nks_gm, nks_lam, ni, b, c, dt, F, intgm, ist_lam, nsp, po=po, lamstep=lamstep, intrlx=intrlx, debug=True)

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

    snks_gm = f"{'+'.join([str(n) for n in nks_gm])}"
    snks_lam = f"{'+'.join([str(n) for n in nks_lam])}"
    figdir = Path(f'lorenz/l05nestm/ng{nx_gm}nl{nx_lam}kg{snks_gm}kl{snks_lam}i{ni}/nsp{nsp}p{step.po}intrlx{step.intrlx}')
    if not figdir.exists():
        figdir.mkdir(parents=True)

    x0_gm = np.ones(nx_gm)*F
    x0_gm[nx_gm//2-1] += 0.001*F
    nt = 36*200
    for k in range(nt):
        x0_gm = step.gm(x0_gm)
    gm2lam = interp1d(step.ix_gm,x0_gm)
    x0_lam = gm2lam(step.ix_lam)
    for k in range(20):
        x0_gm, x0_lam = step(x0_gm,x0_lam)
        ## boundary
        if (k+1)%2==0:
            figb,axsb=plt.subplots(ncols=2,figsize=[12,6],constrained_layout=True)
            #gm2lam = interp1d(step.ix_gm,x0_gm)
            x0_lam_ext = np.dot(step.gm2lamext,x0_gm)
            x0_lam_ext[step.lghost:-step.rghost] = x0_lam[:]
            x0l_lam_ext, x0s_lam_ext = step.lam.decomp(x0_lam_ext)
            x0_gm2lamext = np.dot(step.gm2lamext,x0_gm)
            for ax in axsb:
                    ax.plot(step.ix_lam,x0_lam,c='magenta',lw=3.0,label='LAM')
                    ax.plot(step.ix_lam_ext,x0l_lam_ext,c='tab:blue',label='LAM, large')
                    ax.plot(step.ix_lam_ext,x0s_lam_ext,c='tab:orange',label='LAM, small')
                    ax.plot(step.ix_lam_ext,x0_gm2lamext,c='k',label='GM')
                    #if nsp>0:
                    #    step.bound_rlx(x0_lam_ext,0.0,x0_gm2lamext,x0_gm2lamext)
                    #    x0l_lam_ext, x0s_lam_ext = step.lam.decomp(x0_lam_ext)
                    #    ax.plot(step.ix_lam_ext,x0l_lam_ext,c='tab:green',ls='dashed',label='LAM, large, relaxed')
                    #    ax.plot(step.ix_lam_ext,x0s_lam_ext,c='tab:red',ls='dashed',label='LAM, small, relaxed')
                    ax.vlines([step.ix_lam[0],step.ix_lam[-1]],0,1,ls='dashdot',colors='k',transform=ax.get_xaxis_transform())
                    ax.hlines([0],0,1,colors='gray',alpha=0.7,transform=ax.get_yaxis_transform())
                    ax.set_xticks(step.ix_lam_ext[::10])
            axsb[0].set_xlim(step.ix_lam_ext[0],step.ix_lam_ext[step.ghost+max(20,step.nsp*2)])
            axsb[0].fill_between(step.ix_lam_ext, 0, 1, where=step.ix_lam_ext < step.ix_lam[nsp],
                            color='gray', alpha=0.3, transform=axsb[0].get_xaxis_transform())
            axsb[1].set_xlim(step.ix_lam_ext[-step.ghost-max(20,step.nsp*2)-1],step.ix_lam_ext[-1])
            axsb[1].fill_between(step.ix_lam_ext, 0, 1, where=step.ix_lam_ext > step.ix_lam[-nsp-1],
                            color='gray', alpha=0.3, transform=axsb[1].get_xaxis_transform())
            axsb[0].legend()
            figb.suptitle(f"t={(k+1)*6:.1f}h")
            figb.savefig(figdir/f"bounds{k:03d}_F{int(F)}b{int(b)}c{c:.1f}.png",dpi=300)
            plt.show(block=False)
            plt.close(fig=figb)
    fig, ax = plt.subplots()
    ax.plot(step.ix_gm,x0_gm,lw=2.0)
    ax.plot(step.ix_lam,x0_lam,lw=1.0)
    #gm2lam = interp1d(step.ix_gm,x0_gm)
    x0_lam_ext = np.dot(step.gm2lamext,x0_gm)
    x0_lam_ext[step.lghost:-step.rghost] = x0_lam[:]
    plt.plot(step.ix_lam_ext,x0_lam_ext,ls='dashed')
    plt.show(block=False)
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
    yticklabels = []
    nt = int( 10 * 4 ) #* 0.05 / dt )
    icol=0
    for k in range(20,nt+20):
        x0_gm, x0_lam = step(x0_gm,x0_lam)
        #gm2lam = interp1d(step.ix_gm, x0_gm)
        x0_lam_ext = np.dot(step.gm2lamext,x0_gm)
        x0_lam_ext[step.lghost:-step.rghost] = x0_lam[:]
        x0l_lam_ext, x0s_lam_ext = step.lam.decomp(x0_lam_ext)
        if (k+1)%4==0:
            axs[0].plot(step.ix_gm,x0_gm+ydiff,lw=2.0,c=cmap(icol))
            axs[1].plot(step.ix_lam,x0_lam+ydiff,lw=2.0,alpha=0.6,c=cmap(icol))
            x0l_lam = x0l_lam_ext[step.lghost:-step.rghost]
            x0s_lam = x0s_lam_ext[step.lghost:-step.rghost]
            axs[1].plot(step.ix_lam,x0l_lam+ydiff,lw=1.0,ls='dashed',c=cmap(icol))
            axs[1].plot(step.ix_lam,x0s_lam+ydiff,lw=1.0,c=cmap(icol))
            yticks.append(ydiff)
            yticklabels.append(f"{(k+1)*6}h")
            ydiff-=20.0
            icol += 1
        if (k+1)%2==0:
            ## boundary
            figb,axsb=plt.subplots(ncols=2,figsize=[12,6],constrained_layout=True)
            x0_gm2lamext = np.dot(step.gm2lamext,x0_gm)
            for ax in axsb:
                ax.plot(step.ix_lam,x0_lam,c='magenta',lw=3.0,label='LAM')
                ax.plot(step.ix_lam_ext,x0l_lam_ext,c='tab:blue',label='LAM, large')
                ax.plot(step.ix_lam_ext,x0s_lam_ext,c='tab:orange',label='LAM, small')
                ax.plot(step.ix_lam_ext,x0_gm2lamext,c='k',label='GM')
                #if nsp>0:
                #    step.bound_rlx(x0_lam_ext,0.0,x0_gm2lamext,x0_gm2lamext)
                #    x0l_lam_ext, x0s_lam_ext = step.lam.decomp(x0_lam_ext)
                #    ax.plot(step.ix_lam_ext,x0l_lam_ext,c='tab:green',ls='dashed',label='LAM, large, relaxed')
                #    ax.plot(step.ix_lam_ext,x0s_lam_ext,c='tab:red',ls='dashed',label='LAM, small, relaxed')
                ax.vlines([step.ix_lam[0],step.ix_lam[-1]],0,1,ls='dashdot',colors='k',transform=ax.get_xaxis_transform())
                ax.hlines([0],0,1,colors='gray',alpha=0.7,transform=ax.get_yaxis_transform())
                ax.set_xticks(step.ix_lam_ext[::10])
            axsb[0].set_xlim(step.ix_lam_ext[0],step.ix_lam_ext[step.lghost+max(20,step.nsp*2)])
            axsb[0].fill_between(step.ix_lam_ext, 0, 1, where=step.ix_lam_ext < step.ix_lam[nsp],
                        color='gray', alpha=0.3, transform=axsb[0].get_xaxis_transform())
            axsb[1].set_xlim(step.ix_lam_ext[-step.rghost-max(20,step.nsp*2)-1],step.ix_lam_ext[-1])
            axsb[1].fill_between(step.ix_lam_ext, 0, 1, where=step.ix_lam_ext > step.ix_lam[-nsp-1],
                        color='gray', alpha=0.3, transform=axsb[1].get_xaxis_transform())
            axsb[0].legend()
            figb.suptitle(f"t={(k+1)*6:.1f}h")
            figb.savefig(figdir/f"bounds{k:03d}_F{int(F)}b{int(b)}c{c:.1f}.png",dpi=300)
            plt.show(block=False)
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
    axs[0].set_yticklabels(yticklabels)
    #axs[0].set_yticklabels([])
    axs[1].set_yticklabels([])
    axs[0].grid(True)
    axs[1].grid(True)
    axs[0].set_title('GM')
    axs[1].set_title('LAM')
    fig.suptitle(f"Nesting Lorenz, N_gm={nx_gm}, K_gm={snks_gm}"\
        +f"\n N_lam={nx_lam}, K_lam={snks_lam}, I={ni}\n F={F}, b={b}, c={c}, intrlx={step.intrlx}")
    fig.savefig(figdir/f"F{int(F)}b{int(b)}c{c:.1f}.png",dpi=300)
    plt.show()
    plt.close()
