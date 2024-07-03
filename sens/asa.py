import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize

class ASA():
    def __init__(self,vt,cost,jac,adj,*args):
        self.vt = vt # verification time
        self.cost = cost # routine for calculating J
        self.jac = jac # routine for calculating dJ/dxT
        self.args = args # arguments for cost and jac
        self.adj = adj # Adjoint model operator

    def __call__(self,xb):
        xT = xb[self.vt]
        dJdxT = self.jac(xT,*self.args)
        dJdx0 = dJdxT.copy()
        self.dJdx = [dJdxT]
        for i in range(self.vt):
            dJdx0 = self.adj(xb[self.vt-i-2],dJdx0)
            self.dJdx.append(dJdx0)
        return dJdx0

    def plot_hov(self):
        dJdx = np.array(self.dJdx)
        vlim = max(np.max(dJdx),-np.min(dJdx))
        t = np.arange(dJdx.shape[0])
        x = np.arange(dJdx.shape[1])
        fig, ax = plt.subplots()
        mp = ax.pcolormesh(x,t[::-1],dJdx,shading='auto',norm=Normalize(-vlim,vlim),cmap='coolwarm')
        fig.colorbar(mp,ax=ax,shrink=0.6,pad=0.01)
        plt.show()

    def calc_dxopt(self,xb,dJdx0):
        J = self.cost(xb[self.vt],*self.args)
        optscale = -1.0 * J / np.dot(dJdx0,dJdx0)
        return optscale * dJdx0

    def check_djdx(self,xb,dJdx0,dxopt,model,tlm,title='ASA',plot=True,scale=1.0):
        xp = xb[0] + dxopt
        dxp = dxopt.copy()
        for i in range(self.vt):
            # nonlinear evolution
            xp = model(xp)
            # TLM evolution
            dxp = tlm(xb[i],dxp)
        J = self.cost(xb[self.vt],*self.args)
        res_nl = (self.cost(xp,*self.args) - J)/J 
        res_tl = (self.cost(xb[self.vt]+dxp,*self.args) - J)/J
        
        if plot:
            plt.plot(xb[0],ls='dashed',label='x0')
            plt.plot(dJdx0*scale,label=r'dJdx0$\times$'+f'{scale:.0f}')
            plt.plot(dxopt*scale,label=r'dxopt$\times$'+f'{scale:.0f}')
            plt.grid()
            plt.legend()
            plt.title(f'{title} vt={self.vt}h')
            plt.show()

            fig, axs = plt.subplots(ncols=2)
            axs[0].plot(xb[self.vt],ls='dashed',label='base')
            axs[0].plot(xp,ls='dashed',label='full model')
            axs[1].plot(xp-xb[self.vt],label='full - base')
            axs[1].plot(dxp,label='TLM')
            ymin, ymax = axs[1].get_ylim()
            ylim = max(-ymin,ymax)
            axs[1].set_ylim(-ylim,ylim)
            for ax in axs:
                ax.grid()
                ax.legend()
            plt.show()
        return res_nl, res_tl

