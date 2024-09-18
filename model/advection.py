import numpy as np 
from scipy import fft

## Model for the advection equation
##
## by S. Nakano (Jul. 2021)
## imported by S. Nakashita (Sep. 2024)

class Advection_state:
    def __init__(self, nx, tlag=2, u0=2.0, xnu0=0.0, F0=0.0):
        self.nx = nx
        self.rho = np.zeros((tlag+1, nx))
        self.u = np.full((tlag+1),u0)
        self.forcing = np.full((tlag+1),F0)
        self.xnu = np.full((tlag+1),xnu0)

class Advection():
    def __init__(self, state, dx, dt, tstype=1):
        self.nx = state.nx
        self.dx = dx
        self.dxs = self.dx * self.dx
        self.dt = dt
        self.tstype = tstype
        ## tstype=1: Forward Euler
        ## tstype=2: Leap frog
        ## tstype=3: RK4
    
    def get_params(self):
        return self.nx, self.dx, self.dt

    def __call__(self,state):
        if self.tstype==1:
            self.euler(state)
        elif self.tstype==2:
            self.leapfrog(state)
        elif self.tstype==3:
            self.rk4(state)

    def forward(self, state):
        tlen = state.rho.shape[0]

        state.rho[1:tlen] = state.rho[0:tlen-1]

        ### finite-difference
        #self.A = - state.u[0] * (np.roll(state.rho[1,:],-1)-np.roll(state.rho[1,:],1)) / (2.0*self.dx)
        #if self.tstype==2:
        #    self.S = state.xnu[0] + (np.roll(state.rho[2,:],-1)-2*state.rho[2,:]+np.roll(state.rho[2,:],1)) / self.dxs
        #else:
        #    self.S = state.xnu[0] + (np.roll(state.rho[1,:],-1)-2*state.rho[1,:]+np.roll(state.rho[1,:],1)) / self.dxs
        ## spectral
        rhos = fft.fft(state.rho,axis=1,norm='ortho')
        freq = fft.fftfreq(state.rho.shape[1])
        drhos = rhos[1,] * 2.0j * np.pi * freq / self.dx
        if self.tstype==2:
            ddrhos = - rhos[2,] * (2.0*np.pi)**2 * freq * freq / self.dxs
        else:
            ddrhos = - rhos[1,] * (2.0*np.pi)**2 * freq * freq / self.dxs
        self.A = - state.u[0] * fft.ifft(drhos,norm='ortho')
        self.S = state.xnu[0] * fft.ifft(ddrhos,norm='ortho')
        self.F = np.arange(self.nx)*self.dx
        self.F = np.where(self.F<60.0, state.forcing[0]*np.sin(np.pi*self.F/60.0), 0.0)

        state.forcing[1:tlen] = state.forcing[0:tlen-1]
        state.u[1:tlen] = state.u[0:tlen-1]
        state.xnu[1:tlen] = state.xnu[0:tlen-1]

    def euler(self, state):
        self.forward(state)
        state.rho[0,:] = state.rho[1,:] + self.dt * (self.A + self.S + self.F).real

    def leapfrog(self, state):
        self.forward(state)
        state.rho[0,:] = state.rho[2,:] + 2.0 * self.dt * (self.A + self.S + self.F).real
    
    def rk4(self, state):
        rho0 = state.rho[0,:].copy()
        self.forward(state)
        k1 = self.dt * (self.A + self.S + self.F).real
        state.rho[0,:] = rho0 + k1*0.5
        self.forward(state)
        k2 = self.dt * (self.A + self.S + self.F).real
        state.rho[0,:] = rho0 + k2*0.5
        self.forward(state)
        k3 = self.dt * (self.A + self.S + self.F).real
        state.rho[0,:] = rho0 + k3 
        self.forward(state)
        k4 = self.dt * (self.A + self.S + self.F).real
        state.rho[1,:] = rho0
        state.rho[0,:] = rho0 + (k1 + 2.0*k2 + 2.0*k3 + k4)/6.0

    def step_t(self,state,prtb):
        if self.tstype==1:
            self.euler_t(state,prtb)
        elif self.tstype==2:
            self.leapfrog_t(state,prtb)
        elif self.tstype==3:
            self.rk4_t(state,prtb)
    
    def forward_t(self,state,prtb):
        tlen = prtb.rho.shape[0]
        
        prtb.rho[1:tlen] = prtb.rho[0:tlen-1]

        ### finite-difference
        #self.At = - state.u[0] * (np.roll(prtb.rho[1,:],-1)-np.roll(prtb.rho[1,:],1)) / (2.0*self.dx)
        #if self.tstype==2:
        #    self.St = state.xnu[0] + (np.roll(prtb.rho[2,:],-1)-2*prtb.rho[2,:]+np.roll(prtb.rho[2,:],1)) / self.dxs
        #else:
        #    self.St = state.xnu[0] + (np.roll(prtb.rho[1,:],-1)-2*prtb.rho[1,:]+np.roll(prtb.rho[1,:],1)) / self.dxs
        ## spectral
        rhos = fft.fft(prtb.rho,axis=1,norm='ortho')
        freq = fft.fftfreq(prtb.rho.shape[1])
        drhos = rhos[1,] * 2.0j * np.pi * freq / self.dx
        if self.tstype==2:
            ddrhos = - rhos[2,] * (2.0*np.pi)**2 * freq * freq / self.dxs
        else:
            ddrhos = - rhos[1,] * (2.0*np.pi)**2 * freq * freq / self.dxs
        self.At = - state.u[0] * fft.ifft(drhos,norm='ortho')
        self.St = state.xnu[0] * fft.ifft(ddrhos,norm='ortho')
    
    def euler_t(self, state, prtb):
        self.forward_t(state, prtb)
        prtb.rho[0,:] = prtb.rho[1,:] + self.dt * (self.At + self.St).real

    def leapfrog_t(self, state, prtb):
        self.forward_t(state, prtb)
        prtb.rho[0,:] = prtb.rho[2,:] + 2.0 * self.dt * (self.At + self.St).real
    
    def rk4_t(self, state, prtb):
        rho0 = state.rho[0,:].copy()
        rho1 = state.rho[1,:].copy()
        drho0 = prtb.rho[0,:].copy()
        self.forward_t(state, prtb)
        self.forward(state)
        k1 = self.dt * (self.A + self.S + self.F).real
        dk1 = self.dt * (self.At + self.St).real
        state.rho[0,:] = rho0 + k1*0.5
        prtb.rho[0,:] = drho0 + dk1*0.5
        self.forward_t(state,prtb)
        self.forward(state)
        k2 = self.dt * (self.A + self.S + self.F).real
        dk2 = self.dt * (self.At + self.St).real
        state.rho[0,:] = rho0 + k2*0.5
        prtb.rho[0,:] = drho0 + dk2*0.5
        self.forward_t(state,prtb)
        self.forward(state)
        k3 = self.dt * (self.A + self.S + self.F).real
        dk3 = self.dt * (self.At + self.St).real
        state.rho[0,:] = rho0 + k3 
        prtb.rho[0,:] = drho0 + dk3 
        self.forward_t(state,prtb)
        self.forward(state)
        k4 = self.dt * (self.A + self.S + self.F).real
        dk4 = self.dt * (self.At + self.St).real
        state.rho[1,:] = rho1
        state.rho[0,:] = rho0
        prtb.rho[0,:] = drho0 + (dk1 + 2.0*dk2 + 2.0*dk3 + dk4)/6.0

    def step_adj(self,state,prtb,kc=0):
        if self.tstype==1:
            self.euler_adj(state,prtb,kc=kc)
        elif self.tstype==2:
            self.leapfrog_adj(state,prtb,kc=kc)
        elif self.tstype==3:
            self.rk4_adj(state,prtb,kc=kc)
    
    def forward_adj(self,state,prtb,kc=0):
        tlen = prtb.rho.shape[0]

        ### finite-difference
        #if self.tstype==2:
        #    self.Ad = 2.0*state.xnu[kc]/self.dxs*np.roll(prtb.rho[kc,],1)
        #    self.Bd = -4.0*state.xnu[kc]/self.dxs*prtb.rho[kc,]
        #    self.Cd = state.u[kc] / (2.0*self.dx)*np.roll(prtb.rho[kc,],-1)
        #else:
        #    self.Ad = (state.xnu[kc]/self.dxs - state.u[kc] / (2.0*self.dx))*np.roll(prtb.rho[kc,],1)
        #    self.Bd = -2.0*state.xnu[kc]/self.dxs*prtb.rho[kc,]
        #    self.Cd = (state.xnu[kc]/self.dxs + state.u[kc] / (2.0*self.dx))*np.roll(prtb.rho[kc,],-1)
        ## spectral
        rhos = fft.fft(prtb.rho,axis=1,norm='ortho')
        freq = fft.fftfreq(prtb.rho.shape[1])
        drhos = - rhos[kc,] * 2.0j * np.pi * freq / self.dx
        if self.tstype==2:
            ddrhos = - rhos[kc+1,] * (2.0*np.pi)**2 * freq * freq / self.dxs
        else:
            ddrhos = - rhos[kc,] * (2.0*np.pi)**2 * freq * freq / self.dxs
        drhos[1:] = drhos[1:] * np.sqrt(2.0)
        ddrhos[1:] = ddrhos[1:] * np.sqrt(2.0)
        self.Ad = - state.u[kc] * fft.ifft(drhos,norm='ortho')
        self.Sd = state.xnu[kc] * fft.ifft(ddrhos,norm='ortho')
    
    def euler_adj(self,state,prtb,kc=0):
        self.forward_adj(state, prtb, kc=kc)
        prtb.rho[kc+1,:] = \
            prtb.rho[kc,:] + self.dt * (self.Ad + self.Sd).real


def gc5(r,c):
    z = r/c
    return np.where(z<1.0, 1.0 - 5.0*(z**2)/3.0 + 0.625*(z**3) + 0.5*(z**4) - 0.25*(z**5), np.where(z<2.0, 4.0 - 5.0*z + 5.0*(z**2)/3.0 + 0.625*(z**3) - 0.5*(z**4) + (z**5)/12.0 - 2.0/z/3.0, 0.0))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import sys
    sys.path.append('../analysis')
    from corrfunc import Corrfunc

    nx = 300
    dx = 0.1 # km
    L = nx*dx
    dt = 0.001
    u0 = 2.0
    xnu0 = 0.0 #0.5
    F0 = 0.0
    ncyc = 1
    state1 = Advection_state(nx, tlag=ncyc, u0=u0, xnu0=xnu0, F0=F0)
    state2 = Advection_state(nx, tlag=ncyc, u0=u0, xnu0=xnu0, F0=F0)
    state3 = Advection_state(nx, tlag=ncyc, u0=u0, xnu0=xnu0, F0=F0)

    model1 = Advection(state1, dx, dt, tstype=1)
    model2 = Advection(state2, dx, dt, tstype=2)
    model3 = Advection(state3, dx, dt, tstype=3)

    # initial field
    ix = np.arange(nx)*dx 
    r = np.array([min(ix[i],L-ix[i]) for i in range(nx)])
    cfunc = Corrfunc(dx*10)
    rho0 = np.roll(cfunc(r, ftype='gc5'),nx//2)
    state1.rho[0,] = rho0
    state2.rho[0,] = rho0
    state2.rho[1,] = rho0
    state3.rho[0,] = rho0

    ntmax = 1000
    ntsave = 100
    tinterval = ntsave * dt 
    xoffset = int(u0 * tinterval / dx)

    fig, axs = plt.subplots(nrows=3,sharex=True,constrained_layout=True)
    axs[0].set_ylabel('Euler')
    axs[1].set_ylabel('Leapfrog')
    axs[2].set_ylabel('RK4')
    cmap = plt.get_cmap('tab10')
    rhoa = rho0
    for it in range(ntmax):
        model1(state1)
        #model2(state2)
        #model3(state3)
        if it%ntsave==0:
            axs[0].plot(ix,state1.rho[0,])
        #    axs[1].plot(ix,state2.rho[0,])
        #    axs[2].plot(ix,state3.rho[0,])
            for ax in axs:
                ax.plot(ix,rhoa,c='k',ls='dashed')
            rhoa = np.roll(rhoa,xoffset)
    #ax.legend([
    #    Line2D([0],[0],color=cmap(0),lw=2),
    #    Line2D([0],[0],color=cmap(1),lw=2),
    #    Line2D([0],[0],color=cmap(2),lw=2)
    #],
    #['Forward','leapfrog','RK4'])
    plt.show()
    plt.close()
    #exit()

    ## TLM, ADJ check
    state12 = Advection_state(nx, tlag=ncyc, u0=u0, xnu0=xnu0, F0=F0)
    state22 = Advection_state(nx, tlag=ncyc, u0=u0, xnu0=xnu0, F0=F0)
    state32 = Advection_state(nx, tlag=ncyc, u0=u0, xnu0=xnu0, F0=F0)
    prtb1 = Advection_state(nx, tlag=ncyc, u0=u0, xnu0=xnu0, F0=F0)
    prtb2 = Advection_state(nx, tlag=ncyc, u0=u0, xnu0=xnu0, F0=F0)
    prtb3 = Advection_state(nx, tlag=ncyc, u0=u0, xnu0=xnu0, F0=F0)

    alp = 1.0e-5
    drho0 = np.random.randn(nx)*0.1
    prtb1.rho[0,] = drho0
    prtb2.rho[0,] = drho0
    prtb2.rho[1,] = drho0
    prtb3.rho[0,] = drho0
    schemes = ["Euler"] #,"Leapfrog","RK4"]
    for scheme, model, st0, st1, prtb in zip(schemes,[model1,model2,model3],[state1,state2,state3],[state12,state22,state32],[prtb1,prtb2,prtb3]):
        print(scheme)
        # forward (TLM)
        st1.rho[0,] = st0.rho[0,] + alp*drho0
        st1.rho[1,] = st0.rho[1,] + alp*drho0
        for icyc in range(ncyc):
            model(st0)
            model(st1)
            model.step_t(st0,prtb)
        diff1 = np.dot((st1.rho[0,]-st0.rho[0,]),(st1.rho[0,]-st0.rho[0,]))
        diff2 = np.dot(prtb.rho[0,],prtb.rho[0,])
        print(f"|M(x+a*dx)-M(x)|={np.sqrt(diff1):.4e}")
        print(f"      a*|TLM*dx|={alp*np.sqrt(diff2):.4e}")
        # backward (ADJ)
        p0 = prtb.rho[ncyc]
        dax = prtb.rho[0].copy()
        #prtb.rho[1:,] = 0.0
        for icyc in range(ncyc):
            model.step_adj(st0,prtb,kc=icyc)
        diff1 = np.dot(dax,dax)
        diff2 = np.dot(p0,prtb.rho[ncyc])
        print(f"|(Ax)^T Ax|={np.sqrt(diff1)}")
        print(f"|x^T(A^TAx)|={np.sqrt(diff2)}")
        print(f"|(Ax)^T Ax|-|x^T(A^TAx)|={np.sqrt(diff1)-np.sqrt(diff2)}")
    