import numpy as np 
import logging

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')

# Linear inflation functions (Duc et al. 2020a, QJRMS)
#   f(lambda) = a*lambda + b
# gamma = singular values of normalized ensemble perturbations in observation space
# lambda = eigenvalues of transform matrix
#   lambda = 1.0 / (1 + gamma^2)^{1/2}
# stdb = background ensemble spread
# stda = analysis ensemble spread (before inflated)
class inflfunc():
    def __init__(self,infltype,paramtype=0,nit=3):
        self.infltype = infltype
        # inflation type
        # "const" : constant 
        #   f(lambda) = 1 - alpha*(1 - param) = const.
        #   alpha must be equal or smaller than 0
        # "mult" : multiplicative
        #   f(lambda) = (1 + alpha*(param-1))*lambda
        #   alpha must be equal or larger than 0
        # "fixed" : fixed (RTPP)
        #   f(lambda) = alpha*lambda + (1 - alpha)
        #   alpha must be equal or smaller than 1
        # "vary" : parameter-varying
        #   f(lambda) = (1 + alpha_1*(1-param))*lambda + alpha_2*(1-param)
        #   both alpha_2 and (alpha_1+alpha_2) must be equal or larger than 0
        self.paramtype = paramtype
        # parameterization type
        # 0 : inverse of averaged lambda
        # 1 : averaged lambda
        # 2 : stdb / stda (~RTPS)
        # 3 : stda / stdb
        self.nit = nit # iteration number for adaptive estimation
        self.rhosave = []
        self.pdrsave = []

    def __call__(self,lam,stdb=None,stda=None,\
        alpha1=None,alpha2=None,a=None,b=None):
        if self.paramtype == 0:
            param = 1.0 / np.mean(lam)
        elif self.paramtype == 1:
            param = np.mean(lam)
        elif self.paramtype == 2:
            param = stdb / stda
        elif self.paramtype == 3:
            param = stda / stdb
        logging.info(f"param={param}")

        if self.infltype == 'const':
            if alpha1 is None:
                alpha1 = (1.0 - b)/(1.0 - param)
            #if alpha > 1.0:
            #    raise ValueError(f'alpha={alpha} must be equal or smaller than 1')
            laminf = 1.0 - alpha1 * (1.0 - param)
        elif self.infltype == 'mult':
            if alpha1 is None:
                alpha1 = (a - 1.0)/(param - 1.0)
            if alpha1 < 0.0:
                raise ValueError(f'alpha={alpha1} must be positive or 0')
            if self.paramtype % 2 == 0:
                laminf = (1.0 + alpha1*(param - 1.0))*lam
            else:
                laminf = (1.0 + alpha1*(1.0 - param))*lam
        elif self.infltype == 'fixed':
            if alpha1 is None:
                alpha1 = a
            if alpha1 > 1.0:
                raise ValueError(f'alpha={alpha1} must be equal or smaller than 1')
            laminf = alpha*lam + (1.0 - alpha1)
        elif self.infltype == 'vary':
            if alpha1 is None:
                alpha1 = (a - 1.0)/(1.0 - param)
            if alpha2 is None:
                alpha2 = b / (1.0 - param)
            #if (alpha1+alpha2) < 0.0 or alpha2 < 0.0:
            #    raise ValueError(f'both alpha_2={alpha2} and (alpha_1+alpha_2)={alpha1+alpha2} must be equal or larger than 0')
            laminf = (1.0 + alpha1*(1.0 - param))*lam + alpha2*(1.0 - param)
        
        return laminf

    def lam2gam(self,lam):
        return np.sqrt(1.0/lam/lam - 1.0)
    
    def gam2lam(self,gamma):
        return 1.0/np.sqrt(1.0 + gamma*gamma)

    def g2f(self,gamma,g):
        ga2 = gamma*gamma
        return g / np.sqrt(ga2 + ga2*g*g)
    
    def prf(self,gamma,lam):
        return gamma*lam
    
    def pdr(self,do,gamma,lam,laminf):
        # do: innovation projected onto the left-singular vector space
        ga2 = gamma*gamma
        lam4 = lam*lam*lam*lam
        do2 = do*do
        num = np.sum(ga2*lam4*do2)
        den = np.sum(ga2*laminf*laminf)
        self.pdrsave.append(num/den)

    def est(self,do,gamma,form='mult',eps=1.0e-6):
        # do: innovation projected onto the left-singular vector space
        do2 = do*do
        nmode = do.size
        rho0 = 1.0
        it=0
        while (it<self.nit):
            ga1 = gamma*rho0
            la1 = self.gam2lam(ga1)
            ga2 = gamma*gamma
            la2 = la1*la1
            la4 = la2*la2
            rho2 = rho0*rho0/nmode*np.sum(la2 + ga2*la4*do2)
            rho2 = max(1.0, rho2)
            rho = np.sqrt(rho2)
            diff = np.sqrt((rho-rho0)*(rho-rho0))
            if diff < eps: break
            rho0 = rho
            it+=1
            logger.info(f'iter{it}: rho={rho0} diff={diff}')
        self.rhosave.append(rho)
        return rho * gamma

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    lam = np.linspace(0.0,1.0,50,endpoint=False)

    infltype = 'vary'
    paramtype = 0
    ifunc = inflfunc(infltype, paramtype)
    gamma = ifunc.lam2gam(lam)

    fig = plt.figure(figsize=[10,6],constrained_layout=True)
    gs = GridSpec(1,5,figure=fig)
    ax1 = fig.add_subplot(gs[:2])
    ax2 = fig.add_subplot(gs[2:])

    f_noassim = np.ones(lam.size)
    ax1.plot(lam,f_noassim,c='r',marker='o',label='No assimilation')
    r_noassim = ifunc.prf(gamma,f_noassim)
    ax2.plot(gamma,r_noassim,c='r',marker='o',label='No assimilation')
    f_noinfl = lam
    ax1.plot(lam,f_noinfl,c='b',marker='s',label='No inflation')
    r_noinfl = ifunc.prf(gamma,f_noinfl)
    ax2.plot(gamma,r_noinfl,c='b',marker='s',label='No inflation')
    #alist = [1.4, 1.9, 2.4]
    #alist = [0.7, 0.3,-0.1]
    alist = [0.2, 0.6, 1.0, 1.4]
    #blist = [0.5, 0.8, 1.1]
    blist = [0.7, 0.5, 0.3, 0.1]
    colors = ['orange','green','magenta','cyan']
    markers = ['^','*','v','D']
    infltypes = ['const','mult','fixed','vary']
    i = 0
    #for a, b in zip(alist,blist):
    for infltype in infltypes:
        if infltype == 'const':
            paramtype=1
            a=0.0
            b=0.7
            label=f'Const inflation b={b:.1f}'
        elif infltype == 'mult':
            paramtype=2
            a=1.3
            b=0.0
            label=f'Mltpl inflation a={a:.1f}'
        elif infltype == 'fixed':
            paramtype=2
            a=0.3
            b = 1.0 - a
            label=f'RTPP a,b={a:.1f},{b:.1f}'
        elif infltype == 'vary':
            paramtype=1
            a=0.6
            b=0.7
            label=f'PVLinear a,b={a:.1f},{b:.1f}'
        ifunc = inflfunc(infltype, paramtype)
        f = np.array([ifunc(lam1,1.5,1.0,a=a,b=b) for lam1 in lam])
        ax1.plot(lam,f,c=colors[i],marker=markers[i],label=label)
        r = ifunc.prf(gamma,f)
        ax2.plot(gamma,r,c=colors[i],marker=markers[i],label=label)
        i += 1

    ax1.set_xlabel(r'$\lambda$')
    ax1.set_ylabel(r'$f(\lambda)$')
    ax1.grid()
    ax1.set_xlim(-0.01,1.01)
    ax1.set_ylim(0.0,1.4)
    ax1.legend(loc='upper left')
    ax2.set_xlabel(r'$\gamma$')
    ax2.set_ylabel(r'$r(\gamma)$')
    ax2.grid()
    ax2.set_xlim(0.0,3.0)
    ax2.set_ylim(0.0,3.0)
    ax2.legend(loc='upper left')

    #fig.savefig("D20fig3.png")
    #fig.savefig("D20fig4.png")
    #fig.savefig("D20fig7.png")
   # fig.savefig("D20fig8.png")
    fig.savefig("D20fig10.png")
    plt.show()