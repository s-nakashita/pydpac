import numpy as np 

# Linear inflation functions (Duc et al. 2020a, QJRMS)
#   f(lambda) = a*lambda + b
# gamma = singular values of normalized ensemble perturbations in observation space
# lambda = eigenvalues of transform matrix
#   lambda = 1.0 / (1 + gamma^2)^{1/2}
# stdb = background ensemble spread
# stda = analysis ensemble spread (before inflated)
class inflfunc():
    def __init__(self,infltype,paramtype=0):
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
        # 0 : averaged lambda
        # 1 : stda / stdb
        # 2 : stdb / stda (~RTPS)
        # 3 : inverse of averaged lambda

    def __call__(self,lam,stdb,stda,\
        alpha1=None,alpha2=None,a=None,b=None):
        if self.paramtype == 0:
            param = np.mean(lam)
        elif self.paramtype == 1:
            param = stda / stdb
        elif self.paramtype == 2:
            param = stdb / stda
        elif self.paramtype == 3:
            param = 1.0 / np.mean(lam)
        
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
            if self.paramtype < 2:
                laminf = (1.0 + alpha1*(1.0 - param))*lam
            else:
                laminf = (1.0 + alpha1*(param - 1.0))*lam
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
    
    def prf(self,gamma,lam):
        return gamma*lam
    
    def pdr(self,do,gamma,lam,laminf):
        # do: innovation projected onto the left-singular vector space
        ga2 = gamma*gamma
        lam4 = lam*lam*lam*lam
        do2 = do*do
        num = np.sum(ga2*lam4*do2)
        den = np.sum(ga2*laminf*laminf)
        return num/den

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
            paramtype=0
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
            paramtype=0
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