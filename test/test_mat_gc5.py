import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import os
from pathlib import Path
import shutil
import sys
from test_ncm import ncm

def b0(r,a,c):
    return np.where(abs(r)<0.5*c,2*(a-1)*abs(r)/c+1,np.where(abs(r)<c,2*a*(1-abs(r)/c),0))

def c0(r,a,c):
    r2 = np.zeros(2*r.size-1)
    r2[:r.size] = r[::-1]*-1.0
    r2[r.size-1:] = r[:]
    b2 = b0(r2,a,c)
    b1 = np.hstack((b2,b2))
    #plt.plot(r2,b2);plt.show();plt.close()
    c2 = np.convolve(b1,b2,mode='full')
    #plt.plot(c2);plt.show();plt.close()
    c = c2[r2.size-1:r2.size+r.size-1]
    return c/c[0]

if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
    sys.path.append('../')
    from l05nest import nx_true, nx_gm, nk_gm, nx_lam, nsp, po, nk_lam, ni, b, c, dt_gm, F, intgm, ist_lam, nsp
    sys.path.append('../model')
    from lorenz3 import L05III
    from lorenz_nest import L05nest
    step_true = L05III(nx_true,nk_lam,ni,b,c,dt_gm,F)
    step = L05nest(nx_true, nx_gm, nx_lam, nk_gm, nk_lam, ni, b, c, dt_gm, F, intgm, ist_lam, nsp)

    figdir = '.'
    functype = "gc5_a"
    figdir2 = f'{figdir}/{functype}'
    if not os.path.isdir(figdir2):
        os.mkdir(figdir2)
    #cmap = plt.get_cmap('tab20')
    cmap = plt.get_cmap('viridis')

    alist = (np.linspace(-1,1,17)).tolist()
    intcol = cmap.N // (len(alist) - 1) // 2
    distmax = 2.0*np.pi
    #distmax = nx_true / np.pi
    lb = 0.1
    ##true GM (cyclic)
    dist = np.eye(nx_true)
    fig, ax = plt.subplots(figsize=[8,6],constrained_layout=True)
    for i in range(nx_true):
        dist[i,:] = step_true.calc_dist(i)
        #dist[i,:] = 2.0 * np.pi * dist[i,:] / nx_true
    """
    for i,a in enumerate(alist):
        c = lb * distmax * np.sqrt(10.0/3.0)
        corr = np.zeros(dist.shape[0])
        j0 = nx_true//2
        ctmp = c0(np.roll(dist[:,j0],-j0)[:(nx_true//2)+1],a,c)
        ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:-1])])
        corr = np.roll(ctmp2,j0)
        ax.plot(corr,c=cmap(2*i*intcol),label=f'a={a:.2f}')
    ax.set_title(f"lb={lb:.3f}")
    ax.legend(ncol=1,bbox_to_anchor=(1.0,0.9))
    fig.savefig(f"{figdir2}/funccomp_lb{lb:.3f}.png",dpi=300)
    #plt.show()
    plt.close()
    exit()
    """
    condlist = []
    eiglist = []
    for a in alist:
        c = lb * distmax * np.sqrt(10.0/3.0)
        cmat = np.eye(dist.shape[0])
        for j in range(cmat.shape[1]):
            ctmp = c0(np.roll(dist[:,j],-j)[:(dist.shape[0]//2)+1],a,c)
            ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:-1])])
            cmat[:,j] = np.roll(ctmp2,j)
        condlist.append(la.cond(cmat))
        fig, ax = plt.subplots(figsize=[4,4],constrained_layout=True)
        p = ax.matshow(cmat,vmin=-1.0,vmax=1.0,cmap='bwr')
        fig.colorbar(p,ax=ax,shrink=0.6,pad=0.01)
        ax.set_title(f"a={a:.2f}")
        #ax.set_title(f"lb={lb:.1f}"+r"$\times N/\pi$")
        fig.savefig(f"{figdir2}/mat_true_a{a:.2f}.png",dpi=300)
        plt.close()
        eival, eivec = la.eigh(cmat)
        eiglist.append(eival[::-1])
        if a==0.5:
            z = dist / c
            cmat = np.where(z<1.0, 1.0 - 5.0*(z**2)/3.0 + 0.625*(z**3) + 0.5*(z**4) - 0.25*(z**5), np.where(z<2.0, 4.0 - 5.0*z + 5.0*(z**2)/3.0 + 0.625*(z**3) - 0.5*(z**4) + (z**5)/12.0 - 2.0/z/3.0, 0.0))
            ev1, _ = la.eigh(cmat)
            cond1 = la.cond(cmat)
    fig, ax = plt.subplots(figsize=[10,8])
    ax.plot(alist,condlist)
    ax.plot([0.5],[cond1],lw=0.0,marker='^')
    ax.set_yscale("log")
    ax.set_title(f"condition number, functype={functype}")
    ax.set_xlabel(r"$a$") # for $\exp (-\frac{1}{2}(\frac{d}{2\pi\times L_b})^2)$")
    #ax.set_xlabel(r"$L_b$ for $\exp (-\frac{1}{2}(\frac{d}{L_b\times N/\pi})^2)$")
    fig.savefig(f"{figdir}/matcond{functype}_true.png",dpi=300)
    #plt.show()
    plt.close()
    fig, ax = plt.subplots(figsize=[10,8])
    for i,a in enumerate(alist):
        ev = eiglist[i]
        ax.plot(np.arange(1,ev.size+1),ev,lw=0.0,c=cmap(2*i*intcol),
        marker='x',label=f'a={a:.2f}',alpha=0.5)
        if a==0.5:
            ax.plot(np.arange(1,ev1.size+1),ev1[::-1],lw=1.0,c='k')
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(alist[0],alist[-1]), cmap=cmap),ax=ax,label="a")
    ax.set_yscale("log")
    ax.set_xlabel("mode")
    ax.set_title(f"eigenvalues, functype={functype}")
    fig.savefig(f"{figdir}/mateig{functype}_true.png",dpi=300)
    #plt.show()
    plt.close()
    #exit()
    ##GM (cyclic)
    dist = np.eye(nx_gm)
    for i in range(nx_gm):
        dist[i,:] = step.calc_dist_gm(i)
        #dist[i,:] = 2.0 * np.pi * dist[i,:] / nx_true
    condlist = []
    eiglist = []
    for a in alist:
        c = lb * distmax * np.sqrt(10.0/3.0)
        cmat = np.eye(dist.shape[0])
        for j in range(cmat.shape[1]):
            ctmp = c0(np.roll(dist[:,j],-j)[:(dist.shape[0]//2)+1],a,c)
            ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:-1])])
            cmat[:,j] = np.roll(ctmp2,j)
        condlist.append(la.cond(cmat))
        fig, ax = plt.subplots(figsize=[4,4],constrained_layout=True)
        p = ax.matshow(cmat,vmin=-1.0,vmax=1.0,cmap='bwr')
        fig.colorbar(p,ax=ax,shrink=0.6,pad=0.01)
        ax.set_title(f"a={a:.2f}")
        fig.savefig(f"{figdir2}/mat_gm_a{a:.2f}.png",dpi=300)
        plt.close()
        eival, eivec = la.eigh(cmat)
        eiglist.append(eival[::-1])
        if a==0.5:
            z = dist / c
            cmat = np.where(z<1.0, 1.0 - 5.0*(z**2)/3.0 + 0.625*(z**3) + 0.5*(z**4) - 0.25*(z**5), np.where(z<2.0, 4.0 - 5.0*z + 5.0*(z**2)/3.0 + 0.625*(z**3) - 0.5*(z**4) + (z**5)/12.0 - 2.0/z/3.0, 0.0))
            ev1, _ = la.eigh(cmat)
            cond1 = la.cond(cmat)
    fig, ax = plt.subplots(figsize=[10,8])
    ax.plot(alist,condlist)
    ax.plot([0.5],[cond1],lw=0.0,marker='^')
    ax.set_yscale("log")
    ax.set_title(f"condition number, functype={functype}")
    ax.set_xlabel(r"$a$")
    fig.savefig(f"{figdir}/matcond{functype}_gm.png",dpi=300)
    #plt.show()
    plt.close()
    fig, ax = plt.subplots(figsize=[10,8])
    for i,a in enumerate(alist):
        ev = eiglist[i]
        ax.plot(np.arange(1,ev.size+1),ev,lw=0.0,c=cmap(2*i*intcol),marker='x',label=f'a={a:.2f}',alpha=0.5)
        if a==0.5:
            ax.plot(np.arange(1,ev1.size+1),ev1[::-1],lw=1.0,c='k')
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(alist[0],alist[-1]), cmap=cmap),ax=ax,label="a")
    ax.set_yscale("log")
    ax.set_xlabel("mode")
    ax.set_title(f"eigenvalues, functype={functype}")
    fig.savefig(f"{figdir}/mateig{functype}_gm.png",dpi=300)
    #plt.show()
    plt.close()
    ##LAM (noncyclic)
    dist = np.eye(nx_lam)
    for i in range(nx_lam):
        dist[i,:] = step.calc_dist_lam(i)
        #dist[i,:] = 2.0 * np.pi * dist[i,:] / nx_true
    condlist = []
    eiglist = []
    for a in alist:
        c = lb * distmax * np.sqrt(10.0/3.0)
        cmat = np.eye(dist.shape[0])
        for i in range(cmat.shape[0]):
            if i < dist.shape[0]//2:
                ctmp = c0(np.roll(dist[i,],-i)[:dist.shape[0]-i],a,c)
                ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:-1])])
                cmat[i,] = np.roll(ctmp2,i)[:dist.shape[0]]
            else:
                ctmp = c0(np.flip(dist[i,:i+1]),a,c)
                ctmp2 = np.hstack([np.flip(ctmp),ctmp[1:-1]])
                cmat[i,] = ctmp2[:dist.shape[0]]
        cmat = 0.5*(cmat+cmat.transpose())
        condlist.append(la.cond(cmat))
        fig, ax = plt.subplots(figsize=[4,4],constrained_layout=True)
        p = ax.matshow(cmat,vmin=-1.0,vmax=1.0,cmap='bwr')
        fig.colorbar(p,ax=ax,shrink=0.6,pad=0.01)
        ax.set_title(f"a={a:.2f}")
        fig.savefig(f"{figdir2}/mat_lam_a{a:.2f}.png",dpi=300)
        plt.close()
        eival, eivec = la.eigh(cmat)
        eiglist.append(eival[::-1])
        if a==0.5:
            z = dist / c
            cmat = np.where(z<1.0, 1.0 - 5.0*(z**2)/3.0 + 0.625*(z**3) + 0.5*(z**4) - 0.25*(z**5), np.where(z<2.0, 4.0 - 5.0*z + 5.0*(z**2)/3.0 + 0.625*(z**3) - 0.5*(z**4) + (z**5)/12.0 - 2.0/z/3.0, 0.0))
            ev1, _ = la.eigh(cmat)
            cond1 = la.cond(cmat)
    fig, ax = plt.subplots(figsize=[10,8])
    ax.plot(alist,condlist)
    ax.plot([0.5],[cond1],lw=0.0,marker='^')
    ax.set_yscale("log")
    ax.set_title(f"condition number, functype={functype}")
    ax.set_xlabel(r"$a$")
    fig.savefig(f"{figdir}/matcond{functype}_lam.png",dpi=300)
    #plt.show()
    plt.close()
    fig, ax = plt.subplots(figsize=[10,8])
    for i,a in enumerate(alist):
        ev = eiglist[i]
        ax.plot(np.arange(1,ev.size+1),ev,lw=0.0,c=cmap(2*i*intcol),marker='x',label=f'a={a:.2f}',alpha=0.5)
        if a==0.5:
            ax.plot(np.arange(1,ev1.size+1),ev1[::-1],lw=1.0,c='k')
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(alist[0],alist[-1]), cmap=cmap),ax=ax,label="a")
    ax.set_yscale("log")
    ax.set_xlabel("mode")
    ax.set_title(f"eigenvalues, functype={functype}")
    fig.savefig(f"{figdir}/mateig{functype}_lam.png",dpi=300)
    #plt.show()
