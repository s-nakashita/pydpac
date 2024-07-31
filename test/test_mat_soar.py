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

def c0(r,c,l):
    if c==0.0:
        return (1.0 + r/l)*np.exp(-r/l)
    else:
        return (np.cos(c*r) + np.sin(c*r)/(c*l))*np.exp(-r/l)

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
    functype = "soar"
    figdir2 = f'{figdir}/{functype}'
    if not os.path.isdir(figdir2):
        os.mkdir(figdir2)
    #cmap = plt.get_cmap('tab20')
    cmap = plt.get_cmap('viridis')

    clist = (np.linspace(0,5,6)).tolist()
    intcol = cmap.N // (len(clist) - 1) // 2
    distmax = 2.0*np.pi
    #distmax = nx_true / np.pi
    lb = 0.05
    ##true GM (cyclic)
    dist = np.eye(nx_true)
    fig, ax = plt.subplots(figsize=[8,6],constrained_layout=True)
    for i in range(nx_true):
        dist[i,:] = step_true.calc_dist(i)
        #dist[i,:] = 2.0 * np.pi * dist[i,:] / nx_true
    for i,c in enumerate(clist):
        l = lb * distmax
        corr = np.zeros(dist.shape[0])
        j0 = nx_true//2
        ctmp = c0(np.roll(dist[:,j0],-j0)[:(nx_true//2)+1],c,l)
        ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:-1])])
        corr = np.roll(ctmp2,j0)
        ax.plot(corr,c=cmap(2*i*intcol),label=f'c={c:.2f}')
    ax.set_title(f"lb={lb:.3f}")
    ax.legend(ncol=1,bbox_to_anchor=(1.0,0.9))
    fig.savefig(f"{figdir2}/funccomp_lb{lb:.3f}.png",dpi=300)
    plt.show()
    plt.close()
    #exit()
    
    condlist = []
    eiglist = []
    for c in clist:
        l = lb * distmax
        cmat = c0(dist,c,l)
        condlist.append(la.cond(cmat))
        fig, ax = plt.subplots(figsize=[4,4],constrained_layout=True)
        p = ax.matshow(cmat,vmin=-1.0,vmax=1.0,cmap='bwr')
        fig.colorbar(p,ax=ax,shrink=0.6,pad=0.01)
        ax.set_title(f"c={c:.2f}")
        #ax.set_title(f"lb={lb:.1f}"+r"$\times N/\pi$")
        fig.savefig(f"{figdir2}/mat_true_c{c:.2f}.png",dpi=300)
        plt.close()
        eival, eivec = la.eigh(cmat)
        eiglist.append(eival[::-1])
    fig, ax = plt.subplots(figsize=[10,8])
    ax.plot(clist,condlist)
    ax.set_yscale("log")
    ax.set_title(f"condition number, functype={functype}")
    ax.set_xlabel(r"$c$") # for $\exp (-\frac{1}{2}(\frac{d}{2\pi\times L_b})^2)$")
    #ax.set_xlabel(r"$L_b$ for $\exp (-\frac{1}{2}(\frac{d}{L_b\times N/\pi})^2)$")
    fig.savefig(f"{figdir}/matcond{functype}_true.png",dpi=300)
    #plt.show()
    plt.close()
    fig, ax = plt.subplots(figsize=[10,8])
    for i,c in enumerate(clist):
        ev = eiglist[i]
        ax.plot(np.arange(1,ev.size+1),ev,lw=0.0,c=cmap(2*i*intcol),
        marker='x',label=f'c={c:.2f}',alpha=0.5)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(clist[0],clist[-1]), cmap=cmap),ax=ax,label="c")
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
    for c in clist:
        l = lb * distmax
        cmat = c0(dist,c,l)
        condlist.append(la.cond(cmat))
        fig, ax = plt.subplots(figsize=[4,4],constrained_layout=True)
        p = ax.matshow(cmat,vmin=-1.0,vmax=1.0,cmap='bwr')
        fig.colorbar(p,ax=ax,shrink=0.6,pad=0.01)
        ax.set_title(f"c={c:.2f}")
        fig.savefig(f"{figdir2}/mat_gm_c{c:.2f}.png",dpi=300)
        plt.close()
        eival, eivec = la.eigh(cmat)
        eiglist.append(eival[::-1])
    fig, ax = plt.subplots(figsize=[10,8])
    ax.plot(clist,condlist)
    ax.set_yscale("log")
    ax.set_title(f"condition number, functype={functype}")
    ax.set_xlabel(r"$c$")
    fig.savefig(f"{figdir}/matcond{functype}_gm.png",dpi=300)
    #plt.show()
    plt.close()
    fig, ax = plt.subplots(figsize=[10,8])
    for i,c in enumerate(clist):
        ev = eiglist[i]
        ax.plot(np.arange(1,ev.size+1),ev,lw=0.0,c=cmap(2*i*intcol),marker='x',label=f'c={c:.2f}',alpha=0.5)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(clist[0],clist[-1]), cmap=cmap),ax=ax,label="c")
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
    for c in clist:
        l = lb * distmax
        cmat = c0(dist,c,l)
        #cmat = 0.5*(cmat+cmat.transpose())
        condlist.append(la.cond(cmat))
        fig, ax = plt.subplots(figsize=[4,4],constrained_layout=True)
        p = ax.matshow(cmat,vmin=-1.0,vmax=1.0,cmap='bwr')
        fig.colorbar(p,ax=ax,shrink=0.6,pad=0.01)
        ax.set_title(f"c={c:.2f}")
        fig.savefig(f"{figdir2}/mat_lam_c{c:.2f}.png",dpi=300)
        plt.close()
        eival, eivec = la.eigh(cmat)
        eiglist.append(eival[::-1])
    fig, ax = plt.subplots(figsize=[10,8])
    ax.plot(clist,condlist)
    ax.set_yscale("log")
    ax.set_title(f"condition number, functype={functype}")
    ax.set_xlabel(r"$c$")
    fig.savefig(f"{figdir}/matcond{functype}_lam.png",dpi=300)
    #plt.show()
    plt.close()
    fig, ax = plt.subplots(figsize=[10,8])
    for i,c in enumerate(clist):
        ev = eiglist[i]
        ax.plot(np.arange(1,ev.size+1),ev,lw=0.0,c=cmap(2*i*intcol),marker='x',label=f'c={c:.2f}',alpha=0.5)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(clist[0],clist[-1]), cmap=cmap),ax=ax,label="c")
    ax.set_yscale("log")
    ax.set_xlabel("mode")
    ax.set_title(f"eigenvalues, functype={functype}")
    fig.savefig(f"{figdir}/mateig{functype}_lam.png",dpi=300)
    #plt.show()
