import numpy as np 
import numpy.linalg as la
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
import os
from pathlib import Path
import shutil
import sys
from test_ncm import ncm
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append('../')
from l05nest import nx_true, nx_gm, nk_gm, nx_lam, nsp, po, nk_lam, ni, b, c, F
sys.path.append(os.path.join(os.path.dirname(__file__),'../analysis'))
from trunc1d import Trunc1d
nobs=15
nmem=240
figdir1 = Path(f'{parent_dir}/data/l05III/nmc_obs{nobs}')
figdir2 = Path(f'{parent_dir}/data/l05nest/nmc_obs{nobs}')
if not figdir1.exists():
    figdir1.mkdir(parents=True)
if not figdir2.exists():
    figdir2.mkdir(parents=True)
functype = "gc5"
if functype == "gc5":
    #from test_mat_gc5 import c0
    sys.path.append('../analysis')
    from corrfunc import Corrfunc
    from minimize import Minimize
    def calc_j(p,*args):
        r, cdata = args
        c0 = Corrfunc(p[1],a=p[0])
        #cgues = c0(r,p[0],p[1])
        cgues = c0(r,ftype="gc5")
        return 0.5 * np.mean((cgues - cdata)**2)
# global Lorenz III
##bmatdir = f"model/lorenz/n{nx_true}k{nk_lam}i{ni}F{int(F)}b{b:.1f}c{c:.1f}"
##f = os.path.join(parent_dir,bmatdir,"B.npy")
#bmatdir = f"work/l05III/var_obs{nobs}"
bmatdir = f"work/l05III/envar_mem{nmem}obs{nobs}"
f = os.path.join(parent_dir,bmatdir,"ix.txt")
ix_true = np.loadtxt(f)
f = os.path.join(parent_dir,bmatdir,"l05III_B48m24.npy")
shutil.copy2(Path(f),figdir1)
bmat = np.load(f)
# GM & LAM
##bmatdir_nest = f"model/lorenz/ng{nx_gm}nl{nx_lam}kg{nk_gm}kl{nk_lam}nsp{nsp}p{po}F{int(F)}b{b:.1f}c{c:.1f}"
#bmatdir_nest = f"work/l05nest/var_obs{nobs}"
bmatdir_nest = f"work/l05nest_K15/envar_m{nmem}obs{nobs}"
f = os.path.join(parent_dir,bmatdir_nest,"ix_gm.txt")
ix_gm = np.loadtxt(f)
f = os.path.join(parent_dir,bmatdir_nest,"ix_lam.txt")
ix_lam = np.loadtxt(f)
#f = os.path.join(parent_dir,bmatdir_nest,"B_gmfull.npy")
f = os.path.join(parent_dir,bmatdir_nest,"l05nest_B48m24_gm.npy")
shutil.copy2(Path(f),figdir2)
bmat_gm = np.load(f)
#f = os.path.join(parent_dir,bmatdir_nest,"B_lam.npy")
f = os.path.join(parent_dir,bmatdir_nest,"l05nest_B48m24_lam.npy")
shutil.copy2(Path(f),figdir2)
bmat_lam = np.load(f)
#f = os.path.join(parent_dir,bmatdir_nest,"B_gm.npy")
f = os.path.join(parent_dir,bmatdir_nest,"l05nest_V48m24_trunc.npy")
shutil.copy2(Path(f),figdir2)
bmat_gm2lam = np.load(f)
#f = os.path.join(parent_dir,bmatdir_nest,"E_lg.npy")
f = os.path.join(parent_dir,bmatdir_nest,"l05nest_B48m24_gm2lam_trunc.npy")
shutil.copy2(Path(f),figdir2)
ebkmat = np.load(f)
#f = os.path.join(parent_dir,bmatdir_nest,"E_gl.npy")
#ekbmat = np.load(f)
ekbmat = ebkmat.T

i0 = np.argmin(np.abs(ix_gm-ix_lam[0]))
if ix_gm[i0]<ix_lam[0]: i0+=1
i1 = np.argmin(np.abs(ix_gm-ix_lam[-1]))
if ix_gm[i1]>ix_lam[-1]: i1-=1
#ix_gm2lam = ix_gm[i0:i1+1]
#ix_gm2lam = ix_lam
ntrunc = 12
trunc_operator = Trunc1d(ix_lam,ntrunc=ntrunc,cyclic=False,nghost=0)
ix_gm2lam = trunc_operator.ix_trunc
#figdir = '.'
#figdir = os.path.join(parent_dir,bmatdir_nest)

matrices = {"bmat":bmat,"bmat_gm":bmat_gm,"bmat_lam":bmat_lam,"bmat_gm2lam":bmat_gm2lam}
    #,"ebkmat":ebkmat,"ekbmat":ekbmat}
titles = {"bmat":"$\mathbf{B}_\mathrm{hres}$","bmat_gm":"$\mathbf{B}_\mathrm{GM}$","bmat_lam":"$\mathbf{B}_\mathrm{LAM}$","bmat_gm2lam":"$\mathbf{V}$","ebkmat":"$\mathbf{E}_\mathrm{bk}$","ekbmat":"$\mathbf{E}_\mathrm{kb}$"}
figdirs = {"bmat":figdir1,"bmat_gm":figdir2,"bmat_lam":figdir2,"bmat_gm2lam":figdir2}
cyclics = {"bmat":True,"bmat_gm":True,"bmat_lam":False,"bmat_gm2lam":False}
xaxis = {"bmat":ix_true,"bmat_gm":ix_gm,"bmat_lam":ix_lam,"bmat_gm2lam":ix_gm2lam}
for label in matrices.keys():
    #if label == "bmat": continue
    mat = matrices[label]
    title = titles[label]
    figdir = figdirs[label]
    cyclic = cyclics[label]
    eival, eivec = la.eigh(mat)
    eival = eival[::-1]
    eivec = eivec[:,::-1]
    print(eival.shape)
    print(eivec.shape)
    npos = np.sum(eival>0.0)
    accum = [eival[:i].sum()/eival.sum() for i in range(1,eival.size+1)]
    npos=0
    while True:
        if accum[npos] > 0.99: break
        npos += 1
    print(npos)
    tmp = np.dot(eivec[:,:npos], np.diag(eival[:npos]))
    print(tmp.shape)
    mat2 = np.dot(eivec[:,:npos], tmp.T)
    print(mat2.shape)

    #fig, ax = plt.subplots()
    #mp = ax.matshow(mat-mat.T)
    #fig.colorbar(mp,ax=ax,pad=0.01,shrink=0.6)
    #ax.set_title(r"$\|$"+title+r"$-($"+title+r"$)^\mathrm{T}\|_F=$"+f"{la.norm(mat-mat.T,ord='fro'):.3e}")
    #plt.show()

    fig, axs = plt.subplots(ncols=2,nrows=2,figsize=[10,10],constrained_layout=True)
    mp=axs[0,0].matshow(mat)
    fig.colorbar(mp,ax=axs[0,0],pad=0.01,shrink=0.6)
    axs[0,0].set_title(title)
    axs[0,1].plot(np.arange(1,eival.size+1),eival)
    axs[0,1].vlines([npos],0,1,colors='k',ls='dashdot',transform=axs[0,1].get_xaxis_transform())
    axs[0,1].set_yscale("log")
    axs[0,1].set_ylabel("Eigenvalues")
    axs[0,1].set_xlim(1,eival.size)
    ax2 = axs[0,1].twinx()
    accum = [np.sum(eival[:i]/eival[:npos].sum()) if eival[i-1]>0.0 else 1.0 for i in range(1,eival.size+1)]
    ax2.plot(np.arange(1,eival.size+1),accum,c='red',lw=3.0)
    ax2.tick_params(axis='y',labelcolor='red')
    ax2.set_ylabel("contribution",color='red')
    mp2=axs[1,0].matshow(mat2)
    fig.colorbar(mp2,ax=axs[1,0],pad=0.01,shrink=0.6)
    #axs[1,0].set_title(r"$($"+title+r"$)_+$")
    axs[1,0].set_title(r"$($"+title+r"$)_{0.99}}$")
    mp3=axs[1,1].matshow(mat-mat2)
    #tmp = np.dot(eivec[:,:npos], np.diag(1.0/eival[:npos]))
    #mat2inv = np.dot(eivec[:,:npos], tmp.T)
    #mp3=axs[1,1].matshow(np.dot(mat,mat2inv))
    fig.colorbar(mp3,ax=axs[1,1],pad=0.01,shrink=0.6)
    #axs[1,1].set_title(r"$\|$"+title+r"$-($"+title+r"$)_+\|_F=$"+f"{la.norm(mat-mat2,ord='fro'):.3e}")
    axs[1,1].set_title(r"$\|$"+title+r"$-($"+title+r"$)_{0.99}\|_F=$"+f"{la.norm(mat-mat2,ord='fro'):.3e}")
    #axs[1,1].set_title(title+r"$($"+title+r"$)_+^{-1}$")
    fig.savefig(f"{figdir}/{label}.png",dpi=300)
    #plt.show()
    plt.close()

    if label != "ekbmat" and label != "ebkmat":
        stdv = np.sqrt(np.diag(mat))
        cmat = np.diag(1.0/stdv) @ mat @ np.diag(1.0/stdv)
        cmat_ncm = ncm(cmat,maxiter=100)
        fig, axs = plt.subplots(ncols=2,nrows=2,figsize=[10,8],constrained_layout=True)
        mp0=axs[0,0].matshow(cmat)
        fig.colorbar(mp0,ax=axs[0,0],pad=0.01,shrink=0.6)
        axs[0,0].set_title(r"$\mathrm{cond}(\mathbf{C})=$"+f"{la.cond(cmat):.3e}")
        eival,_ = la.eigh(cmat)
        axs[1,0].plot(np.arange(1,eival.size+1),eival[::-1],label=r"$\mathbf{C}$")
        npos = np.sum(eival>0.0)
        print(npos)
        axs[1,0].vlines([npos],0,1,colors='tab:blue',ls='dashdot',transform=axs[1,0].get_xaxis_transform())
        mp1=axs[0,1].matshow(cmat_ncm)
        fig.colorbar(mp1,ax=axs[0,1],pad=0.01,shrink=0.6)
        axs[0,1].set_title(r"$\mathrm{cond}(\mathbf{C}_\mathrm{NCM})=$"+f"{la.cond(cmat_ncm):.3e}")
        eival,_ = la.eigh(cmat_ncm)
        axs[1,0].plot(np.arange(1,eival.size+1),eival[::-1],label=r"$\mathbf{C}_\mathrm{NCM}$")
        npos = np.sum(eival>0.0)
        print(npos)
        axs[1,0].vlines([npos],0,1,colors='tab:orange',ls='dashdot',transform=axs[1,0].get_xaxis_transform())
        axs[1,0].set_ylabel("eigenvalues")
        axs[1,0].set_yscale("log")
        axs[1,0].set_xlim(1,eival.size)
        axs[1,0].legend()
        mp2=axs[1,1].matshow(cmat-cmat_ncm)
        fig.colorbar(mp2,ax=axs[1,1],pad=0.01,shrink=0.6)
        axs[1,1].set_title(r"$\|\mathbf{C}-\mathbf{C}_\mathrm{NCM}\|_F=$"+f"{la.norm(cmat-cmat_ncm,ord='fro'):.3e}")
        fig.savefig(f"{figdir}/{label}_ncm.png",dpi=300)
        #plt.show()
        plt.close()

        nx = cmat.shape[0]
        fig, ax = plt.subplots(figsize=[10,6])
        #if cyclic:
        corrmean = cmat[0,:(nx//2)]
        #else:
        #    corrmean = cmat[0,]
        #ax.plot(np.arange(corrmean.size),corrmean,lw=0.0,marker='.',c='tab:blue')
        corrstdv = corrmean**2
        for i in range(1,nx):
            if cyclic:
                corr = np.roll(cmat[i,:],-i)[:(nx//2)]
            else:
                if i<nx-i:
                    corr = np.roll(cmat[i,:],-i)[:(nx//2)]
                else:
                    corr = np.flip(cmat[i,:i+1])[:(nx//2)]
            corrmean += corr
            corrstdv += corr**2
            #ax.plot(np.arange(corr.size),corr,lw=0.0,marker='.',c='tab:blue')
        corrmean /= float(nx)
        corrstdv = np.sqrt(corrstdv/float(nx) - corrmean**2)
        x = (xaxis[label]-xaxis[label][0])*2.0*np.pi/ix_true.size
        x_deg = np.rad2deg(x)
        #ax.errorbar(x,corrmean,yerr=corrstdv)
        ax.plot(x_deg[:(nx//2)],corrmean,label='NMC mean')
        ax.fill_between(x_deg[:(nx//2)],corrmean-corrstdv,corrmean+corrstdv,alpha=0.5)
        ax.hlines([0],x_deg[0],x_deg[nx//2],colors='k',ls='dotted',zorder=0) #,transform=ax.get_yaxis_transform())
        ax.set_title(title+" correlation")
        i0=0
        while(i0<len(corrmean)):
            if corrmean[i0] < 0.0: break
            i0 += 1
        i1 = i0
        while(i1<len(corrmean)):
            if corrmean[i1] > 0.0: break
            i1 += 1
        print(i0,i1)
        ax.vlines([x_deg[i0],x_deg[i1]],0,1,colors='r',ls='dotted',transform=ax.get_xaxis_transform())
        ## polynomial fittings
        #axs[1].plot(x,corrmean,lw=0.0,marker='x',ms=5.0,label='data')
        #x_latent = np.linspace(0,corrmean.size-1,corrmean.size*2-1)*np.pi/corrmean.size
        x_latent = x[:(nx//2)]
        #for k in range(3,10):
        #    coef = np.polyfit(x,corrmean,k)
        #    fitted_curve = np.poly1d(coef)(x)
        #    #label="y="
        #    #for index, c in enumerate(coef):
        #    #    if (len(coef) - index - 1)==1:
        #    #        label += f"{c:.2e} * x +"
        #    #    elif (len(coef) - index - 1)==0:
        #    #        label += f"{c:.2e}"
        #    #    else:
        #    #        label += f"{c:.2e} * x ** {(len(coef) - index - 1)} + "
        #    res = np.mean((fitted_curve - corrmean)**2)
        #    lab=f"{k}-order, res={res:.3e}"
        #    axs[1].plot(x,fitted_curve,label=lab)
        #    np.savetxt(f"{figdir}/{label}_poly{k}_coef.txt",coef)
        #fitted_curve = interp1d(np.arange(corrmean.size),corrmean,kind="cubic")
        #axs[1].plot(x_latent,fitted_curve(x_latent),label="cubic-spline")
        if functype=="tri":
            dx1 = x[i0] - x[0]
            nj1 = int(2.0*np.pi/dx1)
            print(f"Nj={nj1}")
            cardinal1 = np.where(x_latent-x[0]==0.0,1.0,np.sin(nj1*(x_latent-x[0])/2.0)/np.tan((x_latent-x[0])/2.0)/nj1)
            ax.plot(np.rad2deg(x_latent),cardinal1,ls='dashdot',lw=2.0,label=f'trigonometric, N={nj1}')
            dx1 = (x[i1] - x[0]) / 2.0
            nj2 = int(2.0*np.pi/dx1)
            print(f"Nj={nj2}")
            cardinal2 = np.where(x_latent-x[0]==0.0,1.0,np.sin(nj2*(x_latent-x[0])/2.0)/np.tan((x_latent-x[0])/2.0)/nj2)
            ax.plot(np.rad2deg(x_latent),cardinal2,ls='dashdot',lw=2.0,label=f'trigonometric, N={nj2}')
            ax.plot(np.rad2deg(x_latent),0.5*(cardinal1+cardinal2),ls='dashdot',lw=3.0,label=f'trigonometric, mean')
        elif functype=="gc5":
            c = (x[i1] - x[0]) / 1.2
            r = x_latent - x[0]
            a1 = -0.2
            c0 = Corrfunc(c,a=a1)
            c1 = c0(r,ftype='gc5')
            ax.plot(np.rad2deg(x_latent),c1,ls='dashdot',lw=2.0,label=f'GC5, a={a1:.1f}, c[deg]={np.rad2deg(c):.1f}')
            a2 = -0.1
            c0 = Corrfunc(c,a=a2)
            c2 = c0(r,ftype='gc5')
            ax.plot(np.rad2deg(x_latent),c2,ls='dashdot',lw=2.0,label=f'GC5, a={a2:.1f}, c[deg]={np.rad2deg(c):.1f}')
            # optimization
            args = (r,corrmean)
            minimize = Minimize(2,calc_j,args=args,method='Nelder-Mead')
            p0 = np.array([a1,c])
            popt, flg = minimize(p0)
            c0 = Corrfunc(popt[1],a=popt[0])
            copt = c0(r,ftype='gc5')
            ax.plot(np.rad2deg(x_latent),copt,ls='dashdot',lw=3.0,label=f'GC5opt, a={popt[0]:.1f}, c[deg]={np.rad2deg(popt[1]):.1f}')
        ax.set_xlabel("distance [degree]")
        ax.legend()
        fig.savefig(f"{figdir}/{label}_corr{functype}.png",dpi=300)
        plt.show()
        plt.close()
        # comparison
        fig, axs = plt.subplots(nrows=3,ncols=3,figsize=[12,12],constrained_layout=True)
        p0 = axs[0,0].matshow(cmat,vmin=-1.0,vmax=1.0,cmap='bwr')
        fig.colorbar(p0,ax=axs[0,0],pad=0.01,shrink=0.5)
        axs[0,0].set_title(r"cond$(\mathbf{C}_\mathrm{orig})=$"+f"{la.cond(cmat):.3e}")
        axs[1,0].remove()
        axs[2,0].remove()
        cmattri = np.zeros_like(cmat)
        cmattri1 = np.zeros_like(cmat)
        cmattri2 = np.zeros_like(cmat)
        dist = np.zeros_like(cmattri)
        nx = dist.shape[0]
        for i in range(nx):
            for j in range(nx):
                dist[i,j] = np.abs(x[i]-x[j])
                if cyclic: dist[i,j] = min(dist[i,j],2.0*np.pi-dist[i,j])
            if functype=="tri":
                cmattri1[i] = np.where(dist[i]==0.0,1.0,np.sin(nj1*dist[i]/2.0)/np.tan(dist[i]/2.0)/nj1) 
                cmattri2[i] = np.where(dist[i]==0.0,1.0,np.sin(nj2*dist[i]/2.0)/np.tan(dist[i]/2.0)/nj2)
            elif functype=="gc5":
                c = (x[i1] - x[0]) / 1.2
                c0 = Corrfunc(popt[1],a=popt[0])
                c1 = Corrfunc(c,a=a1)
                c2 = Corrfunc(c,a=a2)
                if cyclic:
                    ctmp = c1(np.roll(dist[i,],-i)[:(dist.shape[0]//2)+1],ftype='gc5')
                    ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:-1])])
                    cmattri1[i,] = np.roll(ctmp2,i)
                    ctmp = c2(np.roll(dist[i,],-i)[:(dist.shape[0]//2)+1],ftype='gc5')
                    ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:-1])])
                    cmattri2[i,] = np.roll(ctmp2,i)
                    ctmp = c0(np.roll(dist[i,],-i)[:(dist.shape[0]//2)+1],ftype='gc5')
                    ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:-1])])
                    cmattri[i,] = np.roll(ctmp2,i)
                else:
                    if i < dist.shape[0]-i:
                        ctmp = c1(np.roll(dist[i,],-i)[:dist.shape[0]-i],ftype='gc5')
                        ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:])])
                        cmattri1[i,] = np.roll(ctmp2,i)[:dist.shape[0]]
                        ctmp = c2(np.roll(dist[i,],-i)[:dist.shape[0]-i],ftype='gc5')
                        ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:])])
                        cmattri2[i,] = np.roll(ctmp2,i)[:dist.shape[0]]
                        ctmp = c0(np.roll(dist[i,],-i)[:dist.shape[0]-i],ftype='gc5')
                        ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:])])
                        cmattri[i,] = np.roll(ctmp2,i)[:dist.shape[0]]
                    else:
                        ctmp = c1(np.flip(dist[i,:i+1]),ftype='gc5')
                        ctmp2 = np.hstack([np.flip(ctmp),ctmp[1:]])
                        cmattri1[i,] = ctmp2[:dist.shape[0]]
                        ctmp = c2(np.flip(dist[i,:i+1]),ftype='gc5')
                        ctmp2 = np.hstack([np.flip(ctmp),ctmp[1:]])
                        cmattri2[i,] = ctmp2[:dist.shape[0]]
                        ctmp = c0(np.flip(dist[i,:i+1]),ftype='gc5')
                        ctmp2 = np.hstack([np.flip(ctmp),ctmp[1:]])
                        cmattri[i,] = ctmp2[:dist.shape[0]]
        if functype=="tri": cmattri = 0.5*cmattri1 + 0.5*cmattri2
        p01 = axs[0,1].matshow(cmattri1,vmin=-1.0,vmax=1.0,cmap='bwr')
        fig.colorbar(p01,ax=axs[0,1],pad=0.01,shrink=0.5)
        p02 = axs[0,2].matshow(cmat-cmattri1,vmin=-1.0,vmax=1.0,cmap='bwr')
        fig.colorbar(p02,ax=axs[0,2],pad=0.01,shrink=0.5)
        p11 = axs[1,1].matshow(cmattri2,vmin=-1.0,vmax=1.0,cmap='bwr')
        fig.colorbar(p11,ax=axs[1,1],pad=0.01,shrink=0.5)
        p12 = axs[1,2].matshow(cmat-cmattri2,vmin=-1.0,vmax=1.0,cmap='bwr')
        fig.colorbar(p12,ax=axs[1,2],pad=0.01,shrink=0.5)
        p21 = axs[2,1].matshow(cmattri,vmin=-1.0,vmax=1.0,cmap='bwr')
        fig.colorbar(p21,ax=axs[2,1],pad=0.01,shrink=0.5)
        p22 = axs[2,2].matshow(cmat-cmattri,vmin=-1.0,vmax=1.0,cmap='bwr')
        fig.colorbar(p22,ax=axs[2,2],pad=0.01,shrink=0.5)
        if functype=="tri":
            axs[0,1].set_title(r"cond$(\mathbf{C}_\mathrm{%s}^{N_j=%d})=$"%(functype,nj1)+f"{la.cond(cmattri1):.3e}")
            axs[0,2].set_title(r"$\|\mathbf{C}_\mathrm{orig}-\mathbf{C}_\mathrm{%s}\|_F=$"%functype+f"{la.norm(cmat-cmattri1,ord='fro'):.3e}")
            axs[1,1].set_title(r"cond$(\mathbf{C}_\mathrm{%s}^{N_j=%d})=$"%(functype,nj2)+f"{la.cond(cmattri2):.3e}")
            axs[1,2].set_title(r"$\|\mathbf{C}_\mathrm{orig}-\mathbf{C}_\mathrm{%s}\|_F=$"%functype+f"{la.norm(cmat-cmattri2,ord='fro'):.3e}")
            axs[2,1].set_title(r"cond$(\mathbf{C}_\mathrm{%s})=$"%functype+f"{la.cond(cmattri):.3e}")
            axs[2,2].set_title(r"$\|\mathbf{C}_\mathrm{orig}-\mathbf{C}_\mathrm{%s}\|_F=$"%functype+f"{la.norm(cmat-cmattri,ord='fro'):.3e}")
        elif functype=="gc5":
            axs[0,1].set_title(r"cond$(\mathbf{C}_\mathrm{%s}^{a=%.1f})=$"%(functype,a1)+f"{la.cond(cmattri1):.3e}")
            axs[0,2].set_title(r"$\|\mathbf{C}_\mathrm{orig}-\mathbf{C}_\mathrm{%s}\|_F=$"%functype+f"{la.norm(cmat-cmattri1,ord='fro'):.3e}")
            axs[1,1].set_title(r"cond$(\mathbf{C}_\mathrm{%s}^{a=%.1f})=$"%(functype,a2)+f"{la.cond(cmattri2):.3e}")
            axs[1,2].set_title(r"$\|\mathbf{C}_\mathrm{orig}-\mathbf{C}_\mathrm{%s}\|_F=$"%functype+f"{la.norm(cmat-cmattri2,ord='fro'):.3e}")
            axs[2,1].set_title(r"cond$(\mathbf{C}_\mathrm{%s}^\mathrm{opt})=$"%functype+f"{la.cond(cmattri):.3e}")
            axs[2,2].set_title(r"$\|\mathbf{C}_\mathrm{orig}-\mathbf{C}_\mathrm{%s}\|_F=$"%functype+f"{la.norm(cmat-cmattri,ord='fro'):.3e}")
        fig.savefig(f"{figdir}/{label}_corrmat{functype}.png",dpi=300)
        plt.show()
        plt.close()
exit()
figdir = figdir2
eival, eivec = la.eigh(bmat_lam)
eival = eival[::-1]
eivec = eivec[:,::-1]
npos = np.sum(eival>0.0)
bsqrt = np.dot(eivec[:,:npos], np.diag(np.sqrt(eival[:npos])))
alpha=0.0
vmat=bmat_gm2lam + alpha*np.eye(bmat_gm2lam.shape[0])
stdv = np.sqrt(np.diag(vmat))
cmat = np.diag(1.0/stdv) @ bmat_gm2lam @ np.diag(1.0/stdv)
cmat_ncm = ncm(cmat,maxiter=100)
eival, eivec = la.eigh(cmat_ncm)
eival = eival[::-1]
eivec = eivec[:,::-1]
npos = np.sum(eival>0.0)
#accum = [eival[:i].sum()/eival.sum() for i in range(1,eival.size+1)]
#npos=0
#while True:
#    if accum[npos] > 0.99: break
#    npos += 1
vsqrt = np.dot(np.diag(stdv),np.dot(eivec[:,:npos],np.diag(np.sqrt(eival[:npos]))))
vsqrtinv = np.dot(np.dot(np.diag(1.0/np.sqrt(eival[:npos])),eivec[:,:npos].T),np.diag(1.0/stdv))
mat = vsqrt @ vsqrt.T
matinv = vsqrtinv.T @ vsqrtinv
fig, axs = plt.subplots(nrows=2,ncols=2,figsize=[8,8],constrained_layout=True)
mp0=axs[0,0].matshow(bmat_gm2lam)
fig.colorbar(mp0,ax=axs[0,0],pad=0.01,shrink=0.6)
axs[0,0].set_title(r"$\mathrm{cond}(\mathbf{V})=$"+f"{la.cond(bmat_gm2lam):.3e}")
mp1=axs[0,1].matshow(mat)
fig.colorbar(mp1,ax=axs[0,1],pad=0.01,shrink=0.6)
axs[0,1].set_title(r"$\mathrm{cond}(\mathbf{V}_{+})=$"+f"{la.cond(mat):.3e}")
mp2=axs[1,0].matshow(matinv)
fig.colorbar(mp2,ax=axs[1,0],pad=0.01,shrink=0.6)
axs[1,0].set_title(r"$\mathbf{V}_{+}^{-1}$")
mp3=axs[1,1].matshow(matinv@mat)
fig.colorbar(mp3,ax=axs[1,1],pad=0.01,shrink=0.6)
axs[1,1].set_title(r"$\mathbf{V}_{+}\mathbf{V}_{+}^{-1}$")
fig.savefig(f"{figdir}/vmat.png",dpi=300)
plt.show()
plt.close()

from scipy.interpolate import interp1d
i0 = np.argmin(np.abs(ix_gm-ix_lam[0]))
if ix_gm[i0]<ix_lam[0]: i0+=1
i1 = np.argmin(np.abs(ix_gm-ix_lam[-1]))
if ix_gm[i1]>ix_lam[-1]: i1-=1
lam2gm = interp1d(ix_lam,np.eye(ix_lam.size),axis=0)
JH2 = lam2gm(ix_gm[i0:i1+1])
mat = bsqrt.T @ JH2.T @ vsqrtinv.T @ vsqrtinv @ JH2 @ bsqrt
fig, ax = plt.subplots()
mp = ax.matshow(mat)
fig.colorbar(mp,ax=ax,pad=0.01,shrink=0.6)
ax.set_title(r"$\mathbf{B}^{1/2}\mathbf{H}_2^\mathrm{T}\mathbf{V}^{-1}\mathbf{H}_2\mathbf{B}^{1/2}$")
plt.show()
