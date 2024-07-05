import numpy as np 
import scipy.linalg as la
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
import sys
from test_ncm import ncm
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append('../')
from l05nest import nx_true, nx_gm, nk_gm, nx_lam, nsp, po, nk_lam, ni, b, c, F
figdir1 = Path(f'{parent_dir}/data/l05III/nmc_var')
figdir2 = Path(f'{parent_dir}/data/l05nest/nmc_var')
if not figdir1.exists():
    figdir1.mkdir(parents=True)
if not figdir2.exists():
    figdir2.mkdir(parents=True)
nobslist=[30,240]
bmatlist=[]
bmatgmlist=[]
bmatlamlist=[]
vmatlist=[]
ebkmatlist=[]
for nobs in nobslist:
    # global Lorenz III
    bmatdir = f"work/l05III/var_obs{nobs}"
    f = os.path.join(parent_dir,bmatdir,"l05III_B48m24.npy")
    bmat = np.load(f)
    bmatlist.append(bmat)
    # GM & LAM
    #bmatdir_nest = f"model/lorenz/ng{nx_gm}nl{nx_lam}kg{nk_gm}kl{nk_lam}nsp{nsp}p{po}F{int(F)}b{b:.1f}c{c:.1f}"
    bmatdir_nest = f"work/l05nest/var_obs{nobs}"
    f = os.path.join(parent_dir,bmatdir_nest,"ix_gm.txt")
    ix_gm = np.loadtxt(f)
    f = os.path.join(parent_dir,bmatdir_nest,"ix_lam.txt")
    ix_lam = np.loadtxt(f)
    #f = os.path.join(parent_dir,bmatdir_nest,"B_gmfull.npy")
    f = os.path.join(parent_dir,bmatdir_nest,"l05nest_B48m24_gm.npy")
    bmat_gm = np.load(f)
    bmatgmlist.append(bmat_gm)
    #f = os.path.join(parent_dir,bmatdir_nest,"B_lam.npy")
    f = os.path.join(parent_dir,bmatdir_nest,"l05nest_B48m24_lam.npy")
    bmat_lam = np.load(f)
    bmatlamlist.append(bmat_lam)
    #f = os.path.join(parent_dir,bmatdir_nest,"B_gm.npy")
    f = os.path.join(parent_dir,bmatdir_nest,"l05nest_V48m24.npy")
    bmat_gm2lam = np.load(f)
    vmatlist.append(bmat_gm2lam)
    #f = os.path.join(parent_dir,bmatdir_nest,"E_lg.npy")
    f = os.path.join(parent_dir,bmatdir_nest,"l05nest_B48m24_gm2lam.npy")
    ebkmat = np.load(f)
    ebkmatlist.append(ebkmat)

matrices = {"bmat":bmatlist,"bmat_gm":bmatgmlist,"bmat_lam":bmatlamlist,"bmat_gm2lam":vmatlist}
    #,"ebkmat":ebkmat,"ekbmat":ekbmat}
titles = {"bmat":"$\mathbf{B}_\mathrm{hres}$","bmat_gm":"$\mathbf{B}_\mathrm{GM}$","bmat_lam":"$\mathbf{B}_\mathrm{LAM}$","bmat_gm2lam":"$\mathbf{V}$","ebkmat":"$\mathbf{E}_\mathrm{bk}$","ekbmat":"$\mathbf{E}_\mathrm{kb}$"}
figdirs = {"bmat":figdir1,"bmat_gm":figdir2,"bmat_lam":figdir2,"bmat_gm2lam":figdir2}
for label in matrices.keys():
    #if label == "bmat": continue
    matlist = matrices[label]
    title = titles[label]
    figdir = figdirs[label]
    fig, axs = plt.subplots(ncols=2,nrows=3,figsize=[10,10],constrained_layout=True)
    for i, nobs in enumerate(nobslist):
        mat=matlist[i]
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

        diag = np.diag(mat)
        dsqrtinv = np.diag(1.0/np.sqrt(diag))
        print(np.diag(dsqrtinv))
        cmat = dsqrtinv @ mat @ dsqrtinv

        nx = mat.shape[0]
        axs[0,0].plot(mat[0,:],label=f'nobs={nobs}')
        axs[1,0].plot(mat[nx//2,:],label=f'nobs={nobs}')
        axs[2,0].plot(mat[-1,:],label=f'nobs={nobs}')
        axs[0,1].plot(cmat[0,:],label=f'nobs={nobs}')
        axs[1,1].plot(cmat[nx//2,:],label=f'nobs={nobs}')
        axs[2,1].plot(cmat[-1,:],label=f'nobs={nobs}')
    axs[0,0].set_title(title)
    axs[0,1].set_title("$\mathbf{C}$")
    axs[0,0].legend()
    fig.savefig(f"{figdir}/{label}.png",dpi=300)
    plt.show()
    plt.close()
