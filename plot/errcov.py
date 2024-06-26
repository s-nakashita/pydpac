import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
from nmc_tools import NMC_tools, wnum2wlen, wlen2wnum
sys.path.append(os.path.join(os.path.dirname(__file__),'../analysis'))
from trunc1d import Trunc1d

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
dscl = False
if len(sys.argv)>4:
    dscl = (sys.argv[4]=='T')

datadir = Path(f'/Volumes/FF520/nested_envar/data/{model}')
#datadir = Path(f'../work/{model}')
preGMpt = 'envar'
gmdir = datadir / 'var_vs_envar_dscl_m80obs30'
lamdir  = datadir / 'var_vs_envar_shrink_dct_preGM_m80obs30'

perts = ["envar", "envar_nest"] #,"var","var_nest"]

t = np.arange(na)
ix_gm = np.loadtxt(gmdir/"ix_gm.txt")
ix_lam = np.loadtxt(lamdir/"ix_lam.txt")[1:-1]
ntrunc = 12
trunc_operator = Trunc1d(ix_lam,ntrunc=ntrunc,ttype='c',cyclic=False) #,resample=False)
ix_trunc = trunc_operator.ix_trunc #[1:-1]
tname = trunc_operator.tname[trunc_operator.ttype]
gm2lam = interp1d(ix_gm,np.eye(ix_gm.size),axis=0)
H_gm2lam = gm2lam(ix_lam)
print(H_gm2lam.shape)

ns = 40
for pt in perts:
    coef_a_list = []
    for icycle in range(ns,na):
        #GM
        f = gmdir/f"data/{preGMpt}/{model}_gm_uf_{op}_{preGMpt}_cycle{icycle}.npy"
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        xfgm = np.load(f)
        xdgm = xfgm - np.mean(xfgm,axis=1)[:,None]
        #LAM
        f = lamdir/f"data/{pt}/{model}_lam_uf_{op}_{pt}_cycle{icycle}.npy"
        if not os.path.isfile(f):
            print("not exist {}".format(f))
            continue
        xflam = np.load(f)
        xdlam = xflam - np.mean(xflam,axis=1)[:,None]
        #
        xdv1 = trunc_operator(H_gm2lam@xdgm)
        xdv2 = trunc_operator(xdlam)
        Vtmp = np.dot(xdv1,xdv1.T)/float(xdv1.shape[1]-1)
        H2BH2tmp = np.dot(xdv2,xdv2.T)/float(xdv2.shape[1]-1)
        H2BVtmp = np.dot(xdv2,xdv1.T)/float(xdv2.shape[1]-1)
        coef_a = np.trace(H2BVtmp)/np.trace(H2BH2tmp)
        coef_a_list.append(coef_a)
        eta = xdv1 - coef_a * xdv2
        H2Betatmp = np.dot(xdv2,eta.T)/float(xdv2.shape[1]-1)
        eta2tmp = np.dot(eta,eta.T)/float(eta.shape[1]-1)
        if icycle == ns:
            V = Vtmp.copy()
            H2BH2 = H2BH2tmp.copy()
            H2BV = H2BVtmp.copy()
            H2Beta = H2Betatmp.copy()
            eta2 = eta2tmp.copy()
        else:
            V = V + Vtmp
            H2BH2 = H2BH2 + H2BH2tmp
            H2BV = H2BV + H2BVtmp
            H2Beta = H2Beta + H2Betatmp
            eta2 = eta2 + eta2tmp
    V = V / float(len(coef_a_list))
    H2BH2 = H2BH2 / float(len(coef_a_list))
    H2BV = H2BV / float(len(coef_a_list))
    H2Beta = H2Beta / float(len(coef_a_list))
    eta2 = eta2 / float(len(coef_a_list))

    fig, ax = plt.subplots(figsize=[6,4],constrained_layout=True)
    cycles = np.arange(ns,na)
    coef_a = np.array(coef_a_list).mean()
    ax.plot(cycles,coef_a_list,label=f'mean={coef_a:.3e}')
    ax.hlines([coef_a],0,1,colors='tab:blue',ls='dashed',transform=ax.get_yaxis_transform())
    ax.legend()
    ax.set_xlabel('cycles')
    ax.set_ylabel('a')
    fig.savefig(datadir/"coef_a.png",dpi=300)
    plt.show()
    plt.close()

    fig, axs = plt.subplots(ncols=3,nrows=2,figsize=[10,8],constrained_layout=True)
    mplist = []
    vlim = max(np.max(V),-np.min(V))
    mp00=axs[0,0].matshow(V,cmap='bwr',norm=Normalize(-vlim,vlim))
    axs[0,0].set_title(r'$\langle \varepsilon^\mathrm{v}(\varepsilon^\mathrm{v})^\mathrm{T}\rangle$')
    mplist.append(mp00)
    mp01=axs[0,1].matshow(coef_a*coef_a*H2BH2,cmap='bwr',norm=Normalize(-vlim,vlim))
    axs[0,1].set_title(r'$a^2\langle H_2(\varepsilon^\mathrm{b})(H_2(\varepsilon^\mathrm{b}))^\mathrm{T}\rangle$')
    mplist.append(mp01)
    mp02=axs[0,2].matshow(coef_a*(H2Beta+H2Beta.T),cmap='bwr',norm=Normalize(-vlim,vlim))
    axs[0,2].set_title(r'$a\langle H_2(\varepsilon^\mathrm{b})\eta^\mathrm{T}+\eta(H_2(\varepsilon^\mathrm{b}))^\mathrm{T}\rangle$')
    mplist.append(mp02)
    summat=coef_a*coef_a*H2BH2+eta2
    mp10=axs[1,0].matshow(summat,cmap='bwr',norm=Normalize(-vlim,vlim))
    axs[1,0].set_title(r'$a^2\langle H_2(\varepsilon^\mathrm{b})(H_2(\varepsilon^\mathrm{b}))^\mathrm{T}\rangle+\langle\eta\eta^\mathrm{T}\rangle$')
    mplist.append(mp10)
    diff = V - summat
    mp11=axs[1,1].matshow(diff,cmap='bwr',norm=Normalize(-vlim,vlim))
    axs[1,1].set_title('[0,0] - [1,0]')
    mplist.append(mp11)
    mp12=axs[1,2].matshow(eta2,cmap='bwr',norm=Normalize(-vlim,vlim))
    axs[1,2].set_title(r'$\langle\eta\eta^\mathrm{T}\rangle$')
    mplist.append(mp12)
    for mp, ax in zip(mplist,axs.flatten()):
        fig.colorbar(mp,ax=ax,shrink=0.6,pad=0.01)
    #fig.colorbar(mp00,ax=axs,shrink=0.6,pad=0.01)
    fig.suptitle(r'$\varepsilon^\mathrm{v}=aH_2(\varepsilon^\mathrm{b})+\eta$')
    fig.savefig(datadir/"errcov_estimate.png",dpi=300)
    plt.show()
    plt.close()
    exit()