import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from analysis.obs import Obs

fileConfig("logging_config.ini")

def cnag(M,d):
# square root of correlation matric (Bishop et al. 2015)
  knum = np.zeros(M)
  knum[0] = 0.0
  knum[M-1] = M/2.0
  for j in range(1,int(M/2)):
    knum[2*j-1] = float(j)
    knum[2*j] = float(j)
  #print(knum)
  lam = M*np.exp(-knum*knum/d/d)/np.sum(np.exp(-knum*knum/d/d))
  #print(lam)
  Es = np.eye(M)
  x = np.arange(M)
  Es[:,0] = 1.0 / np.sqrt(float(M))
  for j in range(1,int(M/2)):
    Es[:,2*j-1] = np.sin(knum[2*j-1]*x*2.0*np.pi/M)*np.sqrt(2.0/float(M))
    Es[:,2*j] = np.cos(knum[2*j]*x*2.0*np.pi/M)*np.sqrt(2.0/float(M))
  Es[:,M-1] = np.cos(knum[M-1]*x*2.0*np.pi/M)/np.sqrt(M)
  C = np.dot(np.dot(Es, np.diag(lam)), Es.T)
  return C, Es, lam

def GC(r, c):
  r1d = np.atleast_1d(r)
  f = np.zeros_like(r1d)
  for i in range(r1d.size):
    z = r1d[i] / c
    if z < 1.0:
      f[i] = 1.0 - 5.0*(z**2)/3.0 + 0.625*(z**3) + 0.5*(z**4) - 0.25*(z**5)
    elif z < 2.0:
      f[i] = 4.0 - 5.0*z + 5.0*(z**2)/3.0 + 0.625*(z**3) - 0.5*(z**4) + (z**5)/12.0 - 2.0/z/3.0
  return f

def locmat(M,c,verbose=False):
    from numpy import linalg as la
    locmat = np.eye(M)
    r = np.zeros(M)
    for i in range(M):
        for j in range(M):
            r[j] = min(abs(i-j),M-abs(i-j))*1.0
        locmat[i,] = GC(r,c)
    #print(r)
    eval,evec = la.eigh(locmat)
    eval = eval[::-1]
    evec = evec[:,::-1]
    if verbose:
        plt.plot(eval)
        #plt.show()
        plt.close()
    nmode = min(20,eval.size)
    lsqrt = np.dot(evec[:,:nmode],np.diag(np.sqrt(eval[:nmode])))
    return locmat, lsqrt

if __name__=="__main__":
    from numpy.random import default_rng
    import numpy.linalg as la
    plt.rcParams['font.size'] = 16
    logger0 = logging.getLogger(__name__)
    # dimensions
    M = 360
    p = 120
    # truth
    xt = np.zeros(M)
    # observarion
    intobs = int(M / p)
    obsloc = np.arange(0,M,intobs)
    print(f"nobs={obsloc.size}")
    sigo = 1.0
    obs = Obs('linear',sigo)
    y = obs.add_noise(obs.h_operator(obsloc, xt))
    JH = obs.dh_operator(obsloc, xt)
    logger0.debug(f"y={y[:10]}")
    logger0.debug(f"JH={JH[1,:]}")

    # background
    sigb = 1.0
    d = 20.0
    C, Es, lam = cnag(M,d)
    B = sigb * sigb * C
    Bsqrt = np.dot(Es, np.diag(sigb*np.sqrt(lam)))
    # first guess
    rng = default_rng()
    xb = xt + np.dot(Bsqrt,rng.normal(loc=0.0,scale=1.0,size=M))

    # optimal KF
    from analysis.kf import Kf
    da_kf = Kf(obs)
    xa, A, _, _, _, dfs_opt, eval_opt = da_kf(xb,B,y,obsloc,evalout=True)
    trA_opt = np.sum(np.diag(A))
    print(f"dfs_opt={dfs_opt:.3f}, trA_opt={trA_opt:.3f}")

    sigo_list = [0.1,0.2,0.5,1.0,2.0,5.0,10.0]
    dfs_list = []
    trA_list = []
    for sigo in sigo_list:
        obs = Obs('linear',sigo)
        y = obs.add_noise(obs.h_operator(obsloc, xt))
        da_kf = Kf(obs)
        xa, A, _, _, _, dfs_opt = da_kf(xb,B,y,obsloc)
        trA_opt = np.sum(np.diag(A))
        dfs_list.append(dfs_opt)
        trA_list.append(trA_opt)
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,6))
    ax[0].plot(sigo_list,np.array(dfs_list)/p,marker='o')
    ax[0].hlines([p/(M+p)],0,1,color='gray',transform=ax[0].get_yaxis_transform(),
    label=r'$N_\mathrm{obs}/(N_\mathrm{state}+N_\mathrm{obs})$')
    ax[0].set_xlabel('observation error')
    ax[0].set_xticks(sigo_list)
    ax[0].set_xticklabels(sigo_list)
    ax[0].legend()
    ax[0].set_title(r'$\mathrm{DFS}/N_\mathrm{obs}$, $\sigma^b=$'+f'{sigb}, '+r'$N_\mathrm{obs}=$'+f'{p}')
    ax[1].plot(sigo_list,np.array(trA_list)/M,marker='o',label=r'$\mathrm{tr}\mathbf{A}^\mathrm{opt}$')
    ax[1].set_xlabel('observation error')
    ax[1].set_xticks(sigo_list)
    ax[1].set_xticklabels(sigo_list)
    #ax[1].legend()
    ax[1].set_title(r'$\mathrm{tr}\mathbf{A}^\mathrm{opt}/N_\mathrm{state}, \sigma^b=$'+f'{sigb}, '+r'$N_\mathrm{obs}=$'+f'{p}')
    for i in range(2):
        ax[i].vlines([sigb],0,1,color='r',ls='dotted',
        transform=ax[i].get_xaxis_transform())
        ax[i].set_xscale('log')
    fig.tight_layout()
    fig.savefig('dfs-sigo.png')
    plt.show()

    nobs_list = [30,60,120,180,270,360]
    dfs_list = []
    trA_list = []
    for p in nobs_list:
        # observarion
        if M % p == 0:
            intobs = int(M / p)
            obsloc = np.arange(0,M,intobs)
        else:
            obsloc = np.random.choice(np.arange(M),size=p,replace=False)
        print(f"nobs={obsloc.size}")
        sigo = 1.0
        obs = Obs('linear',sigo)
        y = obs.add_noise(obs.h_operator(obsloc, xt))
        da_kf = Kf(obs)
        xa, A, _, _, _, dfs_opt = da_kf(xb,B,y,obsloc)
        trA_opt = np.sum(np.diag(A))
        dfs_list.append(dfs_opt)
        trA_list.append(trA_opt)
    fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12,6))
    nobs = np.array(nobs_list)
    ax[0].plot(nobs,np.array(dfs_list)/nobs,marker='o')
    ax[0].plot(nobs,nobs/(M+nobs),c='gray',
    label=r'$N_\mathrm{obs}/(N_\mathrm{state}+N_\mathrm{obs})$')
    ax[0].legend()
    ax[0].set_xlabel('# observation')
    ax[0].set_xticks(nobs)
    ax[0].set_xticklabels(nobs_list)
    ax[0].set_title(r'$\mathrm{DFS}/N_\mathrm{obs}$, $\sigma^b=$'+f'{sigb}, '+r'$\sigma^o=$'+f'{sigo}')
    ax[1].plot(nobs,np.array(trA_list)/M,marker='o',label=r'$\mathrm{tr}\mathbf{A}^\mathrm{opt}$')
    ax[1].set_xlabel('# observation')
    ax[1].set_xticks(nobs)
    ax[1].set_xticklabels(nobs_list)
    #ax[1].legend()
    ax[1].set_title(r'$\mathrm{tr}\mathbf{A}^\mathrm{opt}/N_\mathrm{state}, \sigma^b=$'+f'{sigb}, '+r'$\sigma^o=$'+f'{sigo}')
    fig.tight_layout()
    fig.savefig('dfs-nobs.png')
    plt.show()
    #exit()

    p = 120
    # truth
    xt = np.zeros(M)
    # observarion
    intobs = int(M / p)
    obsloc = np.arange(0,M,intobs)
    print(f"nobs={obsloc.size}")
    sigo = 1.0
    obs = Obs('linear',sigo)
    y = obs.add_noise(obs.h_operator(obsloc, xt))
    #optimal 3DVar
    from analysis.var import Var
    da_var = Var(obs)
    xa, A, _, _, _, dfs_opt, eval_opt = da_var(xb,B,y,obsloc,method='CG',evalout=True)
    trA_opt = np.sum(np.diag(A))
    print(f"dfs_opt={dfs_opt:.3f}, trA_opt={trA_opt:.3f}")

    itmax = 100
    #EnKF
    from analysis.enkf import EnKF
    # first guess ensemble
    ens_list = [10,20,40,80,160,320,640,1280]
    dfs_list = []
    amse_list = []
    trA_list = []
    eval_noloc = np.zeros(p)
    eval_bloc  = np.zeros(p)
    eval_rloc  = np.zeros(p)
    for it in range(itmax):
        print(f"iter={it}")
        rng = default_rng()
        y = obs.add_noise(obs.h_operator(obsloc, xt))
        xb = xt + np.dot(Bsqrt,rng.normal(loc=0.0,scale=1.0,size=M))
        for K in ens_list:
            datype = "etkf"
            da_etkf = EnKF(datype,M,K,obs)
            Xb = np.dot(Bsqrt,rng.normal(loc=0.0,scale=1.0,size=(M,K)))
            Xb -= np.mean(Xb,axis=1)[:,None]
            xe = xb[:,None] + Xb
            Pf = np.dot(Xb,Xb.T)/(K-1)
            xa, A, _, _, _, dfs, eval = da_etkf(xe,Pf,y,obsloc,evalout=True)
            amse = np.sum((xa.mean(axis=1)-xt)**2)
            amse_list.append(amse)
            trA = np.sum(np.diag(A))
            trA_list.append(trA)
            print(f"K={K} dfs={dfs:.3f}, trA={trA:.3f}")
            dfs_list.append(dfs)
            if K==40: eval_noloc += eval
        K=40
        #LETKF
        datype="letkf"
        lsig = 7.0
        iloc = 0
        Xb = np.dot(Bsqrt,rng.normal(loc=0.0,scale=1.0,size=(M,K)))
        Xb -= np.mean(Xb,axis=1)[:,None]
        xe = xb[:,None] + Xb
        Pf = np.dot(Xb,Xb.T)/(K-1)
        da_letkf = EnKF(datype,M,K,obs,iloc=iloc,lsig=lsig)
        xa, A, _, _, _, dfs, eval = da_letkf(xe,Pf,y,obsloc,evalout=True)
        dfs_list.append(dfs)
        amse = np.sum((xa.mean(axis=1)-xt)**2)
        amse_list.append(amse)
        trA = np.sum(np.diag(A))
        trA_list.append(trA)
        print(f"LETKF K={K} dfs={dfs:.3f}, trA={trA:.3f}")
        eval_rloc += eval
        #EnKF with modulated ensemble
        datype = "etkf"
        iloc = 2
        lsig = 10.0 * np.sqrt(10.0/3.0)
        lmat, lsqrt = locmat(M,lsig)
        da_etkfmod = EnKF(datype,M,K,obs,
            iloc=iloc, lsig=lsig, l_mat=lmat, l_sqrt=lsqrt)
        Xb = np.dot(Bsqrt,rng.normal(loc=0.0,scale=1.0,size=(M,K)))
        Xb -= np.mean(Xb,axis=1)[:,None]
        xe = xb[:,None] + Xb
        Pf = np.dot(Xb,Xb.T)/(K-1)
        xa, A, _, _, _, dfs, eval = da_etkfmod(xe,Pf,y,obsloc,evalout=True)
        dfs_list.append(dfs)
        amse = np.sum((xa.mean(axis=1)-xt)**2)
        amse_list.append(amse)
        trA = np.sum(np.diag(A))
        trA_list.append(trA)
        print(f"B-loc K={K} dfs={dfs:.3f}, trA={trA:.3f}")
        eval_bloc += eval

    dfs_mean = np.array(dfs_list).reshape(itmax,-1).mean(axis=0)
    amse_mean = np.array(amse_list).reshape(itmax,-1).mean(axis=0)
    trA_mean = np.array(trA_list).reshape(itmax,-1).mean(axis=0)

    noloc = len(ens_list)
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_axes((0.1,0.2,0.8,0.65))
    ax.plot(ens_list,dfs_mean[:noloc],label='noloc')
    ax.plot([K],dfs_mean[-1],marker='o',lw=0.0,label='B-loc')
    ax.plot([K],dfs_mean[noloc],marker='x',lw=0.0,label='R-loc')
    ax.plot(ens_list,np.ones(noloc)*dfs_opt,c='k',ls='dotted',label='optimal')
    ax.legend()
    ax.set_xticks(ens_list)
    ax.set_xlabel('ensemble size')
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    ax.set_title(r'$\mathrm{tr}\mathbf{HK}^\mathrm{ens}$'+' vs ens size,'+r'$\sigma^o=$'+f'{sigo}'+\
            r', $\mathrm{DFS}^\mathrm{opt}=$'+f'{dfs_opt:.2f}')
    fig.savefig(f'dfs_ens_sigo{sigo:.1f}.png')
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_axes((0.1,0.2,0.8,0.65))
    ax.plot(ens_list,trA_mean[:noloc],c='tab:orange',label=r'$\mathrm{tr}\mathbf{A}^\mathrm{ens}$')
    ax.plot([K],trA_mean[-1],c='tab:orange',marker='o',lw=0.0,label=r'$\mathrm{tr}\mathbf{A}^\mathrm{Bloc}$')
    ax.plot([K],trA_mean[noloc],c='tab:orange',marker='x',lw=0.0,label=r'$\mathrm{tr}\mathbf{A}^\mathrm{Rloc}$')
    ax.plot(ens_list,np.ones(noloc)*trA_opt,c='k',ls='dotted',label=r'$\mathrm{tr}\mathbf{A}^\mathrm{opt}$')
    ax.plot(ens_list,amse_mean[:noloc],c='tab:blue',label='MSE')
    ax.plot([K],amse_mean[-1],c='tab:blue',marker='o',lw=0.0,label='MSE,Bloc')
    ax.plot([K],amse_mean[noloc],c='tab:blue',marker='x',lw=0.0,label='MSE,Rloc')
    ax.legend(ncol=2)
    ax.set_xticks(ens_list)
    ax.set_xlabel('ensemble size')
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    ax.set_title(r'$\mathrm{tr}\mathbf{A}^\mathrm{ens}$ and analysis MSE, '+r'$\sigma^o=$'+f'{sigo}'+\
            r', $\mathrm{DFS}^\mathrm{opt}=$'+f'{dfs_opt:.2f}')
    fig.savefig(f'trA_ens_sigo{sigo:.1f}.png')
    plt.show()
    plt.close()
    #exit()

    eval_noloc /= itmax
    eval_bloc  /= itmax
    eval_rloc  /= itmax
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(eval_opt,label="optimal KF")
    ax.plot(eval_noloc,ls='dotted',label=f"EnKF noloc, K={40}")
    ax.plot(eval_bloc,label="EnKF b-loc")
    #ax.plot(eval_rloc,label="LETKF R-loc")
    ax.legend()
    ax.set_xlabel("# mode")
    fig.savefig(f'spectral_sigo{sigo:.1f}.png')
    plt.show()
    plt.close()

    # domain localization
    #itmax = 1
    datype="letkf"
    lsig = 7.0
    iloc = 0
    K=40
    da_letkf = EnKF(datype,M,K,obs,iloc=iloc,lsig=lsig)
    i = 0
    far, wgt = da_letkf.r_loc(lsig,obsloc,float(i))
    JH = obs.dh_operator(obsloc, xt)
    JHloc = np.delete(JH,far,axis=0)
    R, rmat, _ = obs.set_r(obsloc)
    Rloc = np.delete(np.delete(R,far,axis=0),far,axis=1)
    rmatloc = np.delete(np.delete(rmat,far,axis=0),far,axis=1)
    ## optimal KF
    D = np.dot(np.dot(JHloc,B),JHloc.T) + Rloc
    Dinv = la.inv(D)
    Kopt = np.dot(np.dot(B,JHloc.T),Dinv)
    A = np.dot((np.eye(M)-Kopt@JHloc),B)
    JH_ = np.dot(rmatloc,JHloc)
    infl_mat = np.dot(np.dot(JH_,A),JH_.T)
    eval_opt_loc, _ = la.eigh(infl_mat)
    eval_opt_loc = eval_opt_loc[::-1]
    eval_noloc_loc = np.zeros_like(eval_opt_loc)
    eval_letkf_loc = np.zeros_like(eval_opt_loc)
    for it in range(itmax):
        rng = default_rng()
        Xb = np.dot(Bsqrt,rng.normal(loc=0.0,scale=1.0,size=(M,K)))
        Xb -= np.mean(Xb,axis=1)[:,None]
        xe = xb[:,None] + Xb
        Pf = np.dot(Xb,Xb.T)/(K-1)
        ## EnKF without loc
        R, rmat, _ = obs.set_r(obsloc)
        Rloc = np.delete(np.delete(R,far,axis=0),far,axis=1)
        rmatloc = np.delete(np.delete(rmat,far,axis=0),far,axis=1)
        D = np.dot(np.dot(JHloc,Pf),JHloc.T) + Rloc
        Dinv = la.inv(D)
        Kens = np.dot(np.dot(Pf,JHloc.T),Dinv)
        A = np.dot((np.eye(M)-Kens@JHloc),Pf)
        JH_ = np.dot(rmatloc,JHloc)
        infl_mat = np.dot(np.dot(JH_,A),JH_.T)
        eval, _ = la.eigh(infl_mat)
        eval_noloc_loc += eval[::-1]
        ## LETKF
        Ri = np.diag(np.diag(R)/wgt)
        rmati = np.diag(np.diag(rmat)*np.sqrt(wgt))
        Rloc = np.delete(np.delete(Ri,far,axis=0),far,axis=1)
        rmatloc = np.delete(np.delete(rmati,far,axis=0),far,axis=1)
        JH_ = np.dot(rmatloc,JHloc)
        D = np.dot(np.dot(JHloc,Pf),JHloc.T) + Rloc
        Dinv = la.inv(D)
        Kens = np.dot(np.dot(Pf,JHloc.T),Dinv)
        A = np.dot((np.eye(M)-Kens@JHloc),Pf)
        infl_mat = np.dot(np.dot(JH_,A),JH_.T)
        eval, _ = la.eigh(infl_mat)
        eval_letkf_loc += eval[::-1]
    eval_noloc_loc /= itmax
    eval_letkf_loc /= itmax
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(eval_opt_loc,label="optimal KF")
    ax.plot(eval_noloc_loc,ls='dotted',label=f"EnKF noloc, K={40}")
    ax.plot(eval_letkf_loc,label="LETKF R-loc")
    ax.legend()
    ax.set_xlabel("# mode")
    ax.set_title("truncated eigenvalues")
    fig.savefig(f'spectral_loc_sigo{sigo:.1f}.png')
    plt.show()
    plt.close()

    # global eigenvalues for LETKF
    #itmax = 100
    datype="letkf"
    lsig = 7.0
    iloc = 0
    K=40
    da_letkf = EnKF(datype,M,K,obs,iloc=iloc,lsig=lsig)
    eval_domloc = np.zeros_like(eval_opt)
    eval_letkf = np.zeros_like(eval_opt)
    for it in range(itmax):
        rng = default_rng()
        Xb = np.dot(Bsqrt,rng.normal(loc=0.0,scale=1.0,size=(M,K)))
        Xb -= np.mean(Xb,axis=1)[:,None]
        xe = xb[:,None] + Xb
        Pf = np.dot(Xb,Xb.T)/(K-1)
        R, rmat, _ = obs.set_r(obsloc)
        rsqrt = np.sqrt(R)
        JH = obs.dh_operator(obsloc, xt)
        Kdom = np.zeros((M,p))
        Klet = np.zeros((M,p))
        for i in range(M):
            far, wgt = da_letkf.r_loc(lsig,obsloc,float(i))
            JHloc = np.delete(JH,far,axis=0)
            iloc = np.delete(np.arange(p),far)
            ## domain localization
            Rloc = np.delete(np.delete(R,far,axis=0),far,axis=1)
            rmatloc = np.delete(np.delete(rmat,far,axis=0),far,axis=1)
            D = np.dot(np.dot(JHloc,Pf),JHloc.T) + Rloc
            Dinv = la.inv(D)
            Kdom[i,iloc] = np.dot(np.dot(Pf,JHloc.T),Dinv)[i,]
            ## LETKF
            Ri = np.diag(np.diag(R)/wgt)
            rmati = np.diag(np.diag(rmat)*np.sqrt(wgt))
            Rloc = np.delete(np.delete(Ri,far,axis=0),far,axis=1)
            rmatloc = np.delete(np.delete(rmati,far,axis=0),far,axis=1)
            D = np.dot(np.dot(JHloc,Pf),JHloc.T) + Rloc
            Dinv = la.inv(D)
            Klet[i,iloc] = np.dot(np.dot(Pf,JHloc.T),Dinv)[i,]
        JH_ = np.dot(rmat,JH)
        tmp = np.dot(np.dot(JH_,Kdom),rsqrt)
        infl_mat = 0.5*(tmp+tmp.T)
        eval, _ = la.eigh(infl_mat)
        eval_domloc += eval[::-1]
        tmp = np.dot(np.dot(JH_,Klet),rsqrt)
        infl_mat = 0.5*(tmp+tmp.T)
        eval, _ = la.eigh(infl_mat)
        eval_letkf += eval[::-1]
    eval_domloc /= itmax
    eval_letkf /= itmax
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(eval_opt,label="optimal KF")
    ax.plot(eval_noloc,ls='dotted',label=f"EnKF noloc, K={40}")
    ax.plot(eval_domloc,ls='dashdot',label=f"domain loc")
    ax.plot(eval_letkf,ls='dashed',label="R-loc")
    ax.legend()
    ax.set_xlabel("# mode")
    ax.set_title("global eigenvalues")
    fig.savefig(f'spectral_glb_sigo{sigo:.1f}.png')
    plt.show()