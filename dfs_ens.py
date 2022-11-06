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
        plt.show()
        plt.close()
    nmode = min(20,eval.size)
    lsqrt = np.dot(evec[:,:nmode],np.diag(np.sqrt(eval[:nmode])))
    return locmat, lsqrt

if __name__=="__main__":
    from numpy.random import default_rng
    logger = logging.getLogger(__name__)
    # dimensions
    M = 360
    p = 120
    # truth
    xt = np.zeros(M)
    # observarion
    intobs = int(M / p)
    obsloc = np.arange(0,M,intobs)
    logger.info(f"nobs={obsloc.size}")
    sigo = 1.0
    obs = Obs('linear',sigo)
    y = obs.add_noise(obs.h_operator(obsloc, xt))
    JH = obs.dh_operator(obsloc, xt)
    logger.debug(f"y={y[:10]}")
    logger.debug(f"JH={JH[1,:]}")

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
    xa, A, _, _, _, dfs_opt, eval = da_kf(xb,B,y,obsloc,evalout=True)
    trA_opt = np.sum(np.diag(A))
    logger.info(f"dfs_opt={dfs_opt:.3f}, trA_opt={trA_opt:.3f}")
    plt.plot(eval,label="optimal KF")

    #optimal 3DVar
    from analysis.var import Var
    da_var = Var(obs)
    xa, A, _, _, _, dfs_opt, eval = da_var(xb,B,y,obsloc,method='CG',evalout=True)
    trA_opt = np.sum(np.diag(A))
    logger.info(f"dfs_opt={dfs_opt:.3f}, trA_opt={trA_opt:.3f}")
    plt.plot(eval,ls='dashed',label="optimal Var")

    #EnKF
    from analysis.enkf import EnKF
    datype = "etkf"
    # first guess ensemble
    ens_list = [10,20,40,80,160,320,640,1280]
    dfs_list = []
    for K in ens_list:
        da_etkf = EnKF(datype,M,K,obs)
        Xb = np.dot(Bsqrt,rng.normal(loc=0.0,scale=1.0,size=(M,K)))
        Xb -= np.mean(Xb,axis=1)[:,None]
        xe = xb[:,None] + Xb
        Pf = np.dot(Xb,Xb.T)/(K-1)
        xa, A, _, _, _, dfs, eval = da_etkf(xe,Pf,y,obsloc,evalout=True)
        trA = np.sum(np.diag(A))
        logger.info(f"K={K} dfs={dfs:.3f}, trA={trA:.3f}")
        dfs_list.append(dfs)
        if K==40:
            plt.plot(eval,ls='dotted',label=f"EnKF, K={K}")

    #EnKF with modulated ensemble
    iloc = 2
    K = 40
    lsig = 10.0 * np.sqrt(10.0/3.0)
    lmat, lsqrt = locmat(M,lsig)
    da_etkfmod = EnKF(datype,M,K,obs,
    iloc=iloc, lsig=lsig, l_mat=lmat, l_sqrt=lsqrt)
    Xb = np.dot(Bsqrt,rng.normal(loc=0.0,scale=1.0,size=(M,K)))
    Xb -= np.mean(Xb,axis=1)[:,None]
    xe = xb[:,None] + Xb
    Pf = np.dot(Xb,Xb.T)/(K-1)
    xa, A, _, _, _, dfs_bloc, eval = da_etkfmod(xe,Pf,y,obsloc,evalout=True)
    trA = np.sum(np.diag(A))
    logger.info(f"B-loc K={K} dfs={dfs_bloc:.3f}, trA={trA:.3f}")
    plt.plot(eval,label="EnKF b-loc")

    #LETKF
    datype="letkf"
    lsig = 7.0
    da_letkf = EnKF(datype,M,K,obs,lsig=lsig)
    xa, A, _, _, _, dfs_rloc, eval = da_letkf(xe,Pf,y,obsloc,evalout=True)
    trA = np.sum(np.diag(A))
    logger.info(f"LETKF K={K} dfs={dfs_rloc:.3f}, trA={trA:.3f}")
    plt.plot(eval,label="LETKF R-loc")

    plt.legend()
    plt.show()
    plt.close()

    plt.plot(ens_list,dfs_list)
    plt.plot([K],dfs_bloc,marker='o',label='B-loc')
    plt.plot([K],dfs_rloc,marker='x',label='R-loc')
    plt.plot(ens_list,np.ones(len(ens_list))*dfs_opt,c='k',ls='dotted',label='optimal')
    plt.legend()
    plt.show()