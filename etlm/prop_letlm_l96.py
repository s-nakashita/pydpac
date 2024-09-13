import numpy as np 
from numpy.random import default_rng
import numpy.linalg as la 
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200
from matplotlib.lines import Line2D
import sys
sys.path.append('../')
from model.lorenz import L96
from analysis.corrfunc import Corrfunc
from pathlib import Path
import argparse

# model parameters
nx = 40
dt = 0.05 # 6 hour
F = 8.0
model = L96(nx, dt, F)
ix = np.arange(nx)
r = np.array([min(ix[i],nx-ix[i]) for i in range(nx)])

# settings
parser = argparse.ArgumentParser()
parser.add_argument("-vt","--vt",type=int,default=24,\
    help="verification time (hours)")
parser.add_argument("-ne","--nens",type=int,default=8,\
    help="ensemble size")
parser.add_argument("-lp","--lpatch",type=int,default=1,\
    help="patch size")
parser.add_argument("-lh","--lhalo",type=int,default=1,\
    help="halo size")
argsin = parser.parse_args()
vt = argsin.vt
ioffset = vt // 6
nens = argsin.nens
lpatch = argsin.lpatch
lhalo = argsin.lhalo
# patch locations
npatch = nx // lpatch
ispatch = np.zeros(npatch,dtype=int)
iepatch = np.zeros(npatch,dtype=int)
ishalo = np.zeros(npatch,dtype=int)
iehalo = np.zeros(npatch,dtype=int)
lletlm = np.zeros(npatch,dtype=int)
for i in range(npatch):
    ispatch[i] = i*lpatch
    iepatch[i] = ispatch[i]+lpatch-1
    ishalo[i] = ispatch[i] - lhalo
    iehalo[i] = iepatch[i] + lhalo
    lletlm[i] = iehalo[i] - ishalo[i] + 1
    if iehalo[i] > nx-1:
        iehalo[i] = nx - iehalo[i]
    print(f"ispatch={ispatch[i]} iepatch={iepatch[i]} ishalo={ishalo[i]} iehalo={iehalo[i]} lletlm={lletlm[i]}")

# data
modelname = 'l96'
pt = 'letkf'
datadir = Path(f'/Volumes/dandelion/pyesa/data/{modelname}')
if nens==200:
    xfall = np.load(datadir/f"extfcst_letkf_m{nens}/{modelname}_ufext_linear_{pt}.npy")
else:
    xfall = np.load(datadir/f"extfcst_m{nens}/{modelname}_ufext_linear_{pt}.npy")
print(xfall.shape)
P = np.load(datadir/'Aest.npy')
Lam, v = la.eigh(P)
Psqrt = v @ np.diag(np.sqrt(Lam)) @ v.T
#plt.matshow(Psqrt)
#plt.colorbar()
#plt.show(block=False)

figdir = Path(f'l96/lp{lpatch}lh{lhalo}')
if not figdir.exists(): figdir.mkdir(parents=True)

icyc0 = 50
ncycle = 100
beta = 0.1 # Tikhonov regularization
rng = default_rng(seed=517)
# localization functions (rho0=gc5, rho1=boxcar)
cfunc = Corrfunc(2.0)
rho0 = np.roll(cfunc(r, ftype='gc5'),nx//2)
ic = ix[nx//2]
rho1 = np.where(np.abs(ix-ic)<2.0,1.0,0.0)

rmse_dx = []
rmse_rho0_dyn = []
rmse_rho1_dyn = []
rmse_rho0_ens = []
rmse_rho1_ens = []
for icyc in range(icyc0,icyc0+ncycle):
    print(f"cycle={icyc}")
    # arbitrally perturbations
    dx0 = Psqrt @ rng.normal(loc=0.0,scale=1.0,size=Psqrt.shape[1])
    # ensemble
    Xens = xfall[icyc,:,:,:]
    xb = np.mean(Xens[:,:,:],axis=2)
    Xprtb = Xens - xb[:,:,None]
    if icyc-icyc0 < 1:
        fig, axs = plt.subplots(nrows=3,sharex=True,figsize=[6,6],constrained_layout=True)
        axs[0].set_title(f'cycle{icyc} FT00 K={nens}')
        axs[0].plot(ix,Xens[0,],c='magenta',alpha=0.3,lw=1.0)
        axs[0].plot(ix,xb[0,],c='k',label='ens mean')
        axs[2].plot(ix,dx0,c='k',label=r'$\delta\mathbf{x}_0$',zorder=0)
    dx_dyn = dx0.copy()
    dx_ens = dx0.copy()
    xp = xb[0] + dx0
    if icyc-icyc0 < 1:
        axs[0].plot(ix,xp,c='gray',label='+prtb')
    tlmdyn_list = []
    for i in range(ioffset):
        # nonlinear model
        xp = model(xp)
        # dynamical TLM
        dx_dyn = model.step_t(xb[i],dx_dyn)
        tlm = np.eye(nx)
        for j in range(nx):
            tlm[:,j] = model.step_t(xb[i],tlm[:,j])
        tlmdyn_list.append(tlm)
        # localized ensemble TLM
        Xi = Xprtb[i,:,:]
        Xj = Xprtb[i+1,:,:]
        for l in range(npatch):
            Xip = np.roll(Xi,-ishalo[l],axis=0)[:lletlm[l]]
            Xjp = np.roll(Xj,-ishalo[l],axis=0)[:lletlm[l]]
            cimat = Xip.T @ Xip + beta*np.eye(Xip.shape[1])
            y = Xip.T @ np.roll(dx_ens,-ishalo[l])[:lletlm[l]]
            z = la.pinv(cimat,rcond=1e-4,hermitian=True) @ y
            dx_tmp = Xjp @ z
            dx_ens[ispatch[l]:iepatch[l]+1] = dx_tmp[lhalo:-lhalo]
        if icyc-icyc0 < 1:
            axs[2].plot(ix,dx_dyn,c='b',ls='dashed',alpha=0.5,lw=0.5)
            axs[2].plot(ix,dx_ens,c='r',ls='dashed',alpha=0.5,lw=0.5)
    if icyc-icyc0 < 1:
        axs[1].set_title(f'cycle{icyc} FT{vt:02d} K={nens}')
        axs[1].plot(ix,Xens[ioffset,],c='magenta',alpha=0.3,lw=1.0)
        axs[1].plot(ix,xb[ioffset,],c='k',label='ens mean')
        axs[1].plot(ix,xp,c='gray',label='nl fcst')
        axs[2].set_title(f'prtb development lpatch={lpatch} lhalo={lhalo}')
        axs[2].plot(ix,xp-xb[ioffset],c='gray',label='nl diff')
        axs[2].plot(ix,dx_dyn,c='b',ls='dashed',label='TLM dyn')
        axs[2].plot(ix,dx_ens,c='r',ls='dashed',label='LETLM')
        for ax in axs:
            ax.legend()
        fig.savefig(figdir/f'dx_letlm_c{icyc}vt{vt}ne{nens}.png')
        plt.show(block=False)
    rmse_dx.append(np.sqrt(np.mean((dx_dyn-dx_ens)**2)))

    if icyc-icyc0 < 1:
        figr, axsr = plt.subplots(nrows=2,sharex=True,figsize=[6,6],constrained_layout=True)
        axsr[0].set_title('GC5')
        axsr[0].plot(ix,rho0,c='k',lw=3.0,label=r'$\rho_t$')
        axsr[1].set_title('Boxcar')
        axsr[1].plot(ix,rho1,c='k',lw=3.0,label=r'$\rho_t$')
    rho0_dyn = rho0.copy()
    rho1_dyn = rho1.copy()
    rho0_ens = rho0.copy()
    rho1_ens = rho1.copy()
    rho0_dyn_mat = np.diag(rho0_dyn)
    rho1_dyn_mat = np.diag(rho1_dyn)
    rho0_ens_mat = np.diag(rho0_ens)
    rho1_ens_mat = np.diag(rho1_ens)
    for i in range(ioffset):
        # dynamical TLM
        rho0_dyn_mat_pre = rho0_dyn_mat.copy()
        rho1_dyn_mat_pre = rho1_dyn_mat.copy()
        tlm = tlmdyn_list[ioffset-i-1]
        itlm = la.inv(tlm)
        if icyc-icyc0 < 1:
            ### debug
            fig, axs = plt.subplots(ncols=2,nrows=2,constrained_layout=True,figsize=[6,6])
            vlim = max(np.max(tlm),-np.min(tlm))
            p0=axs[0,0].matshow(tlm,vmin=-vlim,vmax=vlim,cmap='RdBu_r')
            p1=axs[0,1].matshow(itlm,vmin=-vlim,vmax=vlim,cmap='RdBu_r')
            p2=axs[1,0].matshow(itlm@tlm,vmin=-1.0,vmax=1.0,cmap='RdBu_r')
            p3=axs[1,1].matshow(tlm@itlm,vmin=-1.0,vmax=1.0,cmap='RdBu_r')
            fig.colorbar(p0,ax=axs[0,0],shrink=0.6)
            fig.colorbar(p1,ax=axs[0,1],shrink=0.6)
            fig.colorbar(p2,ax=axs[1,0],shrink=0.6)
            fig.colorbar(p3,ax=axs[1,1],shrink=0.6)
            axs[0,0].set_title(r'$\mathbf{M}_{'+f'{i*6},{(i+1)*6}'+r'}$')
            axs[0,1].set_title(r'$\mathbf{M}^{-1}_{'+f'{i*6},{(i+1)*6}'+r'}$')
            axs[1,0].set_title(r'$\mathbf{M}^{-1}\mathbf{M}$')
            axs[1,1].set_title(r'$\mathbf{M}\mathbf{M}^{-1}$')
            fig.suptitle(f'cycle{icyc}')
            fig.savefig(figdir/f'../dtlm_c{icyc}i{i}.png')
            plt.show(block=False)
            plt.close(fig=fig)
        ###
        rho0_dyn_mat = itlm@rho0_dyn_mat@tlm
        rho1_dyn_mat = itlm@rho1_dyn_mat@tlm
        ### minimize Frobenius norm of P_m - P_{m-1}
        #f0 = la.norm(rho0_dyn_mat,ord='fro')
        #f1 = la.norm(rho1_dyn_mat,ord='fro')
        #scale0 = np.sum(rho0_dyn_mat*rho0_dyn_mat_pre)/f0/f0
        #scale1 = np.sum(rho1_dyn_mat*rho1_dyn_mat_pre)/f1/f1
        #rho0_dyn_mat = rho0_dyn_mat * scale0
        #rho1_dyn_mat = rho1_dyn_mat * scale1
        ###
        rho0_dyn = np.diag(rho0_dyn_mat)
        rho1_dyn = np.diag(rho1_dyn_mat)
        # ensemble TLM
        rho0_ens_mat_pre = rho0_ens_mat.copy()
        rho1_ens_mat_pre = rho1_ens_mat.copy()
        Xj = Xprtb[ioffset-i,:,:]
        Xi = Xprtb[ioffset-i-1,:,:]
        for l in range(npatch):
            Xip = np.roll(Xi,-ishalo[l],axis=0)[:lletlm[l]]
            Xjp = np.roll(Xj,-ishalo[l],axis=0)[:lletlm[l]]
            cimat = Xip.T @ Xip + beta*np.eye(Xip.shape[1])
            cjmat = Xjp.T @ Xjp + beta*np.eye(Xjp.shape[1])
            #rho0tmp = np.roll(rho0_ens,-ishalo[l])[:lletlm[l]]
            #rho1tmp = np.roll(rho1_ens,-ishalo[l])[:lletlm[l]]
            rho0tmp = np.roll(np.roll(rho0_ens_mat,-ishalo[l],axis=0),-ishalo[l],axis=1)[:lletlm[l],:lletlm[l]]
            rho1tmp = np.roll(np.roll(rho1_ens_mat,-ishalo[l],axis=0),-ishalo[l],axis=1)[:lletlm[l],:lletlm[l]]
            #Xjl = rho0tmp[:,None] * Xjp
            Xjl = rho0tmp @ Xjp
            cjlmat = Xjp.T @ Xjl
            cmat0 = la.pinv(cjmat, rcond=1e-4, hermitian=True) @ cjlmat @ la.pinv(cimat, rcond=1e-4, hermitian=True)
            #Xjl = rho1tmp[:,None] * Xjp
            Xjl = rho1tmp @ Xjp
            cjlmat = Xjp.T @ Xjl
            cmat1 = la.pinv(cjmat, rcond=1e-4, hermitian=True) @ cjlmat @ la.pinv(cimat, rcond=1e-4, hermitian=True)
            #ii = lhalo
            #for i in range(ispatch[l],iepatch[l]+1):
            #    rho0_ens[i] = np.dot(Xip[ii],np.dot(cmat0,Xip[ii]))
            #    rho1_ens[i] = np.dot(Xip[ii],np.dot(cmat1,Xip[ii]))
            #    ii += 1
            i0 = ispatch[l]
            i1 = iepatch[l]+1
            rho0_ens_mat[i0:i1,i0:i1] = np.dot(Xip[lhalo:-lhalo],np.dot(cmat0,Xip[lhalo:-lhalo].T))
            rho1_ens_mat[i0:i1,i0:i1] = np.dot(Xip[lhalo:-lhalo],np.dot(cmat1,Xip[lhalo:-lhalo].T))
        ## maximum = 1.0
        scale0 = 1.0 / np.max(rho0_ens_mat)
        scale1 = 1.0 / np.max(rho1_ens_mat)
        ## minimize Frobenius norm of P_m - P_{m-1}
        #f0 = la.norm(rho0_ens_mat,ord='fro')
        #f1 = la.norm(rho1_ens_mat,ord='fro')
        #scale0 = np.sum(rho0_ens_mat*rho0_ens_mat_pre)/f0/f0
        #scale1 = np.sum(rho1_ens_mat*rho1_ens_mat_pre)/f1/f1
        ## minimize Frobenius norm of each row of P_m - P_{m-1}
        #f0 = np.sum(rho0_ens_mat*rho0_ens_mat,axis=1)
        #f1 = np.sum(rho1_ens_mat*rho1_ens_mat,axis=1)
        #scale0 = np.where(f0>0.0,np.sum(rho0_ens_mat*rho0_ens_mat_pre,axis=1)/f0/f0,0.0)
        #scale1 = np.where(f1>0.0,np.sum(rho1_ens_mat*rho1_ens_mat_pre,axis=1)/f1/f1,0.0)
        ## preserve the volume of localization function
        #scale0 = np.sum(np.abs(rho0_ens_mat_pre))/np.sum(np.abs(rho0_ens_mat))
        #scale1 = np.sum(np.abs(rho1_ens_mat_pre))/np.sum(np.abs(rho1_ens_mat))
        rho0_ens_mat = rho0_ens_mat * scale0 #[:,None] # scaling
        rho1_ens_mat = rho1_ens_mat * scale1 #[:,None] # scaling
        rho0_ens = np.diag(rho0_ens_mat)
        rho1_ens = np.diag(rho1_ens_mat)
        #rho0_ens = rho0_ens / np.max(rho0_ens) # scaling
        #rho1_ens = rho1_ens / np.max(rho1_ens) # scaling
        if icyc-icyc0 < 1:
            axsr[0].plot(ix,rho0_dyn,c='b',ls='dashed',alpha=0.5,lw=0.5)
            axsr[1].plot(ix,rho1_dyn,c='b',ls='dashed',alpha=0.5,lw=0.5)
            axsr[0].plot(ix,rho0_ens,c='r',ls='dashed',alpha=0.5,lw=0.5)
            axsr[1].plot(ix,rho1_ens,c='r',ls='dashed',alpha=0.5,lw=0.5)
    if icyc-icyc0 < 1:
        axsr[0].plot(ix,rho0_dyn,c='b',ls='dashed',label='dyn')
        axsr[1].plot(ix,rho1_dyn,c='b',ls='dashed',label='dyn')
        axsr[0].plot(ix,rho0_ens,c='r',ls='dashed',label='LETLM')
        axsr[1].plot(ix,rho1_ens,c='r',ls='dashed',label='LETLM')
        axsr[0].legend()
        figr.suptitle(f'cycle{icyc} FT{vt:02d} K={nens} lpatch={lpatch} lhalo={lhalo}')
        figr.savefig(figdir/f'prop_letlm_c{icyc}vt{vt}ne{nens}.png')
        plt.show(block=False)
        ## 
        fig, axs = plt.subplots(ncols=2,nrows=2,figsize=[6,6],constrained_layout=True)
        p0=axs[0,0].matshow(rho0_dyn_mat,vmin=-1.0,vmax=1.0,cmap='RdBu_r')
        p1=axs[0,1].matshow(rho1_dyn_mat,vmin=-1.0,vmax=1.0,cmap='RdBu_r')
        p2=axs[1,0].matshow(rho0_ens_mat,vmin=-1.0,vmax=1.0,cmap='RdBu_r')
        p3=axs[1,1].matshow(rho1_ens_mat,vmin=-1.0,vmax=1.0,cmap='RdBu_r')
        fig.colorbar(p0,ax=axs[0,0],shrink=0.6,pad=0.01)
        fig.colorbar(p1,ax=axs[0,1],shrink=0.6,pad=0.01)
        fig.colorbar(p2,ax=axs[1,0],shrink=0.6,pad=0.01)
        fig.colorbar(p3,ax=axs[1,1],shrink=0.6,pad=0.01)
        axs[0,0].set_title(r'GC5 $\mathbf{P}^\mathrm{dyn}=\mathbf{M}^{-1}\mathrm{diag}(\rho_t)\mathbf{M}$')
        axs[0,1].set_title(r'Boxcar $\mathbf{P}^\mathrm{dyn}=\mathbf{M}^{-1}\mathrm{diag}(\rho_t)\mathbf{M}$')
        axs[1,0].set_title(r'GC5 $\mathbf{P}^\mathrm{ens}=(\mathbf{M}^\mathrm{ens})^\dagger\mathrm{diag}(\rho_t)\mathbf{M}^\mathrm{ens}$')
        axs[1,1].set_title(r'Boxcar $\mathbf{P}^\mathrm{ens}=(\mathbf{M}^\mathrm{ens})^\dagger\mathrm{diag}(\rho_t)\mathbf{M}^\mathrm{ens}$')
        fig.suptitle(f'cycle{icyc} FT{vt:02d} K={nens} lpatch={lpatch} lhalo={lhalo}')
        fig.savefig(figdir/f'propmat_c{icyc}vt{vt}ne{nens}.png')
        plt.show(block=False)
        plt.close(fig=fig)

    # nonlinear check
    y = Psqrt @ rng.normal(loc=0.0,scale=1.0,size=Psqrt.shape[1])
    xp = xb[0] + y
    #rho0y_dyn = rho0_dyn * y
    #rho1y_dyn = rho1_dyn * y
    #rho0y_ens = rho0_ens * y
    #rho1y_ens = rho1_ens * y
    rho0y_dyn = rho0_dyn_mat @ y
    rho1y_dyn = rho1_dyn_mat @ y
    rho0y_ens = rho0_ens_mat @ y
    rho1y_ens = rho1_ens_mat @ y
    xp0_dyn = xb[0] + rho0y_dyn
    xp1_dyn = xb[0] + rho1y_dyn
    xp0_ens = xb[0] + rho0y_ens
    xp1_ens = xb[0] + rho1y_ens
    if icyc-icyc0 < 1:
        fig, axs = plt.subplots(nrows=2,ncols=2,sharex=True,figsize=[8,6],constrained_layout=True)
        axs[0,0].set_title('GC5 FT00')
        axs[0,1].set_title(f'GC5 FT{vt:02d}')
        axs[1,0].set_title('Boxcar FT00')
        axs[1,1].set_title(f'Boxcar FT{vt:02d}')
        axs[0,0].plot(ix,y,c='k',lw=2.0,zorder=0)
        axs[1,0].plot(ix,y,c='k',lw=2.0,zorder=0)
        axs[0,0].plot(ix,rho0y_dyn,c='b')
        axs[1,0].plot(ix,rho1y_dyn,c='b')
        axs[0,0].plot(ix,rho0y_ens,c='r')
        axs[1,0].plot(ix,rho1y_ens,c='r')
        lines = [
            Line2D([0],[0],color='k',lw=2.0),
            Line2D([0],[0],color='b'),
            Line2D([0],[0],color='r')
        ]
        labels = [
            r'$\mathbf{y}$',
            #r'$\mathbf{M}(\rho_0^\mathrm{dyn}\circ\mathbf{y})$',
            r'$\mathbf{P}_0^\mathrm{dyn}\mathbf{y}$',
            #r'$\mathbf{M}(\rho_0^\mathrm{ens}\circ\mathbf{y})$'
            r'$\mathbf{P}_0^\mathrm{ens}\mathbf{y}$'
        ]
        axs[0,0].legend(lines,labels)#loc='upper left',bbox_to_anchor=(1.01,1.0))
    for i in range(ioffset):
        # TLM
        y = model.step_t(xb[i],y)
        rho0y_dyn = model.step_t(xb[i],rho0y_dyn)
        rho1y_dyn = model.step_t(xb[i],rho1y_dyn)
        rho0y_ens = model.step_t(xb[i],rho0y_ens)
        rho1y_ens = model.step_t(xb[i],rho1y_ens)
        # NLM
        xp = model(xp)
        xp0_dyn = model(xp0_dyn)
        xp1_dyn = model(xp1_dyn)
        xp0_ens = model(xp0_ens)
        xp1_ens = model(xp1_ens)
    znl = xp - xb[ioffset]
    znl0_dyn = xp0_dyn - xb[ioffset]
    znl1_dyn = xp1_dyn - xb[ioffset]
    znl0_ens = xp0_ens - xb[ioffset]
    znl1_ens = xp1_ens - xb[ioffset]
    isig = np.arange(nx)[np.abs(rho0*znl)>1.0e-5]
    inoi = np.arange(nx)[np.abs(rho0*znl)<=1.0e-5]
    rmse_rho0_dyn.append([
        np.sqrt(np.mean(((rho0*znl)[isig]-znl0_dyn[isig])**2)),
        np.sqrt(np.mean(((rho0*znl)[inoi]-znl0_dyn[inoi])**2))
        ])
    rmse_rho0_ens.append([
        np.sqrt(np.mean(((rho0*znl)[isig]-znl0_ens[isig])**2)),
        np.sqrt(np.mean(((rho0*znl)[inoi]-znl0_ens[inoi])**2))
        ])
    isig = np.arange(nx)[np.abs(rho1*znl)>1.0e-5]
    inoi = np.arange(nx)[np.abs(rho1*znl)<=1.0e-5]
    rmse_rho1_dyn.append([
        np.sqrt(np.mean(((rho1*znl)[isig]-znl1_dyn[isig])**2)),
        np.sqrt(np.mean(((rho1*znl)[inoi]-znl1_dyn[inoi])**2))
        ])
    rmse_rho1_ens.append([
        np.sqrt(np.mean(((rho1*znl)[isig]-znl1_ens[isig])**2)),
        np.sqrt(np.mean(((rho1*znl)[inoi]-znl1_ens[inoi])**2))
        ])

    if icyc-icyc0 < 1:
        axs[0,1].plot(ix,rho0*y,c='k',lw=2.0,zorder=0)
        axs[1,1].plot(ix,rho1*y,c='k',lw=2.0,zorder=0)
        axs[0,1].plot(ix,rho0y_dyn,c='b')
        axs[1,1].plot(ix,rho1y_dyn,c='b')
        axs[0,1].plot(ix,rho0y_ens,c='r')
        axs[1,1].plot(ix,rho1y_ens,c='r')
        axs[0,1].plot(ix,rho0*znl,c='gray',lw=2.0,ls='dashed',zorder=0)
        axs[1,1].plot(ix,rho1*znl,c='gray',lw=2.0,ls='dashed',zorder=0)
        axs[0,1].plot(ix,znl0_dyn,c='cyan',ls='dashed')
        axs[1,1].plot(ix,znl1_dyn,c='cyan',ls='dashed')
        axs[0,1].plot(ix,znl0_ens,c='magenta',ls='dashed')
        axs[1,1].plot(ix,znl1_ens,c='magenta',ls='dashed')
        lines_lin = [
            Line2D([0],[0],color='k',lw=2.0),
            Line2D([0],[0],color='b'),
            Line2D([0],[0],color='r')
        ]
        labels_lin = [
            r'$\rho_t\circ(\mathbf{M}\mathbf{y})$',
            #r'$\mathbf{M}(\rho_0^\mathrm{dyn}\circ\mathbf{y})$',
            r'$\mathbf{M}(\mathbf{P}_0^\mathrm{dyn}\mathbf{y})$',
            #r'$\mathbf{M}(\rho_0^\mathrm{ens}\circ\mathbf{y})$'
            r'$\mathbf{M}(\mathbf{P}_0^\mathrm{ens}\mathbf{y})$'
        ]
        lines_nonlin = [
            Line2D([0],[0],color='gray',lw=2.0,ls='dashed'),
            Line2D([0],[0],color='cyan',ls='dashed'),
            Line2D([0],[0],color='magenta',ls='dashed')
        ]
        labels_nonlin = [
            r'$\rho_t\circ[M(\mathbf{x}+\mathbf{y})-M(\mathbf{x})]$',
            #r'$M(\mathbf{x}+\rho_0^\mathrm{dyn}\circ\mathbf{y})-M(\mathbf{x})$',
            r'$M(\mathbf{x}+\mathbf{P}_0^\mathrm{dyn}\mathbf{y})-M(\mathbf{x})$',
            #r'$M(\mathbf{x}+\rho_0^\mathrm{ens}\circ\mathbf{y})-M(\mathbf{x})$'
            r'$M(\mathbf{x}+\mathbf{P}_0^\mathrm{ens}\mathbf{y})-M(\mathbf{x})$'
        ]
        axs[0,1].legend(lines_lin,labels_lin,title='TLM')#loc='upper left',bbox_to_anchor=(1.01,1.0))
        axs[1,1].legend(lines_nonlin,labels_nonlin,title='NLM')
        fig.suptitle(f'cycle{icyc} FT{vt:02d} K={nens} lpatch={lpatch} lhalo={lhalo}')
        fig.savefig(figdir/f'test_prop_letlm_c{icyc}vt{vt}ne{nens}.png')
        plt.show(block=False)

np.savetxt(figdir/f'rmse_dx_letlm_vt{vt}ne{nens}.txt',rmse_dx)
np.savetxt(figdir/f'rmse_rho0_letlm_vt{vt}ne{nens}.txt',rmse_rho0_ens)
np.savetxt(figdir/f'rmse_rho1_letlm_vt{vt}ne{nens}.txt',rmse_rho1_ens)
np.savetxt(figdir/f'../rmse_rho0_dyn_vt{vt}.txt',rmse_rho0_dyn)
np.savetxt(figdir/f'../rmse_rho1_dyn_vt{vt}.txt',rmse_rho1_dyn)