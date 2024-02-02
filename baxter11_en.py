import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dst, idst, rfft, irfft, ifft
from scipy.interpolate import interp1d
from numpy.random import default_rng
from analysis.obs import Obs
from analysis.envar import EnVAR
from analysis.envar_nest import EnVAR_nest
from pathlib import Path
plt.rcParams['font.size'] = 14
cmap = plt.get_cmap('tab10')
import logging
from logging.config import fileConfig
fileConfig('logging_config.ini')
logger = logging.getLogger(__name__)

figdir_parent = Path('work/baxter11_en')
if not figdir_parent.exists():
    figdir_parent.mkdir(parents=True)

## Domain and step size definition
L = 1.0
T = 0.5
dx = 0.0625
dt = 0.05
dx_t = dx / 8.
dt_t = dt / 64.
dx_gm = dx
dt_gm = dt
dx_lam = dx / 4.
dt_lam = dt / 16.
lamstep = int(dt_gm / dt_lam)
obsstep = int(dt_lam / dt_t)
nx_t = int(L / dx_t)
ix_t = np.arange(1,nx_t+1)
x_t = np.linspace(dx_t,L,nx_t)
nx_gm = int(L / dx_gm)
ix_gm = np.arange(1,nx_gm+1)*int(nx_t/nx_gm)
x_gm = np.linspace(dx_gm,L,nx_gm)
Ls_lam = 0.5
nx_lam = int((L - Ls_lam) / dx_lam) + 1
x_lam = np.linspace(Ls_lam,L,nx_lam)
ix_lam = x_lam * nx_t / L
nsponge = 3
print(f"nx_t={nx_t} nx_gm={nx_gm} nx_lam={nx_lam}")
print(f"dx_t={dx_t} dx_gm={dx_gm} dx_lam={dx_lam}")
print(f"dt_t={dt_t} dt_gm={dt_gm} dt_lam={dt_lam}")

# Assimilation settings
sigo = 0.5
sigb = 0.5
## background error covariance
sig_gm = np.diag(np.full(x_gm.size,sigb)) #*nx_gm
U_gm = idst(sig_gm,n=len(x_gm),type=1,axis=0,norm='ortho')
#U_gm = irfft(sig_gm,len(u0_gm),axis=0)
print(U_gm.shape)
B_gm = U_gm @ U_gm.transpose()
print(B_gm.shape)
#B_gm = np.diag(np.full(u0_gm.size,sigb*sigb))
sig_lam = np.diag(np.full(x_lam.size-2,sigb)) #*nx_lam
#sig_lam[:7] = sig_lam[:7]/np.sqrt(5.0)
U_lam = idst(sig_lam,n=len(x_lam)-2,type=1,axis=0,norm='ortho')
#U_lam = irfft(sig_lam,len(u0_lam),axis=0)
print(U_lam.shape)
B_lam = U_lam @ U_lam.transpose()
print(B_lam.shape)
#B_lam = np.diag(np.full(u0_lam.size,sigb*sigb))
"""
fig, axs = plt.subplots(ncols=2,constrained_layout=True)
axs[0].plot(U_gm[:,::2])
axs[0].set_title(r'$\mathbf{U}_\mathrm{GM}$')
axs[1].plot(U_lam[:,::4])
axs[1].set_title(r'$\mathbf{U}_\mathrm{LAM}$')
plt.show()
"""
fig, axs = plt.subplots(ncols=2,constrained_layout=True)
p0=axs[0].matshow(B_gm)
fig.colorbar(p0,ax=axs[0],shrink=0.6)
axs[0].set_title(r'$\mathbf{B}_\mathrm{GM}$')
p1=axs[1].matshow(B_lam)
fig.colorbar(p1,ax=axs[1],shrink=0.6)
axs[1].set_title(r'$\mathbf{B}_\mathrm{LAM}$')
#plt.show()
plt.close()

## DA
nmem = 120
infl_parm = 1.05
obsloc = ix_lam[1:-1]
nobs = obsloc.size
obsope = Obs('linear',sigo)
obsope_gm = Obs('linear',sigo,ix=ix_gm)
obsope_lam = Obs('linear',sigo,ix=ix_lam[1:-1],icyclic=False)
envar_gm = EnVAR(nx_gm, nmem, obsope_gm)
envar_lam = EnVAR(nx_lam-2, nmem, obsope_lam)
envar_nest = EnVAR_nest(nx_lam-2, nmem, obsope_lam, ix_gm, ix_lam[1:-1], ntrunc=7, cyclic=False)

## random seed
rng = default_rng()

## start trials
ntrial = 30
rmseb_list = []
rmsea_list = []
rmsea_nest_list = []
errspecb_list = []
errspeca_list = []
errspeca_nest_list = []
itrial = 0
while itrial < ntrial:
    itrial += 1
    logger.info(f"== trial {itrial} nobs={nobs} nmem={nmem} ==")
    savefig=False
    if itrial <= 30:
        savefig=True
        figdir = figdir_parent / f'nobs{nobs}nmem{nmem}_test{itrial}'
        if not figdir.exists():
            figdir.mkdir(parents=True)
    ## nature and backgrounds
    u0_t = 5.*np.sin(np.pi*x_t) + np.sin(2.0*np.pi*x_t) + np.sin(36.*np.pi*x_t)
    u0_gm = 5.*np.sin(np.pi*x_gm) + np.sin(2.0*np.pi*x_gm)
    gm2lam = interp1d(x_gm,u0_gm)
    u0_lam = gm2lam(x_lam)
    
    ## ensemble
    X0_gm = rng.standard_normal(size=(nx_gm,nmem))*sigb*infl_parm
    #X0_gm[-1,:] = 0.0
    X0_gm = X0_gm - np.mean(X0_gm,axis=1)[:,None]
    u_gm = u0_gm[:,None] + X0_gm
    X0_lam = rng.standard_normal(size=(nx_lam,nmem))*sigb*infl_parm
    X0_lam = X0_lam - np.mean(X0_lam,axis=1)[:,None]
    u_lam = np.zeros((nx_lam,nmem))
    u_lam[:,:] = u0_lam[:,None] + X0_lam
    #u_lam[0,:] = u0_lam[0]
    #u_lam[-1,:] = u0_lam[-1]
    #u_lam[1:-1,:] = u0_lam[1:-1,None] + X0_lam
    
    X0_gm = u_gm - np.mean(u_gm,axis=1)[:,None]
    Pf_gm = X0_gm @ X0_gm.transpose() / (nmem-1)
    X0_lam = u_lam - np.mean(u_lam,axis=1)[:,None]
    Pf_lam = X0_lam @ X0_lam.transpose() / (nmem-1)
    
    fig, axs = plt.subplots(figsize=[6,4],ncols=2,constrained_layout=True)
    p0=axs[0].matshow(Pf_gm)
    fig.colorbar(p0,ax=axs[0],shrink=0.6)
    axs[0].set_title(r'$\mathbf{P}^f_\mathrm{GM}$')
    p1=axs[1].matshow(Pf_lam)
    fig.colorbar(p1,ax=axs[1],shrink=0.6)
    axs[1].set_title(r'$\mathbf{P}^f_\mathrm{LAM}$')
    if savefig:
        fig.savefig(figdir/'Pens.png',dpi=300)
        fig.savefig(figdir/'Pens.pdf')
    if itrial==1: plt.show()
    plt.close()

    ## sine transform
    y_t = dst(u0_t[:-1],type=1)/nx_t
    wnum_t = np.arange(1,y_t.size+1)/(2*nx_t*dx_t)
    #y_t = rfft(u0_t)*2./nx_t
    #wnum_t = np.arange(y_t.size)/(nx_t*dx_t)
    y_gm = dst(u_gm[:-1,:],type=1,axis=0)/nx_gm
    wnum_gm = np.arange(1,y_gm.shape[0]+1)/(2*nx_gm*dx_gm)
    #y_gm = rfft(u0_gm)*2./nx_gm
    #wnum_gm = np.arange(y_gm.size)/(nx_gm*dx_gm)
    y_lam = dst(u_lam[1:-1,:],type=1,axis=0)/(nx_lam-1)
    wnum_lam = np.arange(1,y_lam.shape[0]+1)/(2*(nx_lam-1)*dx_lam)
    #y_lam = rfft(u0_lam)*2./nx_lam
    #wnum_lam = np.arange(y_lam.size)/(nx_lam*dx_lam)

    ## adding perturbations to background state
    #yp_gm = y_gm*nx_gm + rng.normal(0, scale=sigb, size=y_gm.size)
    #up_gm = np.zeros_like(u0_gm)
    #up_gm[:-1] = idst(yp_gm,type=1)
    #yp_gm = yp_gm/nx_gm
    ##yp_gm = y_gm*nx_gm/2. + rng.normal(0, scale=sigb, size=y_gm.size)
    ##up_gm = irfft(yp_gm,len(u0_gm))
    yp_lam = y_lam*nx_lam + rng.normal(0, scale=sigb, size=y_lam.shape)
    up_lam = np.zeros_like(u_lam)
    up_lam[0,] = u_lam[0,]; up_lam[-1,] = u_lam[-1,]
    up_lam[1:-1,] = idst(yp_lam,type=1,axis=0)
    yp_lam = yp_lam/nx_lam
    #yp_lam = y_lam*nx_lam/2. + rng.normal(0, scale=sigb, size=y_lam.size)
    #up_lam = irfft(yp_lam,len(u0_lam))

    width=0.15
    fig, axs = plt.subplots(nrows=2,figsize=[8,6],constrained_layout=True)
    axs[0].plot(x_t,u0_t,c=cmap(0),label='nature')
    axs[0].plot(x_gm,np.mean(u_gm,axis=1),c=cmap(1),label='GM mean')
    axs[0].plot(x_gm,u_gm,lw=0.5,ls='dotted',c=cmap(1))
    #axs[0].plot(x_lam,u0_lam)
    #axs[0].plot(x_gm,up_gm)
    axs[0].plot(x_lam,np.mean(up_lam,axis=1),c=cmap(2),label='LAM mean')
    axs[0].plot(x_lam,up_lam,lw=0.5,ls='dotted',c=cmap(2))
    axs[1].bar(wnum_t-width,np.abs(y_t),width=width,label='nature')
    axs[1].bar(wnum_gm,np.mean(np.abs(y_gm),axis=1),\
        yerr=np.std(np.abs(y_gm),axis=1),width=width,label='GM')
    #axs[1].bar(wnum_lam+0.5*width,np.abs(y_lam),width=width)
    #axs[1].bar(wnum_gm-0.5*width,np.abs(yp_gm),width=width)
    axs[1].bar(wnum_lam+width,np.mean(np.abs(yp_lam),axis=1),\
        yerr=np.std(np.abs(y_lam),axis=1),width=width,label='LAM')
    axs[1].set_xlabel(r'wave number $k/L$')
    axs[1].set_xlim(0,20)
    axs[1].set_xticks(np.arange(21))
    axs[1].set_xticks(np.arange(41)/2.,minor=True)
    axs[1].legend()
    if savefig:
        fig.savefig(figdir/'nature+bg.png',dpi=300)
        fig.savefig(figdir/'nature+bg.pdf')
    plt.show(block=False)
    plt.close()

    ## observations
    yobs = obsope.add_noise(obsope.h_operator(obsloc, u0_t))

    ## analysis
    ua_lam = up_lam.copy()
    ua_lam_nest = up_lam.copy()
    ua_gm, _, _, _, _, _ = envar_gm(u_gm, X0_gm, yobs, obsloc)
    ua_lam[1:-1], _, _, _, _, _ = envar_lam(up_lam[1:-1], X0_lam[1:-1], yobs, obsloc)
    ua_lam_nest[1:-1], _, _, _, _, _ = envar_nest(up_lam[1:-1], X0_lam[1:-1], yobs, obsloc, u_gm)

    ## evaluation
    fig, axs = plt.subplots(nrows=3,figsize=[8,8],constrained_layout=True)
    #for ax in axs[0,]:
    axs[0].plot(x_t,u0_t,label='nature')
    #axs[0,0].plot(x_gm, u0_gm, label='GM,bg')
    #axs[0,0].plot(x_gm, ua_gm, label='GM,anl')
    axs[0].plot(x_lam, np.mean(up_lam,axis=1), label='LAM,bg')
    axs[0].plot(x_lam, np.mean(ua_lam,axis=1), label='LAM,anl')
    #axs[0].plot(x_lam, ua_lam, lw=0.5,ls='dotted',c=cmap(2))
    #axs[0,2].plot(x_lam, u0_lam, label='LAM,bg')
    axs[0].plot(x_lam, np.mean(ua_lam_nest,axis=1), label='LAM_nest,anl')
    #axs[0].plot(x_lam, ua_lam_nest, lw=0.5,ls='dotted',c=cmap(3))
    width=0.1
    for ax in axs[1:]:
        ax.bar(wnum_t-1.5*width,np.abs(y_t),width=width,label='nature')
    #for ax in axs[1,]:
    axs[1].set_xlim(0,10)
    axs[1].set_xticks(np.arange(11))
    axs[1].set_xticks(np.arange(21)/2.,minor=True)
    #for ax in axs[2,]:
    axs[2].set_xlim(9,20)
    axs[2].set_xticks(np.arange(9,21))
    axs[2].set_xticks(np.arange(18,41)/2.,minor=True)
    #yb_gm = dst(u0_gm[:-1],type=1)/nx_gm
    #ya_gm = dst(ua_gm[:-1],type=1)/nx_gm
    ##yb_gm = rfft(u0_gm)*2./nx_gm
    ##ya_gm = rfft(ua_gm)*2./nx_gm
    #axs[1,0].bar(wnum_gm,np.abs(yb_gm),width=width,label='GM,bg')
    #axs[1,0].bar(wnum_gm+width,np.abs(ya_gm),width=width,label='GM,anl')
    #axs[2,0].bar(wnum_gm,np.abs(yb_gm),width=width,label='GM,bg')
    #axs[2,0].bar(wnum_gm+width,np.abs(ya_gm),width=width,label='GM,anl')
    yb_lam = dst(up_lam[1:-1],type=1,axis=0)/(nx_lam-1)
    ya_lam = dst(ua_lam[1:-1],type=1,axis=0)/(nx_lam-1)
    #yb_lam = rfft(u0_lam)*2./nx_lam
    #ya_lam = rfft(ua_lam)*2./nx_lam
    axs[1].bar(wnum_lam-0.5*width,np.mean(np.abs(yb_lam),axis=1),\
        yerr=np.std(np.abs(yb_lam),axis=1),width=width,label='LAM,bg')
    axs[1].bar(wnum_lam+0.5*width,np.mean(np.abs(ya_lam),axis=1),\
        yerr=np.std(np.abs(ya_lam),axis=1),width=width,label='LAM,anl')
    axs[2].bar(wnum_lam-0.5*width,np.mean(np.abs(yb_lam),axis=1),\
        yerr=np.std(np.abs(yb_lam),axis=1),width=width,label='LAM,bg')
    axs[2].bar(wnum_lam+0.5*width,np.mean(np.abs(ya_lam),axis=1),\
        yerr=np.std(np.abs(ya_lam),axis=1),width=width,label='LAM,anl')
    ya_lam_nest = dst(ua_lam_nest[1:-1],type=1,axis=0)/(nx_lam-1)
    #ya_lam_nest = rfft(ua_lam_nest)*2./nx_lam
    #axs[1,2].bar(wnum_lam,np.abs(yb_lam),width=width,label='LAM,bg')
    axs[1].bar(wnum_lam+1.5*width,np.mean(np.abs(ya_lam_nest),axis=1),\
        yerr=np.std(np.abs(ya_lam_nest),axis=1),width=width,label='LAM_nest,anl')
    #axs[2,2].bar(wnum_lam,np.abs(yb_lam),width=width,label='LAM,bg')
    axs[2].bar(wnum_lam+1.5*width,np.mean(np.abs(ya_lam_nest),axis=1),\
        yerr=np.std(np.abs(ya_lam_nest),axis=1),width=width,label='LAM_nest,anl')
    for ax in axs.flatten():
        ax.legend()
    if savefig:
        fig.savefig(figdir/'nature+lamanl.png',dpi=300)
        fig.savefig(figdir/'nature+lamanl.pdf')
    plt.show(block=False)
    plt.close()

    ## error
    nature2model = interp1d(x_t,u0_t)
    #errb_gm = u0_gm - nature2model(x_gm)
    #erra_gm = ua_gm - nature2model(x_gm)
    errb_lam = np.mean(up_lam,axis=1) - nature2model(x_lam)
    erra_lam = np.mean(ua_lam,axis=1) - nature2model(x_lam)
    erra_lam_nest = np.mean(ua_lam_nest,axis=1) - nature2model(x_lam)
    rmseb = np.sqrt(np.mean(errb_lam**2))
    rmsea = np.sqrt(np.mean(erra_lam**2))
    rmsea_nest = np.sqrt(np.mean(erra_lam_nest**2))
    rmseb_list.append(rmseb)
    rmsea_list.append(rmsea)
    rmsea_nest_list.append(rmsea_nest)
    #yb_gm = dst(errb_gm[:-1],type=1)/nx_gm
    #ya_gm = dst(erra_gm[:-1],type=1)/nx_gm
    ##yb_gm = rfft(errb_gm)*2./nx_gm
    ##ya_gm = rfft(erra_gm)*2./nx_gm
    yb_lam = dst(errb_lam[1:-1],type=1,axis=0)/(nx_lam-1)
    ya_lam = dst(erra_lam[1:-1],type=1,axis=0)/(nx_lam-1)
    ya_lam_nest = dst(erra_lam_nest[1:-1],type=1,axis=0)/(nx_lam-1)
    #yb_lam = rfft(errb_lam)*2./nx_lam
    #ya_lam = rfft(erra_lam)*2./nx_lam
    #ya_lam_nest = rfft(erra_lam_nest)*2./nx_lam
    errspecb_list.append(np.abs(yb_lam))
    errspeca_list.append(np.abs(ya_lam))
    errspeca_nest_list.append(np.abs(ya_lam_nest))

    width=0.3
    fig, axs = plt.subplots(nrows=3,figsize=[8,8],constrained_layout=True)
    #axs[0,0].plot(x_gm, errb_gm, label='GM,bg')
    #axs[0,0].plot(x_gm, erra_gm, label='GM,anl')
    axs[0].plot(x_lam, errb_lam, label='LAM,bg\n'+f'rmse={rmseb:.3e}')
    axs[0].plot(x_lam, erra_lam, label='LAM,anl\n'+f'rmse={rmsea:.3e}')
    axs[0].plot(x_lam, erra_lam_nest, label='LAM_nest,anl\n'+f'rmse={rmsea_nest:.3e}')
    #for ax in axs[1,]:
    axs[1].set_xlim(0,10)
    axs[1].set_xticks(np.arange(11))
    axs[1].set_xticks(np.arange(21)/2.,minor=True)
    axs[1].set_xlabel('wave number k')
    #for ax in axs[2,]:
    axs[2].set_xlim(9,20)
    axs[2].set_xticks(np.arange(9,21))
    axs[2].set_xticks(np.arange(18,41)/2.,minor=True)
    axs[2].set_xlabel('wave number k')
    #axs[1].bar(wnum_gm-width,np.abs(yb_gm),width=width,label='GM,bg')
    #axs[1].bar(wnum_gm,np.abs(ya_gm),width=width,label='GM,anl')
    #axs[2].bar(wnum_gm-width,np.abs(yb_gm),width=width,label='GM,bg')
    #axs[2].bar(wnum_gm,np.abs(ya_gm),width=width,label='GM,anl')
    axs[1].bar(wnum_lam-width,np.abs(yb_lam),width=width,label='LAM,bg')
    axs[1].bar(wnum_lam      ,np.abs(ya_lam),width=width,label='LAM,anl')
    axs[1].bar(wnum_lam+width,np.abs(ya_lam_nest),width=width,label='LAM_nest,anl')
    axs[2].bar(wnum_lam-width,np.abs(yb_lam),width=width,label='LAM,bg')
    axs[2].bar(wnum_lam      ,np.abs(ya_lam),width=width,label='LAM,anl')
    axs[2].bar(wnum_lam+width,np.abs(ya_lam_nest),width=width,label='LAM_nest,anl')
    #for ax in axs[2,:]:
    axs[0].legend(bbox_to_anchor=(1.01,0.9))
    axs[1].set_ylim(0,0.2)
    fig.suptitle('error')
    if savefig:
        fig.savefig(figdir/'err.png',dpi=300)
        fig.savefig(figdir/'err.pdf')
    plt.show(block=False)
    plt.close()

fig, ax = plt.subplots(figsize=[10,6],constrained_layout=True)
ax.plot(np.arange(1,ntrial+1),rmseb_list,c=cmap(0),marker='^',\
    label='LAM,bg\n'+f'mean={np.mean(rmseb_list):.3f}')
ax.plot(np.arange(1,ntrial+1),rmsea_list,c=cmap(1),marker='^',\
    label='LAM,anl\n'+f'mean={np.mean(rmsea_list):.3f}')
ax.plot(np.arange(1,ntrial+1),rmsea_nest_list,c=cmap(2),marker='^',\
    label='LAM_nest,anl\n'+f'mean={np.mean(rmsea_nest_list):.3f}')
ax.hlines([np.mean(rmseb_list)],0,1,colors=cmap(0),ls='dashed',\
    transform=ax.get_yaxis_transform())
ax.hlines([np.mean(rmsea_list)],0,1,colors=cmap(1),ls='dashed',\
    transform=ax.get_yaxis_transform())
ax.hlines([np.mean(rmsea_nest_list)],0,1,colors=cmap(2),ls='dashed',\
    transform=ax.get_yaxis_transform())
ax.legend(bbox_to_anchor=(1.01,0.9))
ax.set_xlim(0,ntrial+1)
ax.set_xlabel('trial')
ax.set_ylabel('RMSE')
ax.set_title(f'ntrial={ntrial} nobs={nobs} nmem={nmem}, EnVar')
fig.savefig(figdir_parent/f'rmse_nobs{nobs}nmem{nmem}.png',dpi=300)
fig.savefig(figdir_parent/f'rmse_nobs{nobs}nmem{nmem}.pdf')
plt.show()

fig, axs = plt.subplots(figsize=[10,6],nrows=2)
errspecb = np.array(errspecb_list)
print(errspecb.shape)
errspecb_mean = np.mean(errspecb,axis=0)
errspecb_std  = np.std(errspecb,axis=0)
errspeca = np.array(errspeca_list)
errspeca_mean = np.mean(errspeca,axis=0)
errspeca_std  = np.std(errspeca,axis=0)
errspeca_nest = np.array(errspeca_nest_list)
errspeca_nest_mean = np.mean(errspeca_nest,axis=0)
errspeca_nest_std  = np.std(errspeca_nest,axis=0)
width=0.3
for ax in axs:
    ax.bar(wnum_lam-width,errspecb_mean,yerr=errspecb_std,width=width,label='LAM,bg')
    ax.bar(wnum_lam,      errspeca_mean,yerr=errspeca_std,width=width,label='LAM,anl')
    ax.bar(wnum_lam+width,errspeca_nest_mean,yerr=errspeca_nest_std,width=width,label='LAM_nest,anl')
    ax.set_ylabel('Absolute error')
axs[0].set_xlim(0,10)
axs[0].set_xticks(np.arange(11))
axs[0].set_xticks(np.arange(21)/2.,minor=True)
axs[0].set_ylim(0,0.2)
axs[1].set_xlim(9,20)
axs[1].set_xticks(np.arange(9,21))
axs[1].set_xticks(np.arange(18,41)/2.,minor=True)
axs[1].set_ylim(0,1.1)
axs[1].legend()
axs[1].set_xlabel('wave number k')
fig.suptitle(f'ntrial={ntrial} nobs={nobs} nmem={nmem}, EnVar')
fig.savefig(figdir_parent/f'errspec_nobs{nobs}nmem{nmem}.png',dpi=300)
fig.savefig(figdir_parent/f'errspec_nobs{nobs}nmem{nmem}.pdf')
plt.show()