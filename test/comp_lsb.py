import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
plt.rcParams['figure.dpi'] = 150
# model
from model.lorenz_nestm import L05nestm
model = "l05nestm"
## model parameter
### true
nx_true = 960
nk_true = 32
nks_true = [256,128,64,32]
### GM
gm_same_with_nature = False # DEBUG: Lorenz III used for GM
intgm = 4                    # grid interval
if gm_same_with_nature:
    intgm = 1
nx_gm = nx_true // intgm     # number of points
nk_gm = nk_true // intgm     # advection length scale
nks_gm = np.array(nks_true) // intgm     # advection length scales
dt_gm = 0.05 / 36            # time step (=1/6 hour, Kretchmer et al. 2015)
#dt_gm = 0.05 / 48            # time step (=1/8 hour, Yoon et al. 2012)
### LAM
nx_lam = 240                 # number of LAM points
ist_lam = 240                # first grid index
nsp = 10                     # width of sponge region
po = 1                       # order of relaxation function
intrlx = 1                   # interval of boundary relaxation (K15)
#intrlx = 48                   # interval of boundary relaxation (Y12)
lamstep = 1                  # time steps relative to 1 step of GM
nk_lam = 32                  # advection length
nks_lam = nks_true           # advection lengths
ni = 12                      # spatial filter width
b = 10.0                     # frequency of small-scale perturbation
c = 0.6                      # coupling factor
F = 15.0                     # forcing
## forecast model forward operator
step = L05nestm(nx_true, nx_gm, nx_lam, nks_gm, nks_lam, \
        ni, b, c, dt_gm, F, intgm, ist_lam, nsp, \
        lamstep=lamstep, intrlx=intrlx, po=po, gm_same_with_nature=gm_same_with_nature)
ix_gm = step.ix_gm
ix_lam = step.ix_lam
ix_true = step.ix_true
nx_gm = ix_gm.size
nx_lam = ix_lam.size
nx_true = ix_true.size
ix_lam_rad = ix_lam * 2.0 * np.pi / nx_true

# analysis
nmem = 80
## observation operator
from analysis.obs import Obs
obsope = Obs('linear',1.0,ix=ix_lam[1:-1],icyclic=False)
from scipy.interpolate import interp1d
## interpolation operator from GM to LAM
tmp_gm2lam = interp1d(ix_gm,np.eye(nx_gm),axis=0)
H_gm2lam = tmp_gm2lam(ix_lam)
## truncation operator (default)
from analysis.trunc1d import Trunc1d
trunc_kwargs = {'ntrunc':None,'cyclic':False,'ttype':'c','nghost':0}
#truncope = Trunc1d(ix_lam_rad,**trunc_kwargs)
## DA
from analysis.envar import EnVAR
from analysis.envar_nest import EnVAR_nest
from analysis.var import Var
from analysis.var_nest import Var_nest

envar = EnVAR(nx_lam-2,nmem,obsope)
#envar_nest = EnVAR_nest(nx_lam-2,nmem,obsope,ix_gm,ix_lam[1:-1],**trunc_kwargs)
envar_nest_initargs = (nx_lam-2,nmem,obsope,ix_gm,ix_lam[1:-1])
var_kwargs = dict(\
    ix=ix_lam[1:-1],ioffset=1,\
    sigb=0.6,functype='gc5',lb=np.deg2rad(28.77),a=-0.11,bmat=None,\
    cyclic=False,calc_dist1=step.calc_dist1_lam\
    )
var = Var(obsope,nx_lam-2,**var_kwargs)
var_nest_initargs = (obsope,ix_gm,ix_lam[1:-1])
var_nest_kwargs = {**var_kwargs,**trunc_kwargs,\
    **dict(vmat=None,sigv=0.6,lv=np.deg2rad(12.03),a_v=0.12,\
    calc_dist1_gm=step.calc_dist1_gm)
    }
del var_nest_kwargs['ix']
#var_nest = Var_nest(*var_nest_initargs,**var_nest_kwargs)

# data
gmdir = Path('../work/l05nestm/var_vs_envar_dscl_m80obs30')
lamdir = Path('../work/l05nestm/var_vs_envar_shrink_dct_preGM_m80obs30')

def loaddata(icycle):
    u_gm = np.load(gmdir/f'data/envar/l05nestm_gm_uf_linear_envar_cycle{icycle}.npy')
    u_tmp = np.load(lamdir/f'data/envar/l05nestm_lam_uf_linear_envar_cycle{icycle}.npy')
    u_lam = H_gm2lam @ u_gm
    u_lam[1:-1,:] = u_tmp[:,:]
    return u_gm, u_lam

def loadobs(icycle,obsloc=''):
    obsdir = Path(f'../work/l05nestm/var_vs_envar_shrink_dct_preGM{obsloc}_m80obs30')
    yobsall = np.load(obsdir/'obs_linear_10000.npy')
    yloc = yobsall[icycle,:,0]
    yobs = yobsall[icycle,:,1]
    iobslam = np.where((yloc > ix_lam[0])&(yloc<ix_lam[-1]), 1, 0)
    yloc_lam = yloc[iobslam==1]
    yobs_lam = yobs[iobslam==1]
    return yloc, yobs, yloc_lam, yobs_lam

# plotting function

def plot_spectra(data,labels,colors=None,styles=None,markers=None):
    from plot.nmc_tools import NMC_tools, wnum2wlen, wlen2wnum
    from matplotlib.ticker import FixedLocator, FixedFormatter
    from matplotlib.colors import ListedColormap

    nmc_lam = NMC_tools(ix_lam_rad,cyclic=False,ttype='c')

    if colors is None:
        cmap = plt.get_cmap('tab10')
    else:
        cmap = ListedColormap(colors)
    if styles is None:
        styles = ['solid']*len(data)
    if markers is None:
        markers = ['o']*len(data)
    fig, ax = plt.subplots(figsize=[6,4],constrained_layout=True)
    i = 0
    for u, label in zip(data,labels):
        wnum, psd = nmc_lam.psd(u,axis=0,average=False)
        psd = np.sum(psd,axis=1)/(u.shape[1]-1)
        ax.loglog(wnum,psd,c=cmap(i),ls=styles[i],marker=markers[i],ms=4,markevery=8,label=label)
        i+=1
    ax.grid()
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1.0))
    ax.set_xlabel(r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)")
    ax.xaxis.set_major_locator(FixedLocator([480,240,120,60,30,12,2]))
    ax.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','12','2']))
    secax = ax.secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
    secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
    secax.xaxis.set_major_locator(FixedLocator([np.pi,np.pi/6.,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
    secax.xaxis.set_major_formatter(FixedFormatter([r'$\pi$',r'$\frac{\pi}{6}$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
    return fig, ax

def plot_state(u_lam,u_dict,u_fil_dict,ntrunc_list,pt,yloc_lam=None,yobs_lam=None):
    fig, axs = plt.subplots(nrows=2,ncols=1+len(ntrunc_list),figsize=[12,6],\
        sharex=True,sharey=True,constrained_layout=True)
    figd,axsd = plt.subplots(nrows=2,ncols=len(ntrunc_list),figsize=[10,6],\
        sharex=True,sharey=True,constrained_layout=True)

    if pt=='envar':
        axs[0,0].plot(ix_lam,u_lam,c='gray',lw=0.5)
        axs[0,0].plot(ix_lam,u_lam.mean(axis=1),c='b',lw=2.0,label='mean')
    else:
        axs[0,0].plot(ix_lam,u_lam,c='b',lw=2.0)
    axs[1,0].remove()
    for i, ntrunc in enumerate(ntrunc_list):
        u = u_dict[ntrunc]
        u_fil = u_fil_dict[ntrunc]
        u_dif = u - u_lam
        u_diffil = u_fil - u_lam
        if pt=='envar':
            axs[0,i+1].plot(ix_lam,u,c='gray',lw=0.5)
            axs[0,i+1].plot(ix_lam,u.mean(axis=1),c='b',lw=2.0,label='mean')
            axsd[0,i].plot(ix_lam,u_dif,c='gray',lw=0.5)
            axsd[0,i].plot(ix_lam,u_dif.mean(axis=1),c='b',lw=2.0,label='mean')
            axs[1,i+1].plot(ix_lam,u_fil,c='gray',lw=0.5)
            axs[1,i+1].plot(ix_lam,u_fil.mean(axis=1),c='b',lw=2.0,label='mean')
            axsd[1,i].plot(ix_lam,u_diffil,c='gray',lw=0.5)
            axsd[1,i].plot(ix_lam,u_diffil.mean(axis=1),c='b',lw=2.0,label='mean')
        else:
            axs[0,i+1].plot(ix_lam,u,c='b',lw=2.0)
            axsd[0,i].plot(ix_lam,u_dif,c='b',lw=2.0)
            axs[1,i+1].plot(ix_lam,u_fil,c='b',lw=2.0)
            axsd[1,i].plot(ix_lam,u_diffil,c='b',lw=2.0)
        axs[0,i+1].set_title(f'ntrunc={ntrunc}')
        axsd[0,i].set_title(f'diff, ntrunc={ntrunc}')
        axs[1,i+1].set_title(f'ntrunc={ntrunc}, filtered')
        axsd[1,i].set_title(f'diff, ntrunc={ntrunc}, filtered')
    if yloc_lam is not None and yobs_lam is not None:
        for ax in axs.ravel():
            ax.plot(yloc_lam,yobs_lam,lw=0.0,marker='x',c='r',zorder=0,label='obs')
    axs[0,0].legend()
    axsd[0,0].legend()
    return fig, axs, figd, axsd

if __name__=="__main__":
    icycle = 50
    pt = 'envar'
    obsloc = ''
    if len(sys.argv)>1:
        icycle=int(sys.argv[1])
    if len(sys.argv)>2:
        pt=sys.argv[2]
    if len(sys.argv)>3:
        obsloc=sys.argv[3]
    
    figdir = Path(f'comp_lsb/{pt}{obsloc}')
    if not figdir.exists():
        figdir.mkdir(parents=True)

    ntrunc_list = [6,12,24]
    u_gm, u_lam = loaddata(icycle)
    if pt=='envar':
        pf_lam = envar.calc_pf(u_lam)
    else:
        pf_lam = var.calc_pf(u_lam,cycle=0)
        u_gm = np.mean(u_gm,axis=1)
        u_lam = np.mean(u_lam,axis=1)
    yloc, yobs, yloc_lam, yobs_lam = loadobs(icycle,obsloc=obsloc)
    u_dscl = H_gm2lam @ u_gm

    fig, axs = plt.subplots(nrows=3,figsize=[6,8],constrained_layout=True)
    if pt=='envar':
        axs[0].plot(ix_gm,u_gm,c='gray',lw=0.5)
        axs[0].plot(ix_gm,u_gm.mean(axis=1),c='b',lw=2.0,label='mean')
    else:
        axs[0].plot(ix_gm,u_gm,c='b',lw=2.0)
    #axs[0].plot(yloc,yobs,lw=0.0,marker='x',c='r',label='obs')
    axs[0].vlines([ix_lam[0],ix_lam[-1]],0,1,colors='k',ls='dotted',zorder=0,transform=axs[0].get_xaxis_transform())
    axs[0].set_title('GM')
    if pt=='envar':
        axs[1].plot(ix_lam,u_dscl,c='gray',lw=0.5)
        axs[1].plot(ix_lam,u_dscl.mean(axis=1),c='b',lw=2.0,label='mean')
    else:
        axs[1].plot(ix_lam,u_dscl,c='b',lw=2.0)
    #axs[1].plot(yloc_lam,yobs_lam,lw=0.0,marker='x',c='r',label='obs')
    axs[1].set_title('GM in the LAM domain')
    if pt=='envar':
        axs[2].plot(ix_lam,u_lam,c='gray',lw=0.5)
        axs[2].plot(ix_lam,u_lam.mean(axis=1),c='b',lw=2.0,label='mean')
    else:
        axs[2].plot(ix_lam,u_lam,c='b',lw=2.0)
    #axs[2].plot(yloc_lam,yobs_lam,lw=0.0,marker='x',c='r',label='obs')
    axs[2].set_title('LAM')
    if pt=='envar':
        axs[0].legend()
    fig.suptitle('background')
    fig.savefig(figdir/f'bg_c{icycle}.png')
    plt.show()
    plt.close()

    if pt=='envar':
        fig, ax = plot_spectra([u_dscl,u_lam],['GM','LAM'],colors=['b','r'])
        fig.suptitle('background ensemble spread')
        fig.savefig(figdir/f'bg_psd_c{icycle}.png')
        plt.show()
        plt.close()

    # conventional DA
    if pt=='envar':
        conv = envar
    else:
        conv = var
    args = (u_lam[1:-1],pf_lam,yobs_lam,yloc_lam)
    u_anl, pa_lam, _, _, _, _ = conv(*args,icycle=icycle)
    ua_lam = u_lam.copy()
    ua_lam[1:-1] = u_anl

    ftrunc_list = []
    u_bld_dict = {}
    ua_bld_dict = {}
    u_bldfil_dict = {}
    ua_bldfil_dict = {}
    ua_nest_dict = {}
    ua_nestfil_dict = {}
    for ntrunc in ntrunc_list:
        trunc_kwargs.update(ntrunc=ntrunc,filter=False,resample=False)
        ## background blending + LAM DA
        truncope = Trunc1d(ix_lam_rad,**trunc_kwargs)
        ftrunc_list.append(truncope.ftrunc)
        udf = truncope(u_dscl - u_lam)
        u_bld = u_lam + udf
        u_bld_dict[ntrunc] = u_bld
        if pt=='envar':
            pf_bld = envar.calc_pf(u_bld)
        else:
            pf_bld = pf_lam
        args = (u_bld[1:-1],pf_bld,yobs_lam,yloc_lam)
        u_anl, pa_bld, _, _, _, _ = conv(*args,icycle=icycle)
        ua_bld = u_bld.copy()
        ua_bld[1:-1] = u_anl
        ua_bld_dict[ntrunc] = ua_bld
        ## Nested DA
        trunc_kwargs.update(resample=True)
        if pt=='envar':
            nest = EnVAR_nest(*envar_nest_initargs,**trunc_kwargs)
        else:
            var_nest_kwargs.update(**trunc_kwargs)
            nest = Var_nest(*var_nest_initargs,**var_nest_kwargs)
        args = (u_lam[1:-1],pf_lam,yobs_lam,yloc_lam,u_gm)
        u_anl, pa_nest, _, _, _, _ = nest(*args,icycle=icycle)
        ua_nest = u_lam.copy()
        ua_nest[1:-1] = u_anl
        ua_nest_dict[ntrunc] = ua_nest

        # low-pass raymond filter
        trunc_kwargs.update(filter=True,resample=False)
        ## background blending + LAM DA
        truncope = Trunc1d(ix_lam_rad,**trunc_kwargs)
        udf = truncope(u_dscl - u_lam)
        u_bld = u_lam + udf
        u_bldfil_dict[ntrunc] = u_bld
        if pt=='envar':
            pf_bld = envar.calc_pf(u_bld)
        else:
            pf_bld = pf_lam
        args = (u_bld[1:-1],pf_bld,yobs_lam,yloc_lam)
        u_anl, pa_bld, _, _, _, _ = conv(*args,icycle=icycle)
        ua_bld = u_bld.copy()
        ua_bld[1:-1] = u_anl
        ua_bldfil_dict[ntrunc] = ua_bld
        ## Nested DA
        trunc_kwargs.update(resample=True)
        if pt=='envar':
            nest = EnVAR_nest(*envar_nest_initargs,**trunc_kwargs)
        else:
            var_nest_kwargs.update(**trunc_kwargs)
            nest = Var_nest(*var_nest_initargs,**var_nest_kwargs)
        args = (u_lam[1:-1],pf_lam,yobs_lam,yloc_lam,u_gm)
        u_anl, pa_nest, _, _, _, _ = nest(*args,icycle=icycle)
        ua_nest = u_lam.copy()
        ua_nest[1:-1] = u_anl
        ua_nestfil_dict[ntrunc] = ua_nest

    # comparison
    colors = ['k','b','b','r','r','g','g']
    styles = ['solid'] + ['solid','dashed']*len(ntrunc_list)
    markers = ['o'] + ['^','x']*len(ntrunc_list)
    ## background blending
    fig, axs, figd, axsd = plot_state(u_lam,u_bld_dict,u_bldfil_dict,ntrunc_list,pt)
    axs[0,0].set_title('LAM (original)')
    fig.suptitle('background blending')
    figd.suptitle('background blending')
    fig.savefig(figdir/f"bg_bld_c{icycle}.png")
    figd.savefig(figdir/f"bg_bld_diff_c{icycle}.png")
    plt.show()
    plt.close(fig=fig)
    plt.close(fig=figd)
    if pt=='envar':
        data = [u_lam]
        labels = ['LAM bg']
        for key in u_bld_dict.keys():
            data.append(u_bld_dict[key])
            labels.append(f'bld,n={key}')
            data.append(u_bldfil_dict[key])
            labels.append(f'bld,n={key},filt')
        fig, ax = plot_spectra(data,labels,colors=colors,styles=styles,markers=markers)
        ax.vlines(ftrunc_list,0,1,colors=['b','r','g'],alpha=0.5,lw=1.0,transform=ax.get_xaxis_transform())
        fig.suptitle('blended ensemble spread')
        fig.savefig(figdir/f'bg_bld_psd_c{icycle}.png')
        plt.show()
        plt.close()

    ## background blending + DA
    fig, axs, figd, axsd = plot_state(ua_lam,ua_bld_dict,ua_bldfil_dict,ntrunc_list,pt,yloc_lam,yobs_lam)
    axs[0,0].set_title('conventional DA')
    fig.suptitle('background blending + DA')
    figd.suptitle('background blending + DA')
    fig.savefig(figdir/f"anl_bld_c{icycle}.png")
    figd.savefig(figdir/f"anl_bld_diff_c{icycle}.png")
    plt.show()
    plt.close(fig=fig)
    plt.close(fig=figd)
    if pt=='envar':
        data = [ua_lam]
        labels = ['DA']
        for key in ua_bld_dict.keys():
            data.append(ua_bld_dict[key])
            labels.append(f'bld,n={key}')
            data.append(ua_bldfil_dict[key])
            labels.append(f'bld,n={key},filt')
        fig, ax = plot_spectra(data,labels,colors=colors,styles=styles,markers=markers)
        ax.vlines(ftrunc_list,0,1,colors=['b','r','g'],alpha=0.5,lw=1.0,transform=ax.get_xaxis_transform())
        fig.suptitle('blended analysis ensemble spread')
        fig.savefig(figdir/f'anl_bld_psd_c{icycle}.png')
        plt.show()
        plt.close()

    ## Nested DA
    fig, axs, figd, axsd = plot_state(ua_lam,ua_nest_dict,ua_nestfil_dict,ntrunc_list,pt,yloc_lam,yobs_lam)
    axs[0,0].set_title('conventional DA')
    fig.suptitle('Nested DA')
    figd.suptitle('Nested DA')
    fig.savefig(figdir/f"anl_nest_c{icycle}.png")
    figd.savefig(figdir/f"anl_nest_diff_c{icycle}.png")
    plt.show()
    plt.close(fig=fig)
    plt.close(fig=figd)
    if pt=='envar':
        data = [ua_lam]
        labels = ['DA']
        for key in ua_nest_dict.keys():
            data.append(ua_nest_dict[key])
            labels.append(f'nest,n={key}')
            data.append(ua_nestfil_dict[key])
            labels.append(f'nest,n={key},filt')
        fig, ax = plot_spectra(data,labels,colors=colors,styles=styles,markers=markers)
        ax.vlines(ftrunc_list,0,1,colors=['b','r','g'],alpha=0.5,lw=1.0,transform=ax.get_xaxis_transform())
        fig.suptitle('Nested analysis ensemble spread')
        fig.savefig(figdir/f'anl_nest_psd_c{icycle}.png')
        plt.show()
        plt.close()
