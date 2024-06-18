import sys
import os
import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.colors import Normalize
from nmc_tools import NMC_tools, wnum2wlen, wlen2wnum
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__),'../analysis'))
from trunc1d import Trunc1d

plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 24

op = sys.argv[1]
model = sys.argv[2]
#na = int(sys.argv[3])

#t = np.arange(na)+1
ns = 40 # spinup

datadir = Path(f'work/{model}')
#datadir = Path(f'/Volumes/FF520/nested_envar/data/{model}')
#preGMpt = 'envar'
#dscldir = datadir / 'var_vs_envar_dscl_m80obs30'
#lamdir  = datadir / 'var_vs_envar_preGM_m80obs30'
lamdir  = datadir / 'envar_nestc_shrink_preGM_m80obs30'

perts = ["envar_nest","envar_nestc","mlef_nest","mlef_nestc"]
labels = {"envar":"EnVar", "envar_nest":"Nested EnVar", "envar_nestc":"Nested EnVar_c", "var":"3DVar", "var_nest":"Nested 3DVar"}
linecolor = {"envar":'tab:orange',"envar_nest":'tab:green',"envar_nestc":'lime',"var":"tab:olive","var_nest":"tab:brown"}

ix_t = np.loadtxt(lamdir/"ix_true.txt")
ix_gm = np.loadtxt(lamdir/"ix_gm.txt")
ix_lam = np.loadtxt(lamdir/"ix_lam.txt")[1:-1]
nx_t = ix_t.size
nx_gm = ix_gm.size
nx_lam = ix_lam.size
xlim = 15.0
nghost = 0 # ghost region for periodicity in LAM
ix_t_rad = ix_t * 2.0 * np.pi / nx_t
ix_gm_rad = ix_gm * 2.0 * np.pi / nx_t
ix_lam_rad = ix_lam * 2.0 * np.pi / nx_t
Lx_gm = 2.0 * np.pi
#dwindow = (1.0 + np.cos(np.pi*np.arange(1,nghost+1)/nghost))*0.5
Lx_lam = 2.0 * np.pi * nx_lam / nx_t

nmc = NMC_tools(ix_lam_rad,cyclic=False)

ntrunc = 12
trunc_operator = Trunc1d(ix_lam,ntrunc=ntrunc,ttype='s',cyclic=False,nghost=0)
ix_trunc = trunc_operator.ix_trunc
nx_gmlam = ix_trunc.size
ix_trunc_rad = ix_trunc * 2.0 * np.pi / nx_t
nmc_trunc = NMC_tools(ix_trunc_rad,cyclic=False)

#figsp, axsp = plt.subplots(figsize=[10,8],constrained_layout=True)
#psd_dict = {}

#pt="envar_nestc"
scycle = 40 #40
ecycle = 60 #1000
for pt in perts:
    ncycle_lam = 0
    vmat_exist = False
    for icycle in range(scycle,ecycle+1):
        f = lamdir/"{}_lam_spf_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
        if f.exists():
            spftmp = np.load(f)
            pftmp = spftmp @ spftmp.T
            if ncycle_lam==0:
                pflam = pftmp
                #wnum_lam, psdlam, _ = nmc.psd(spftmp,axis=0,average=True)
            else:
                pflam = pflam + pftmp
                #wnum_lam, psdtmp, _ = nmc.psd(spftmp,axis=0,average=True)
                #psdlam = psdlam + psdtmp
            ncycle_lam += 1
        else:
            f = lamdir/"data/{2}/{0}_lam_spf_{1}_{2}_cycle{3}.npy".format(model, op, pt, icycle)
            if f.exists():
                spftmp = np.load(f)
                pftmp = spftmp @ spftmp.T
                if ncycle_lam==0:
                    pflam = pftmp
                    #wnum_lam, psdlam, _ = nmc.psd(spftmp,axis=0,average=True)
                else:
                    pflam = pflam + pftmp
                    #wnum_lam, psdtmp, _ = nmc.psd(spftmp,axis=0,average=True)
                    #psdlam = psdlam + psdtmp
                ncycle_lam += 1
        f = lamdir/"{}_lam_svmat_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
        if f.exists():
            svtmp = np.load(f)
            vtmp = svtmp @ svtmp.T
            if not vmat_exist:
                vmat = vtmp
                #wnum_v, psdv, _ = nmc_trunc.psd(svtmp,axis=0,average=True)
            else:
                vmat = vmat + vtmp
                #wnum_v, psdtmp, _ = nmc_trunc.psd(svtmp,axis=0,average=True)
                #psdv = psdv + psdtmp
            vmat_exist=True
        else:
            f = lamdir/"data/{2}/{0}_lam_svmat_{1}_{2}_cycle{3}.npy".format(model, op, pt, icycle)
            if f.exists():
                svtmp = np.load(f)
                vtmp = svtmp @ svtmp.T
                if not vmat_exist:
                    vmat = vtmp
                    #wnum_v, psdv, _ = nmc_trunc.psd(svtmp,axis=0,average=True)
                else:
                    vmat = vmat + vtmp
                    #wnum_v, psdtmp, _ = nmc_trunc.psd(svtmp,axis=0,average=True)
                    #psdv = psdv + psdtmp
                vmat_exist=True
        if vmat_exist:
            ctmp = spftmp @ svtmp.T
            if ncycle_lam == 1:
                cmat = ctmp
            else:
                cmat = cmat + ctmp
        
            eb, sb, cbt = np.linalg.svd(spftmp,full_matrices=False)
            #ndofb = int(np.sum(sb>1.0e-10))
            sbfull = sb.copy()
            lamb = sb*sb
            lamsum = np.sum(lamb)
            ndofb = 0
            contrib=0.0
            while ndofb <= sb.size:
                contrib+=lamb[ndofb]/lamsum
                ndofb += 1
                if contrib > 0.99: break
            eb = eb[:,:ndofb]
            sb = sb[:ndofb]
            cbt = cbt[:ndofb,:]
            cb = cbt.transpose()
            ev, sv, cvt = np.linalg.svd(svtmp,full_matrices=False)
            #ndofv = int(np.sum(sv>1.0e-10))
            svfull = sv.copy()
            lamv = sv*sv
            lamsum = np.sum(lamv)
            ndofv = 0
            contrib=0.0
            while ndofv <= sv.size:
                contrib+=lamv[ndofv]/lamsum
                ndofv += 1
                if contrib > 0.99: break
            ev = ev[:,:ndofv]
            sv = sv[:ndofv]
            cvt = cvt[:ndofv,:]
            cv = cvt.transpose()
            fig, axs = plt.subplots(figsize=[10,8],nrows=2,ncols=3,constrained_layout=True)
            vlim=0.75
            axs[0,0].set_title(r'$\mathbf{E}_\mathrm{b}$'+f'={eb.shape}',fontsize=16)
            mp00 = axs[0,0].matshow(eb,cmap='bwr',norm=Normalize(-vlim,vlim))
            fig.colorbar(mp00,ax=axs[0,0],shrink=0.6,pad=0.01)
            axs[1,0].set_title(r'$\mathbf{C}_\mathrm{b}$'+f'={cb.shape}',fontsize=16)
            mp10 = axs[1,0].matshow(cb,cmap='bwr',norm=Normalize(-vlim,vlim))
            fig.colorbar(mp10,ax=axs[1,0],shrink=0.6,pad=0.01)
            axs[0,1].set_title(r'$\mathbf{E}_\mathrm{v}$'+f'={ev.shape}',fontsize=16)
            mp01 = axs[0,1].matshow(ev,cmap='bwr',norm=Normalize(-vlim,vlim))
            fig.colorbar(mp01,ax=axs[0,1],shrink=0.6,pad=0.01)
            axs[1,1].set_title(r'$\mathbf{C}_\mathrm{v}$'+f'={cv.shape}',fontsize=16)
            mp11 = axs[1,1].matshow(cv,cmap='bwr',norm=Normalize(-vlim,vlim))
            fig.colorbar(mp11,ax=axs[1,1],shrink=0.6,pad=0.01)
            axs[0,2].plot(np.arange(1,sbfull.size+1),sbfull,label=r'$\sigma_\mathrm{b}$')
            axs[0,2].plot(np.arange(1,svfull.size+1),svfull,label=r'$\sigma_\mathrm{v}$')
            axs[0,2].vlines([ndofb,ndofv],0,1,colors=['tab:blue','tab:orange'],ls='dashed',transform=axs[0,2].get_xaxis_transform())
            axs[0,2].legend()
            axs[1,2].remove()
            fig.suptitle(f"cycle={icycle}")
            fig.savefig(lamdir/"{}_spfsvd_{}_{}_cycle{}.png".format(model,op,pt,icycle))
            plt.show(block=False)
            plt.close()

            # Schur complement of Pf
            fig, axs = plt.subplots(figsize=[8,8],nrows=2,ncols=2,constrained_layout=True)
            Q = svtmp @ (np.eye(cvt.shape[1])-cb@cbt) @ svtmp.T
            ubn = eb @ np.diag(1.0/sb)
            uvn = ev @ np.diag(1.0/sv)
            tmp = np.eye(sv.size) - cvt@cb@cbt@cv
            Qpinv = uvn @ np.linalg.inv(tmp) @ uvn.transpose()
            axs[0,0].set_title(r'$\mathbf{C}_\mathrm{b}\mathbf{C}_\mathrm{b}^\mathrm{T}$',fontsize=16)
            mp00 = axs[0,0].matshow(cb@cbt,cmap='bwr',norm=Normalize(-vlim,vlim))
            fig.colorbar(mp00,ax=axs[0,0],shrink=0.6,pad=0.01)
            axs[0,1].set_title(r'$\mathbf{C}_\mathrm{v}^\mathrm{T}\mathbf{C}_\mathrm{b}$',fontsize=16)
            mp01 = axs[0,1].matshow(cvt@cb,cmap='bwr',norm=Normalize(-vlim,vlim))
            fig.colorbar(mp01,ax=axs[0,1],shrink=0.6,pad=0.01)
            vlim = max(np.max(Q),-np.min(Q))
            axs[1,0].set_title(r'$\mathbf{E}_\mathrm{v}\mathbf{\Sigma}_\mathrm{v}\mathbf{C}_\mathrm{v}^\mathrm{T}(\mathbf{I}-\mathbf{C}_\mathrm{b}\mathbf{C}_\mathrm{b}^\mathrm{T})\mathbf{C}_\mathrm{v}\mathbf{\Sigma}_\mathrm{v}\mathbf{E}_\mathrm{v}^\mathrm{T}$',fontsize=12)
            mp10 = axs[1,0].matshow(Q,cmap='bwr',norm=Normalize(-vlim,vlim))
            fig.colorbar(mp10,ax=axs[1,0],shrink=0.6,pad=0.01)
            vlim = max(np.max(Qpinv),-np.min(Qpinv))
            axs[1,1].set_title(r'$\mathbf{E}_\mathrm{v}\mathbf{\Sigma}_\mathrm{v}^{-1}\mathbf{C}_\mathrm{v}^\mathrm{T}(\mathbf{I}-\mathbf{C}_\mathrm{b}\mathbf{C}_\mathrm{b}^\mathrm{T})^{-1}\mathbf{C}_\mathrm{v}\mathbf{\Sigma}_\mathrm{v}^{-1}\mathbf{E}_\mathrm{v}^\mathrm{T}$',fontsize=12)
            mp11 = axs[1,1].matshow(Qpinv,cmap='bwr',norm=Normalize(-vlim,vlim))
            fig.colorbar(mp11,ax=axs[1,1],shrink=0.6,pad=0.01)
            fig.suptitle(f"cycle={icycle}")
            fig.savefig(lamdir/"{}_schur_{}_{}_cycle{}.png".format(model,op,pt,icycle))
            plt.show(block=False)
            plt.close()

            Pc = np.hstack((np.vstack((pftmp,ctmp.T)),np.vstack((ctmp,vtmp))))
            # MP-inverse of Pc
            Pci00 = ubn @ (np.eye(sb.size) + cbt@cv@np.linalg.inv(tmp)@cvt@cb) @ ubn.transpose()
            Pci01 = -1.0 * ubn @ cbt@cv@np.linalg.inv(tmp) @ uvn.transpose()
            Pci10 = Pci01.transpose()
            Pci11 = Qpinv
            Pci = np.hstack((np.vstack((Pci00,Pci10)),np.vstack((Pci01,Pci11))))
            fig, axs = plt.subplots(figsize=[10,10],nrows=2,ncols=2,constrained_layout=True)
            axs[0,0].set_title(r"$\mathbf{P}_\mathrm{c}$")
            vlim = max(np.max(Pc),-np.min(Pc))
            mp00 = axs[0,0].matshow(Pc,cmap='coolwarm',norm=Normalize(-vlim,vlim))
            fig.colorbar(mp00,ax=axs[0,0],shrink=0.6,pad=0.01)
            axs[0,1].set_title(r"$\mathbf{P}_\mathrm{c}^\dagger$")
            vlim = max(np.max(Pci),-np.min(Pci))
            mp01 = axs[0,1].matshow(Pci,cmap='coolwarm',norm=Normalize(-vlim,vlim))
            fig.colorbar(mp01,ax=axs[0,1],shrink=0.6,pad=0.01)
            axs[1,0].set_title(r"$\mathbf{P}_\mathrm{c}\mathbf{P}_\mathrm{c}^\dagger\mathbf{P}_\mathrm{c}$")
            mp10 = axs[1,0].matshow(Pc@Pci@Pc,cmap='PiYG',norm=Normalize(vmin=-1.0,vmax=1.0))
            fig.colorbar(mp10,ax=axs[1,0],shrink=0.6,pad=0.01)
            Pcpi = np.linalg.pinv(Pc)
            axs[1,1].set_title("np.linalg.pinv")
            vlim = max(np.max(Pcpi),-np.min(Pcpi))
            mp11=axs[1,1].matshow(Pcpi,cmap='coolwarm',norm=Normalize(-vlim,vlim))
            fig.colorbar(mp11,ax=axs[1,1],shrink=0.6,pad=0.01)
            fig.suptitle(f"cycle={icycle}")
            fig.savefig(lamdir/"{}_pfc_{}_{}_cycle{}.png".format(model,op,pt,icycle))
            plt.show(block=False)
            plt.close()

        qmat_exist = False
        hess_exist = False
        f = lamdir/"{}_lam_qmat_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
        if f.exists():
            qmat = np.load(f)
            q2mat = qmat @ qmat.T
            f = lamdir/"{}_lam_dk2_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
            dk2 = np.load(f)
            qmat_exist=True
        else:
            f = lamdir/"data/{2}/{0}_lam_qmat_{1}_{2}_cycle{3}.npy".format(model, op, pt, icycle)
            if f.exists():
                qmat = np.load(f)
                q2mat = qmat.T @ qmat
                f = lamdir/"data/{2}/{0}_lam_dk2_{1}_{2}_cycle{3}.npy".format(model, op, pt, icycle)
                dk2 = np.load(f)
                qmat_exist=True
        if qmat_exist:
            fig, axs = plt.subplots(nrows=2,ncols=2,figsize=[12,12],constrained_layout=True)
            xaxis = np.arange(qmat.shape[0])
            vlim = max(np.max(qmat),-np.min(qmat))
            mp0 = axs[0,0].matshow(qmat,\
                cmap='coolwarm',norm=Normalize(-vlim,vlim))
            fig.colorbar(mp0,ax=axs[0,0],pad=0.01,shrink=0.6)
            if pt=="envar_nest" or pt=="mlef_nest":
                axs[0,0].set_title(r'$\mathbf{Q}=(\mathbf{Z}^\mathrm{v})^{\dagger}\mathbf{Z}^\mathrm{b}$')
            elif pt=="envar_nestc" or pt=="mlef_nestc":
                axs[0,0].set_title(r'$\mathbf{Q}^\prime='
                r'(\mathbf{X}^\mathrm{b} , '
                r'\mathbf{Z}^\mathrm{v}'
                r')^{\dagger}'
                r'(\mathbf{X}^\mathrm{b} , '
                r'\mathbf{Z}^\mathrm{b}'
                r')$')
            if pt=="envar_nest" or pt=="mlef_nest":
                q2mat = q2mat + np.diag(np.ones(q2mat.shape[0]))
                vlim = max(np.max(q2mat),-np.min(q2mat))
                mp01 = axs[0,1].matshow(q2mat,\
                cmap='coolwarm',norm=Normalize(-vlim,vlim))
                axs[0,1].set_title(r'$\mathbf{I}+\mathbf{Q}^\mathrm{T}\mathbf{Q}$')
            elif pt=="envar_nestc" or pt=="mlef_nestc":
                vlim = max(np.max(q2mat),-np.min(q2mat))
                mp01 = axs[0,1].matshow(q2mat,\
                cmap='coolwarm',norm=Normalize(-vlim,vlim))
                axs[0,1].set_title(r'$\mathbf{Q}^{\prime\mathrm{T}}\mathbf{Q}^\prime$')
            fig.colorbar(mp01,ax=axs[0,1],pad=0.01,shrink=0.6)
            q2inv = np.linalg.pinv(q2mat)
            vlim = max(np.max(q2inv),-np.min(q2inv))
            mp10=axs[1,0].matshow(q2inv,\
                cmap='coolwarm',norm=Normalize(-vlim,vlim))
            fig.colorbar(mp10,ax=axs[1,0],pad=0.01,shrink=0.6)
            if pt=="envar_nest" or pt=="mlef_nest":
                axs[1,0].set_title(r'$[\mathbf{I}+\mathbf{Q}^\mathrm{T}\mathbf{Q}]^{-1}$')
            elif pt=="envar_nestc" or pt=="mlef_nestc":
                axs[1,0].set_title(r'$[\mathbf{Q}^{\prime\mathrm{T}}\mathbf{Q}^\prime]^{\dagger}$')
            for ax in axs.flatten()[:-1]:
                ax.set_aspect("equal")

            axs[1,1].plot(dk2,xaxis)
            if pt=="envar_nest" or pt=="mlef_nest":
                axs[1,1].set_title(r'$\tilde{\mathbf{d}}^\mathrm{v}=(\mathbf{Z}^\mathrm{v})^{\dagger}\mathbf{d}^\mathrm{v}$')
            elif pt=="envar_nestc" or pt=="mlef_nestc":
                axs[1,1].set_title(r'$\tilde{\mathbf{d}}^\mathrm{v}=(\mathbf{X}^\mathrm{b} , '
                r'\mathbf{Z}^\mathrm{v}'
                r')^{\dagger}(\mathbf{0} , \mathbf{d}^\mathrm{v})$')
            fig.suptitle(f"cycle={icycle}")
            fig.savefig(lamdir/"{}_qmat_{}_{}_cycle{}.png".format(model,op,pt,icycle))
            #fig.savefig(lamdir/"{}_qmat_{}_{}_cycle{}.pdf".format(model,op,pt,icycle))
            plt.show(block=False)
            plt.close()

        # hessian
        f = lamdir/"{}_lam_hess_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
        if f.exists():
            hess = np.load(f)
            f = lamdir/"{}_lam_tmat_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
            tmat = np.load(f)
            f = lamdir/"{}_lam_heinv_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
            heinv = np.load(f)
            hess_exist = True
        else:
            f = lamdir/"data/{2}/{0}_lam_hess_{1}_{2}_cycle{3}.npy".format(model, op, pt, icycle)
            if f.exists():
                hess = np.load(f)
                f = lamdir/"data/{2}/{0}_lam_tmat_{1}_{2}_cycle{3}.npy".format(model, op, pt, icycle)
                tmat = np.load(f)
                f = lamdir/"data/{2}/{0}_lam_heinv_{1}_{2}_cycle{3}.npy".format(model, op, pt, icycle)
                heinv = np.load(f)
                hess_exist = True
        if hess_exist:
            lam, c = np.linalg.eigh(hess)
            cond = np.linalg.cond(hess)
            fig = plt.figure(figsize=[12,10],constrained_layout=True)
            gs = gridspec.GridSpec(nrows=2,ncols=3,figure=fig)
            ax00 = fig.add_subplot(gs[0,0])
            ax01 = fig.add_subplot(gs[0,1])
            ax10 = fig.add_subplot(gs[1,0])
            ax11 = fig.add_subplot(gs[1,1])
            ax2  = fig.add_subplot(gs[0,2])
            m00=ax00.matshow(hess)
            fig.colorbar(m00,ax=ax00,pad=0.01,shrink=0.6)
            if pt=="envar_nest":
                ax00.set_title(r'$\mathbf{H}=[(K-1)(\mathbf{I}+\mathbf{Q}^\mathrm{T}\mathbf{Q})+\mathbf{Z}^\mathrm{T}\mathbf{Z}]$'\
                    +'\n'+r'$=\mathbf{C}\mathbf{\Lambda}\mathbf{C}^\mathrm{T}$',fontsize=14)
            elif pt=="mlef_nest":
                ax00.set_title(r'$\mathbf{H}=[\mathbf{I}+\mathbf{Q}^\mathrm{T}\mathbf{Q}+\mathbf{Z}^\mathrm{T}\mathbf{Z}]$'\
                    +'\n'+r'$=\mathbf{C}\mathbf{\Lambda}\mathbf{C}^\mathrm{T}$',fontsize=14)
            elif pt=="envar_nestc":
                ax00.set_title(r'$\mathbf{H}=[(K-1)\mathbf{Q}^{\prime\mathrm{T}}\mathbf{Q}^\prime+\mathbf{Z}^\mathrm{T}\mathbf{Z}]$'\
                    +'\n'+r'$=\mathbf{C}\mathbf{\Lambda}\mathbf{C}^\mathrm{T}$',fontsize=16)
            elif pt=="mlef_nestc":
                ax00.set_title(r'$\mathbf{H}=[\mathbf{Q}^{\prime\mathrm{T}}\mathbf{Q}^\prime+\mathbf{Z}^\mathrm{T}\mathbf{Z}]$'\
                    +'\n'+r'$=\mathbf{C}\mathbf{\Lambda}\mathbf{C}^\mathrm{T}$',fontsize=16)
            #ax00.set_title('Hessian'+r'$=\mathbf{V}\mathbf{\Lambda}\mathbf{V}^\mathrm{T}$')
            m01=ax01.matshow(c)
            fig.colorbar(m01,ax=ax01,pad=0.01,shrink=0.6)
            ax01.set_ylabel('member index')
            ax01.set_xlabel('eigenmode')
            ax01.set_title(r'$\mathbf{C}$')
            ax2.plot(lam)
            ax2.set_title(r'$\lambda$'+f' cond={cond:.4e}',fontsize=18)
            ax2.set_xlabel('eigenmode')
            m10=ax10.matshow(tmat)
            fig.colorbar(m10,ax=ax10,pad=0.01,shrink=0.6)
            ax10.set_title(r'$\mathbf{H}^{-1/2}=\mathbf{C}\mathbf{\Lambda}^{-1/2}\mathbf{C}^\mathrm{T}$')
            m11=ax11.matshow(heinv)
            fig.colorbar(m11,ax=ax11,pad=0.01,shrink=0.6)
            ax11.set_title(r'$\mathbf{H}^{-1}$')
            if pt=="envar_nestc" or pt=="mlef_nestc":
                ax12 = fig.add_subplot(gs[1,2])
                nmem = c.shape[0]
                if pt=="envar_nestc":
                    ndof = np.sum((lam-float(nmem-1))>-1.0e-5)
                else:
                    ndof = np.sum((lam-1.0)>-1.0e-5)
                nnul = lam.size - ndof
                #tmat2 = np.eye(hess.shape[0]) - np.dot(c[:,nnul:],np.dot(np.diag(np.ones(ndof)-1.0/np.sqrt(lam[nnul:])),c[:,nnul:].transpose()))
                #m12 = ax12.matshow(tmat2)
                #fig.colorbar(m12,ax=ax12,pad=0.01,shrink=0.6)
                #ax12.set_title(r'$\mathbf{I}-\mathbf{C}[\mathbf{I}-\mathbf{\Lambda}^{-1/2}]\mathbf{C}^\mathrm{T}$',fontsize=16)

                sigma = np.sqrt(lam[nnul:])
                modlam = (sigma - np.ones(ndof))/(sigma*sigma*sigma)
                tmat3 = np.dot(c[:,nnul:],np.dot(np.diag(modlam),c[:,nnul:].transpose()))
                tmat3 = np.dot(tmat3,hess)
                m12 = ax12.matshow(np.eye(tmat3.shape[0])-tmat3)
                fig.colorbar(m12,ax=ax12,pad=0.01,shrink=0.6)
                if pt=="envar_nestc":
                    ax12.set_title(r'$\mathbf{I}-\mathbf{C}[\mathbf{I}-\mathbf{\Lambda}^{-1/2}]\mathbf{\Lambda}^{-1}\mathbf{C}^\mathrm{T}[(K-1)\mathbf{Q}^{\prime\mathrm{T}}\mathbf{Q}^\prime+\mathbf{Z}^\mathrm{T}\mathbf{Z}]$',fontsize=14)
                else:
                    ax12.set_title(r'$\mathbf{I}-\mathbf{C}[\mathbf{I}-\mathbf{\Lambda}^{-1/2}]\mathbf{\Lambda}^{-1}\mathbf{C}^\mathrm{T}[\mathbf{Q}^{\prime\mathrm{T}}\mathbf{Q}^\prime+\mathbf{Z}^\mathrm{T}\mathbf{Z}]$',fontsize=14)
                #fig3, ax3 = plt.subplots(figsize=[6,6])
                #m3=ax3.matshow(tmat3)
                #fig3.colorbar(m3,ax=ax3,pad=0.01,shrink=0.6)
                #ax3.set_title(r'$\mathbf{C}[\mathbf{I}-\mathbf{\Lambda}^{-1/2}]\mathbf{\Lambda}^{-1}\mathbf{C}^\mathrm{T}[(K-1)\mathbf{Q}^{\prime\mathrm{T}}\mathbf{Q}^\prime+\mathbf{Z}^\mathrm{T}\mathbf{Z}]$',fontsize=16)
                #fig3.savefig(lamdir/"{}_tmatmod_{}_{}_cycle{}.png".format(model,op,pt,icycle))
                #plt.show(block=False)
                #plt.close(figure=fig3)
            fig.suptitle(f"cycle={icycle}")
            fig.savefig(lamdir/"{}_hess_{}_{}_cycle{}.png".format(model,op,pt,icycle))
            plt.show(block=False)
            plt.close()
            
        if pt=="envar_nestc" or pt=="mlef_nestc":
            dxc_exist = False
            f = lamdir/"{}_lam_dxc1_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
            if f.exists():
                dxc1 = np.load(f)
                f = lamdir/"{}_lam_dxc2_{}_{}_cycle{}.npy".format(model, op, pt, icycle)
                dxc2 = np.load(f)
                dxc_exist = True
            else:
                f = lamdir/"data/{2}/{0}_lam_dxc1_{1}_{2}_cycle{3}.npy".format(model, op, pt, icycle)
                if f.exists():
                    dxc1 = np.load(f)
                    f = lamdir/"data/{2}/{0}_lam_dxc2_{1}_{2}_cycle{3}.npy".format(model, op, pt, icycle)
                    dxc2 = np.load(f)
                    dxc_exist = True
            if dxc_exist:
                fig, axs = plt.subplots(ncols=2,figsize=[12,12],constrained_layout=True)
                m0 = axs[0].matshow(dxc1)
                fig.colorbar(m0,ax=axs[0],pad=0.01,shrink=0.6)
                axs[0].set_title(r"$\mathbf{X}^\mathrm{b}, \mathbf{Z}^\mathrm{v}$")
                m1 = axs[1].matshow(dxc2)
                fig.colorbar(m1,ax=axs[1],pad=0.01,shrink=0.6)
                axs[1].set_title(r"$\mathbf{X}^\mathrm{b}, \mathbf{Z}^\mathrm{b}$")
                for ax in axs:
                    ax.hlines([ix_lam.size-1],0,1,colors='r',ls='dashed',transform=ax.get_yaxis_transform())
                fig.suptitle(f"cycle={icycle}")
                fig.savefig(lamdir/"{}_dxc_{}_{}_cycle{}.png".format(model,op,pt,icycle))
                plt.show(block=False)
                plt.close()

    if ncycle_lam == 0: continue
    pflam = pflam / float(ncycle_lam)
    if vmat_exist:
        vmat = vmat / float(ncycle_lam)
        cmat = cmat / float(ncycle_lam)
    fig, axs = plt.subplots(nrows=2,ncols=2,figsize=[12,12],constrained_layout=True)
    #vlim = 0.15
    vlim = max(np.max(pflam),-np.min(pflam))
    mp00 = axs[0,0].pcolormesh(ix_lam,ix_lam,pflam,shading='auto',\
        cmap='coolwarm',norm=Normalize(-vlim,vlim))
    fig.colorbar(mp00,ax=axs[0,0],pad=0.01,shrink=0.6)
    axs[0,0].set_title(r'$\mathbf{P}^\mathrm{b}$')
    if vmat_exist:
        vlim = max(np.max(vmat),-np.min(vmat))
        mp11 = axs[1,1].pcolormesh(ix_trunc,ix_trunc,vmat,shading='auto',\
            cmap='coolwarm',norm=Normalize(-vlim,vlim))
        fig.colorbar(mp11,ax=axs[1,1],pad=0.01,shrink=0.6)
        axs[1,1].set_title(r'$\mathbf{P}^\mathrm{v}$')
        vlim = max(np.max(cmat),-np.min(cmat))
        mp01 = axs[0,1].pcolormesh(ix_trunc,ix_lam,cmat,shading='auto',\
            cmap='coolwarm',norm=Normalize(-vlim,vlim))
        fig.colorbar(mp01,ax=axs[0,1],pad=0.01,shrink=0.6)
        axs[0,1].set_title(r'$\mathbf{X}^\mathrm{b}(\mathbf{Z}^\mathrm{v})^\mathrm{T}$')
        mp10 = axs[1,0].pcolormesh(ix_lam,ix_trunc,cmat.T,shading='auto',\
            cmap='coolwarm',norm=Normalize(-vlim,vlim))
        fig.colorbar(mp10,ax=axs[1,0],pad=0.01,shrink=0.6)
        axs[1,0].set_title(r'$\mathbf{Z}^\mathrm{v}(\mathbf{X}^\mathrm{b})^\mathrm{T}$')
    for ax in axs.flatten():
        ax.set_ylim(ix_lam[-1],ix_lam[0])
        ticks = ix_lam[::(nx_lam)//8]
        ax.set_aspect("equal")
        ax.set_xticks(ticks)
        ax.set_yticks(ticks[::-1])

    #fig.suptitle(f"{labels[pt]}, cycle={scycle}-{ecycle}")
    fig.savefig(lamdir/"{}_pf_{}_{}_cycle{}-{}.png".format(model,op,pt,scycle,ecycle))
    fig.savefig(lamdir/"{}_pf_{}_{}_cycle{}-{}.pdf".format(model,op,pt,scycle,ecycle))
    plt.show()
    plt.close()
