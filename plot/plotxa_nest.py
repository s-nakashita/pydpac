import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
plt.rcParams['font.size'] = 16
sys.path.append(os.path.join(os.path.dirname(__file__),'../analysis'))
from trunc1d import Trunc1d

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
dscl = False
if len(sys.argv)>4:
    dscl = (sys.argv[4]=='T')
print(dscl)
perts = ["mlef", "mlef_nest", "mlef_nestc", \
    "envar", "envar_nest", "envar_nestc", \
    "etkf", "po", "srf", "letkf", \
    "kf", "var","var_nest",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
cmap = "coolwarm"
f = "truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[:na,]
print(xt.shape)
nx = xt.shape[1]
t = np.arange(na)
ix_t = np.loadtxt("ix_true.txt")
ix_gm = np.loadtxt("ix_gm.txt")
ix_lam = np.loadtxt("ix_lam.txt")
ntrunc = 12
trunc_operator = Trunc1d(ix_lam,ntrunc=ntrunc,ttype='s',cyclic=False) #,resample=False)
ix_trunc = trunc_operator.ix_trunc #[1:-1]
trunc_operator_test = Trunc1d(ix_lam,ntrunc=ntrunc,ttype='c',cyclic=False) #,resample=False)
ttest = trunc_operator_test.tname[trunc_operator_test.ttype]

xt2x = interp1d(ix_t, xt)
xlim = 15.0
for pt in perts:
    #GM
    ## nature and analysis
    if dscl:
        f = "xagmonly_{}_{}.npy".format(op, pt)
        f2 = "xfgmonly_{}_{}.npy".format(op, pt)
    else:
        f = "xagm_{}_{}.npy".format(op, pt)
        f2 = "xfgm_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xagm = np.load(f)
    xfgm = np.load(f2)
    print(xagm.shape)
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=[10,6],\
        constrained_layout=True,sharey=True)
    nx = ix_t.size
    mp0 = axs[0].pcolormesh(ix_t, t, xt, shading='auto',\
        cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[0].set_xticks(ix_t[::(nx//8)])
    axs[0].set_yticks(t[::max(1,na//8)])
    axs[0].set_xlabel("site")
    axs[0].set_ylabel("DA cycle")
    axs[0].set_title("nature")
    p0 = fig.colorbar(mp0,ax=axs[0],orientation="horizontal")
    nx = ix_gm.size
    mp1 = axs[1].pcolormesh(ix_gm, t, xagm, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[1].set_xticks(ix_gm[::(nx//8)])
    axs[1].set_yticks(t[::max(1,na//8)])
    axs[1].set_xlabel("site")
    axs[1].set_title("analysis")
    p1 = fig.colorbar(mp1,ax=axs[1],orientation="horizontal")
    for ax in axs.flatten():
        ax.vlines([ix_lam[0],ix_lam[-1]],0,1,\
            colors='black',linestyle='dashdot',transform=ax.get_xaxis_transform())
    fig.suptitle("nature and analysis in GM : "+pt+" "+op)
    if dscl:
        fig.savefig("{}_xagmonly_{}_{}.png".format(model,op,pt))
    else:
        fig.savefig("{}_xagm_{}_{}.png".format(model,op,pt))
    plt.close()
    ## error and spread
    fig2 = plt.figure(figsize=[10,7],constrained_layout=True)
    gs0 = gridspec.GridSpec(1, 2, figure=fig2)
    gs00 = gs0[0].subgridspec(5, 1)
    ax00 = fig2.add_subplot(gs00[1:, :])
    ax01 = fig2.add_subplot(gs00[0, :])
    xt2gm = xt2x(ix_gm)
    xdgm = xagm - xt2gm
    xdfgm = xfgm - xt2gm
    vlim = np.nanmax(np.abs(xdgm))
    mp2 = ax00.pcolormesh(ix_gm, t, xdgm, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-vlim, vmax=vlim))
    ax00.set_xticks(ix_gm[::(nx//8)])
    ax00.set_yticks(t[::max(1,na//8)])
    ax00.set_xlabel("site")
    p2 = fig2.colorbar(mp2,ax=ax00,orientation="horizontal")
    ax01.plot(ix_gm,np.nanmean(np.abs(xdgm),axis=0))
    ax01.set_xlim(ix_gm[0],ix_gm[-1])
    ax01.set_xticks(ix_gm[::(nx//8)])
    ax01.set_xticklabels([])
    ax01.set_title("error")
    for ax in [ax00,ax01]:
        ax.vlines([ix_lam[0],ix_lam[-1]],0,1,\
            colors='black',linestyle='dashdot',transform=ax.get_xaxis_transform())
    #if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
    if dscl:
        f = "xsagmonly_{}_{}.npy".format(op, pt)
    else:
        f = "xsagm_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xsagm = np.load(f)
    print(xsagm.shape)
    gs01 = gs0[1].subgridspec(5, 1)
    ax10 = fig2.add_subplot(gs01[1:, :])
    ax11 = fig2.add_subplot(gs01[0, :])
    mp3 = ax10.pcolormesh(ix_gm, t, xsagm, shading='auto')
    ax10.set_xticks(ix_gm[::(nx//8)])
    ax10.set_yticks(t[::max(1,na//8)])
    ax10.set_xlabel("site")
    p3 = fig2.colorbar(mp3,ax=ax10,orientation="horizontal")
    ax11.plot(ix_gm,np.nanmean(xsagm,axis=0))
    ax11.set_xlim(ix_gm[0],ix_gm[-1])
    ax11.set_xticks(ix_gm[::(nx//8)])
    ax11.set_xticklabels([])
    for ax in [ax10,ax11]:
        ax.vlines([ix_lam[0],ix_lam[-1]],0,1,\
        colors='black',linestyle='dashdot',transform=ax.get_xaxis_transform())
    if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
        ax11.set_title("spread")
        fig2.suptitle("error and spread in GM : "+pt+" "+op)
    else:
        ax11.set_title("analysis error standard deviation")
        fig2.suptitle("error and stdv in GM : "+pt+" "+op)
    if dscl:
        fig2.savefig("{}_xdgmonly_{}_{}.png".format(model,op,pt))
    else:
        fig2.savefig("{}_xdgm_{}_{}.png".format(model,op,pt))
    plt.close()
    #LAM
    ## nature and analysis
    if dscl:
        f = "xadscl_{}_{}.npy".format(op, pt)
        f2 = "xfdscl_{}_{}.npy".format(op, pt)
    else:
        f = "xalam_{}_{}.npy".format(op, pt)
        f2 = "xflam_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xalam = np.load(f)
    xflam = np.load(f2)
    print(xalam.shape)
    fig, axs = plt.subplots(nrows=1,ncols=2,figsize=[10,6],\
        constrained_layout=True,sharey=True)
    nx = ix_t.size
    mp0 = axs[0].pcolormesh(ix_t, t, xt, shading='auto',\
        cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[0].set_yticks(t[::max(1,na//8)])
    axs[0].set_xlabel("site")
    axs[0].set_ylabel("DA cycle")
    axs[0].set_title("nature")
    p0 = fig.colorbar(mp0,ax=axs[0],orientation="horizontal")
    nx = ix_lam.size
    axs[0].set_xticks(ix_lam[::(nx//8)])
    mp1 = axs[1].pcolormesh(ix_lam, t, xalam, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    p1 = fig.colorbar(mp1,ax=axs[1],orientation="horizontal")
    axs[1].set_xticks(ix_lam[::(nx//8)])
    axs[1].set_yticks(t[::max(1,na//8)])
    axs[1].set_xlabel("site")
    axs[1].set_title("analysis")
    for ax in axs.flatten():
        ax.set_xlim(ix_lam[0],ix_lam[-1])
    fig.suptitle("nature and analysis in LAM : "+pt+" "+op)
    if dscl:
        fig.savefig("{}_xadscl_{}_{}.png".format(model,op,pt))
    else:
        fig.savefig("{}_xalam_{}_{}.png".format(model,op,pt))
    plt.close()
    ## error and spread
    fig2 = plt.figure(figsize=[10,7],constrained_layout=True)
    gs0 = gridspec.GridSpec(1, 2, figure=fig2)
    gs00 = gs0[0].subgridspec(5, 1)
    ax00 = fig2.add_subplot(gs00[1:, :])
    ax01 = fig2.add_subplot(gs00[0, :])
    xt2lam = xt2x(ix_lam)
    xdlam = xalam - xt2lam
    xdflam = xflam - xt2lam
    vlim = np.nanmax(np.abs(xdlam))
    mp2 = ax00.pcolormesh(ix_lam, t, xdlam, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-vlim, vmax=vlim))
    ax00.set_xticks(ix_lam[::(nx//8)])
    ax00.set_yticks(t[::max(1,na//8)])
    ax00.set_xlabel("site")
    p2 = fig2.colorbar(mp2,ax=ax00,orientation="horizontal")
    ax01.plot(ix_lam,np.nanmean(np.abs(xdlam),axis=0))
    ax01.set_xticks(ix_lam[::(nx//8)])
    ax01.set_xticklabels([])
    ax01.set_title("error")
    for ax in [ax00,ax01]:
        ax.set_xlim(ix_lam[0],ix_lam[-1])
    if pt == "var" or pt == "var_nest" or pt == "4dvar":
        ix = ix_lam[1:-1]
    else:
        ix = ix_lam.copy()
    if dscl:
        f = "xsadscl_{}_{}.npy".format(op, pt)
    else:
        f = "xsalam_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xsalam = np.load(f)
    print(xsalam.shape)
    gs01 = gs0[1].subgridspec(5, 1)
    ax10 = fig2.add_subplot(gs01[1:, :])
    ax11 = fig2.add_subplot(gs01[0, :])
    mp3 = ax10.pcolormesh(ix, t, xsalam, shading='auto')
    ax10.set_xticks(ix[::(ix.size//8)])
    ax10.set_yticks(t[::max(1,na//8)])
    ax10.set_xlabel("site")
    p3 = fig2.colorbar(mp3,ax=ax10,orientation="horizontal")
    ax11.plot(ix,np.nanmean(xsalam,axis=0))
    ax11.set_xticks(ix[::(ix.size//8)])
    ax11.set_xticklabels([])
    for ax in [ax10,ax11]:
        ax.set_xlim(ix_lam[0],ix_lam[-1])
    if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
        ax11.set_title("spread")
        fig2.suptitle("error and spread in LAM : "+pt+" "+op)
    else:
        ax11.set_title("analysis error standard deviation")
        fig2.suptitle("error and stdv in LAM : "+pt+" "+op)
    if dscl:
        fig2.savefig("{}_xddscl_{}_{}.png".format(model,op,pt))
    else:
        fig2.savefig("{}_xdlam_{}_{}.png".format(model,op,pt))
    plt.close()
    ## first 10 cycles
    i0 = np.argmin(np.abs(ix_gm - ix_lam[0]))
    if ix_gm[i0]<ix_lam[0]: i0+=1
    i1 = np.argmin(np.abs(ix_gm - ix_lam[-1]))
    if ix_gm[i1]>ix_lam[-1]: i1-=1
    tmp_lam2gm = interp1d(ix_lam,np.eye(ix_lam.size),axis=0)
    JH2 = tmp_lam2gm(ix_gm[i0:i1+1])
    for icycle in range(40,50):
        xt1 = xt[icycle]
        xagm1 = xagm[icycle]
        xalam1 = xalam[icycle]
        fig, ax = plt.subplots(figsize=[6,4])
        ax.plot(ix_t,xt1,label='nature')
        ax.plot(ix_gm,xagm1,label='GM')
        ax.plot(ix_lam,xalam1,label='LAM')
        f = "data/{2}/{0}_lam_dk_{1}_{2}_cycle{3}.npy".format(model,op,pt,icycle)
        if os.path.isfile(f):
            dk = np.load(f)
            ax.plot(ix_trunc,dk,label='dk')
        #else:
        #    ax.plot(ix_gm[i0:i1+1],xagm1[i0:i1+1]-JH2@xalam1,label='dk')
        ax.set_title(f't={t[icycle]}, analysis')
        ax.legend()
        fig.savefig("data/{2}/{0}_xa_{1}_{2}_c{3}.png".format(model,op,pt,icycle))
        plt.close()
        #
        xfgm1 = xfgm[icycle]
        xflam1 = xflam[icycle]
        fig, ax = plt.subplots(figsize=[6,4])
        ax.plot(ix_t,xt1,label='nature')
        ax.plot(ix_gm,xfgm1,label='GM')
        ax.plot(ix_lam,xflam1,label='LAM')
        f = "data/{2}/{0}_lam_dk_{1}_{2}_cycle{3}.npy".format(model,op,pt,icycle)
        if os.path.isfile(f):
            dk = np.load(f)
            ax.plot(ix_trunc,dk,label='dk')
        #else:
        #    ax.plot(ix_gm[i0:i1+1],xagm1[i0:i1+1]-JH2@xalam1,label='dk')
        ax.set_title(f't={t[icycle]}, forecast')
        ax.legend()
        fig.savefig("data/{2}/{0}_xf_{1}_{2}_c{3}.png".format(model,op,pt,icycle))
        plt.close()
    ## error cross-covariance evaluation
    xdv = trunc_operator_test(xdfgm.transpose()).transpose()
    print(xdv.shape)
    ns = 40
    vlim = max(np.max(xdflam[ns:,:]),-np.min(xdflam[ns:,:]),\
        np.max(xdv[ns:,:]),-np.min(xdv[ns:,:]))
    fig, axs = plt.subplots(ncols=2,figsize=[8,6],constrained_layout=True)
    mp0 = axs[0].pcolormesh(ix_lam,t,xdflam, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-vlim, vmax=vlim))
    nx=ix_lam.size
    axs[0].set_xticks(ix_lam[::(nx//6)])
    axs[0].set_yticks(t[::max(1,na//8)])
    axs[0].set_title(r"$\varepsilon^\mathrm{b}$")
    mp1 = axs[1].pcolormesh(ix_trunc,t,xdv, shading='auto', \
    cmap=cmap, norm=Normalize(vmin=-vlim, vmax=vlim))
    nx=ix_trunc.size
    axs[1].set_xticks(ix_trunc[::(nx//5)])
    axs[1].set_yticks(t[::max(1,na//8)])
    axs[1].set_title(r"$\varepsilon^\mathrm{v}$")
    fig.colorbar(mp1,ax=axs[1],shrink=0.6,pad=0.01)
    fig.suptitle(pt)
    fig.savefig("{}_xdcomp_{}_{}.png".format(model,op,pt))
    B = np.dot(xdflam[ns:,:].transpose(),xdflam[ns:,:])/float(na-ns)
    V = np.dot(xdv[ns:,:].transpose(),xdv[ns:,:])/float(na-ns)
    Ebv = np.dot(xdflam[ns:,:].transpose(),xdv[ns:,:])/float(na-ns)
    Evb = np.dot(xdv[ns:,:].transpose(),xdflam[ns:,:])/float(na-ns)
    W = np.hstack((np.vstack((B,Evb)),np.vstack((Ebv,V))))
    fig2, axs2 = plt.subplots(ncols=2,figsize=[10,6],constrained_layout=True)
    vlim = np.mean(np.diag(B))
    mp0 = axs2[0].matshow(W,\
        cmap='bwr', norm=Normalize(vmin=-vlim, vmax=vlim))
    fig2.colorbar(mp0,ax=axs2[0],shrink=0.6,pad=0.01)
    axs2[0].hlines([ix_lam.size],0,1,colors='k',ls='dashed',transform=axs2[0].get_yaxis_transform())
    axs2[0].vlines([ix_lam.size],0,1,colors='k',ls='dashed',transform=axs2[0].get_xaxis_transform())
    axs2[0].set_title(r'$\mathbf{W}$')
    lam, c = np.linalg.eigh(W)
    axs2[1].semilogy(lam)
    axs2[1].set_ylabel('eigenvalues')
    axs2[1].set_title(f'cond(W)={np.linalg.cond(W):.4e}')
    fig2.suptitle(pt)
    fig2.savefig("{}_errcov_{}_{}_{}.png".format(model,op,pt,ttest))
    plt.show()
    plt.close()