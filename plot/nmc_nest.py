import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
dscl = False
if len(sys.argv)>4:
    dscl = (sys.argv[4]=='T')
print(dscl)
perts = ["mlef", "mlefw", "etkf", "po", "srf", "letkf", "kf", "var","var_nest",\
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
xt2x = interp1d(ix_t, xt)
xlim = 15.0
ns = na//10
ne = na
print(f"na={na} ns={ns} ne={ne}")
for pt in perts:
    #GM
    # 6-hr forecast
    if dscl:
        f = "{}_xfgmonly_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xfgm_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x6hgm = np.load(f)
    print(x6hgm.shape)
    nx = ix_gm.size
    # 12-hr forecast
    if dscl:
        f = "{}_xf12gmonly_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xf12gm_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x12hgm = np.load(f)
    print(x12hgm.shape)
    nx = ix_gm.size
    # 24-hr forecast
    if dscl:
        f = "{}_xf24gmonly_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xf24gm_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x24hgm = np.load(f)
    print(x24hgm.shape)
    nx = ix_gm.size
    # 48-hr forecast
    if dscl:
        f = "{}_xf48gmonly_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xf48gm_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x48hgm = np.load(f)
    print(x48hgm.shape)
    nx = ix_gm.size
    ## 12h - 6h
    x12m6_gm = x12hgm[ns:ne] - x6hgm[ns:ne]
    #x12m6_gm = x12m6_gm - x12m6_gm.mean(axis=1)[:,None]
    print(x12m6_gm.shape)
    B12m6_gm = np.dot(x12m6_gm.T,x12m6_gm)/float(ne-ns+1)*0.5
    ## 24h - 12h
    x24m12_gm = x24hgm[ns:ne] - x12hgm[ns:ne]
    #x24m12_gm = x24m12_gm - x24m12_gm.mean(axis=1)[:,None]
    print(x24m12_gm.shape)
    B24m12_gm = np.dot(x24m12_gm.T,x24m12_gm)/float(ne-ns+1)*0.5
    ## 48h - 24h
    x48m24_gm = x48hgm[ns:ne] - x24hgm[ns:ne]
    #x48m24_gm = x48m24_gm - x48m24_gm.mean(axis=1)[:,None]
    print(x48m24_gm.shape)
    B48m24_gm = np.dot(x48m24_gm.T,x48m24_gm)/float(ne-ns+1)*0.5
    ## plot
    fig, axs = plt.subplots(nrows=1,ncols=4,figsize=[12,6],constrained_layout=True,sharey=True)
    mp0 = axs[0].pcolormesh(ix_gm, t[ns:ne], x6hgm[ns:ne], shading='auto',\
        cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[0].set_xticks(ix_gm[::(nx//8)])
    axs[0].set_yticks(t[ns:ne:(na//8)])
    axs[0].set_xlabel("site")
    axs[0].set_ylabel("DA cycle")
    axs[0].set_title("6h")
    p0 = fig.colorbar(mp0,ax=axs[0],orientation="horizontal")
    vlim = max(np.max(x12m6_gm),-np.min(x12m6_gm))
    mp1 = axs[1].pcolormesh(ix_gm, t[ns:ne], x12m6_gm, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[1].set_xticks(ix_gm[::(nx//8)])
    axs[1].set_yticks(t[ns:ne:(na//8)])
    axs[1].set_xlabel("site")
    axs[1].set_title("12h - 6h")
    p1 = fig.colorbar(mp1,ax=axs[1],orientation="horizontal")
    vlim = max(np.max(x24m12_gm),-np.min(x24m12_gm))
    mp2 = axs[2].pcolormesh(ix_gm, t[ns:ne], x24m12_gm, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[2].set_xticks(ix_gm[::(nx//8)])
    axs[2].set_yticks(t[ns:ne:(na//8)])
    axs[2].set_xlabel("site")
    axs[2].set_title("24h - 12h")
    p2 = fig.colorbar(mp2,ax=axs[2],orientation="horizontal")
    vlim = max(np.max(x48m24_gm),-np.min(x48m24_gm))
    mp3 = axs[3].pcolormesh(ix_gm, t[ns:ne], x48m24_gm, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[3].set_xticks(ix_gm[::(nx//8)])
    axs[3].set_yticks(t[ns:ne:(na//8)])
    axs[3].set_xlabel("site")
    axs[3].set_title("48h - 24h")
    p3 = fig.colorbar(mp3,ax=axs[3],orientation="horizontal")
    fig.suptitle("forecast in GM : "+pt+" "+op)
    fig.savefig("{}_xfgm_{}_{}.png".format(model,op,pt))
    plt.show()
    plt.close()
    fig = plt.figure(figsize=[12,10],constrained_layout=True)
    gs = gridspec.GridSpec(3,1,figure=fig)
    gs0 = gs[:2].subgridspec(1,3)
    ax00 = fig.add_subplot(gs0[:,0])
    ax01 = fig.add_subplot(gs0[:,1])
    ax02 = fig.add_subplot(gs0[:,2])
    mp0 = ax00.pcolormesh(ix_gm, ix_gm, B12m6_gm, shading='auto')
    ax00.set_xticks(ix_gm[::(nx//8)])
    ax00.set_yticks(ix_gm[::(nx//8)])
    ax00.set_title("12h - 6h")
    ax00.set_aspect("equal")
    p0 = fig.colorbar(mp0,ax=ax00,shrink=0.6,pad=0.01) #,orientation="horizontal")
    mp1 = ax01.pcolormesh(ix_gm, ix_gm, B24m12_gm, shading='auto')
    ax01.set_xticks(ix_gm[::(nx//8)])
    ax01.set_yticks(ix_gm[::(nx//8)])
    ax01.set_title("24h - 12h")
    ax01.set_aspect("equal")
    p1 = fig.colorbar(mp1,ax=ax01,shrink=0.6,pad=0.01) #,orientation="horizontal")
    mp2 = ax02.pcolormesh(ix_gm, ix_gm, B48m24_gm, shading='auto')
    ax02.set_xticks(ix_gm[::(nx//8)])
    ax02.set_yticks(ix_gm[::(nx//8)])
    ax02.set_title("48h - 24h")
    ax02.set_aspect("equal")
    p2 = fig.colorbar(mp2,ax=ax02,shrink=0.6,pad=0.01)
    gs1 = gs[2].subgridspec(1,2)
    ax10 = fig.add_subplot(gs1[:,0])
    ax11 = fig.add_subplot(gs1[:,1])
    ### diagonal
    ax10.plot(ix_gm,np.diag(B12m6_gm),label="12h - 6h")
    ax10.plot(ix_gm,np.diag(B24m12_gm),label="24h - 12h")
    ax10.plot(ix_gm,np.diag(B48m24_gm),label="48h - 24h")
    ax10.set_xticks(ix_gm[::(nx//8)])
    ax10.set_title("Diagonal")
    ax10.legend()
    ### row
    ax11.plot(ix_gm,B12m6_gm[nx//2,:],label="12h - 6h")
    ax11.plot(ix_gm,B24m12_gm[nx//2,:],label="24h - 12h")
    ax11.plot(ix_gm,B48m24_gm[nx//2,:],label="48h - 24h")
    ax11.set_xticks(ix_gm[::(nx//8)])
    ax11.set_title("Row")
    ax11.legend()
    fig.suptitle("NMC in GM : "+pt+" "+op)
    if dscl:
        fig.savefig("{}_nmcgmonly_{}_{}.png".format(model,op,pt))
    else:
        fig.savefig("{}_nmcgm_{}_{}.png".format(model,op,pt))
    plt.show()
    plt.close()
    np.save("{}_B12m6_gm.npy".format(model),B12m6_gm)
    np.save("{}_B24m12_gm.npy".format(model),B24m12_gm)
    np.save("{}_B48m24_gm.npy".format(model),B48m24_gm)
    #exit()
    #LAM
    # 6-hr forecast
    if dscl:
        f = "{}_xfdscl_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xflam_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x6hlam = np.load(f)
    print(x6hlam.shape)
    nx = ix_lam.size
    # 12-hr forecast
    if dscl:
        f = "{}_xf12dscl_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xf12lam_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x12hlam = np.load(f)
    print(x12hlam.shape)
    nx = ix_lam.size
    # 24-hr forecast
    if dscl:
        f = "{}_xf24dscl_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xf24lam_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x24hlam = np.load(f)
    print(x24hlam.shape)
    # 48-hr forecast
    if dscl:
        f = "{}_xf48dscl_{}_{}.npy".format(model, op, pt)
    else:
        f = "{}_xf48lam_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    x48hlam = np.load(f)
    print(x48hlam.shape)
    nx = ix_lam.size
    ## 12h - 6h
    x12m6_lam = x12hlam[ns:ne] - x6hlam[ns:ne]
    #x12m6_lam = x12m6_lam - x12m6_lam.mean(axis=1)[:,None]
    print(x12m6_lam.shape)
    B12m6_lam = np.dot(x12m6_lam.T,x12m6_lam)/float(ne-ns+1)*0.5
    ## 24h - 12h
    x24m12_lam = x24hlam[ns:ne] - x12hlam[ns:ne]
    #x24m12_lam = x24m12_lam - x24m12_lam.mean(axis=1)[:,None]
    print(x24m12_lam.shape)
    B24m12_lam = np.dot(x24m12_lam.T,x24m12_lam)/float(ne-ns+1)*0.5
    ## 48h - 24h
    x48m24_lam = x48hlam[ns:ne] - x24hlam[ns:ne]
    #x48m24_lam = x48m24_lam - x48m24_lam.mean(axis=1)[:,None]
    print(x48m24_lam.shape)
    B48m24_lam = np.dot(x48m24_lam.T,x48m24_lam)/float(ne-ns+1)*0.5
    ## plot
    fig, axs = plt.subplots(nrows=1,ncols=4,figsize=[12,6],constrained_layout=True,sharey=True)
    mp0 = axs[0].pcolormesh(ix_lam, t[ns:ne], x6hlam[ns:ne], shading='auto',\
        cmap=cmap, norm=Normalize(vmin=-xlim, vmax=xlim))
    axs[0].set_xticks(ix_lam[::(nx//8)])
    axs[0].set_yticks(t[ns:ne:(na//8)])
    axs[0].set_xlabel("site")
    axs[0].set_ylabel("DA cycle")
    axs[0].set_title("6h")
    p0 = fig.colorbar(mp0,ax=axs[0],orientation="horizontal")
    vlim = max(np.max(x12m6_lam),-np.min(x12m6_lam))
    mp1 = axs[1].pcolormesh(ix_lam, t[ns:ne], x12m6_lam, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[1].set_xticks(ix_lam[::(nx//8)])
    axs[1].set_yticks(t[ns:ne:(na//8)])
    axs[1].set_xlabel("site")
    axs[1].set_title("12h - 6h")
    p1 = fig.colorbar(mp1,ax=axs[1],orientation="horizontal")
    vlim = max(np.max(x24m12_lam),-np.min(x24m12_lam))
    mp2 = axs[2].pcolormesh(ix_lam, t[ns:ne], x24m12_lam, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[2].set_xticks(ix_lam[::(nx//8)])
    axs[2].set_yticks(t[ns:ne:(na//8)])
    axs[2].set_xlabel("site")
    axs[2].set_title("24h - 12h")
    p2 = fig.colorbar(mp2,ax=axs[2],orientation="horizontal")
    vlim = max(np.max(x48m24_lam),-np.min(x48m24_lam))
    mp3 = axs[3].pcolormesh(ix_lam, t[ns:ne], x48m24_lam, shading='auto',\
        cmap="PiYG", norm=Normalize(vmin=-vlim, vmax=vlim))
    axs[3].set_xticks(ix_lam[::(nx//8)])
    axs[3].set_yticks(t[ns:ne:(na//8)])
    axs[3].set_xlabel("site")
    axs[3].set_title("48h - 24h")
    p3 = fig.colorbar(mp3,ax=axs[3],orientation="horizontal")
    fig.suptitle("forecast in LAM : "+pt+" "+op)
    fig.savefig("{}_xflam_{}_{}.png".format(model,op,pt))
    #plt.show()
    plt.close()
    fig = plt.figure(figsize=[12,10],constrained_layout=True)
    gs = gridspec.GridSpec(3,1,figure=fig)
    gs0 = gs[:2].subgridspec(1,3)
    ax00 = fig.add_subplot(gs0[:,0])
    ax01 = fig.add_subplot(gs0[:,1])
    ax02 = fig.add_subplot(gs0[:,2])
    mp0 = ax00.pcolormesh(ix_lam, ix_lam, B12m6_lam, shading='auto')
    ax00.set_xticks(ix_lam[::(nx//8)])
    ax00.set_yticks(ix_lam[::(nx//8)])
    ax00.set_title("12h - 6h")
    ax00.set_aspect("equal")
    p0 = fig.colorbar(mp0,ax=ax00,shrink=0.6,pad=0.01) #,orientation="horizontal")
    mp1 = ax01.pcolormesh(ix_lam, ix_lam, B24m12_lam, shading='auto')
    ax01.set_xticks(ix_lam[::(nx//8)])
    ax01.set_yticks(ix_lam[::(nx//8)])
    ax01.set_title("24h - 12h")
    ax01.set_aspect("equal")
    p1 = fig.colorbar(mp1,ax=ax01,shrink=0.6,pad=0.01) #,orientation="horizontal")
    mp2 = ax02.pcolormesh(ix_lam, ix_lam, B48m24_lam, shading='auto')
    ax02.set_xticks(ix_lam[::(nx//8)])
    ax02.set_yticks(ix_lam[::(nx//8)])
    ax02.set_title("48h - 24h")
    ax02.set_aspect("equal")
    p2 = fig.colorbar(mp2,ax=ax02,shrink=0.6,pad=0.01)
    gs1 = gs[2].subgridspec(1,2)
    ax10 = fig.add_subplot(gs1[:,0])
    ax11 = fig.add_subplot(gs1[:,1])
    ### diagonal
    ax10.plot(ix_lam,np.diag(B12m6_lam),label="12h - 6h")
    ax10.plot(ix_lam,np.diag(B24m12_lam),label="24h - 12h")
    ax10.plot(ix_lam,np.diag(B48m24_lam),label="48h - 24h")
    ax10.set_xticks(ix_lam[::(nx//8)])
    ax10.set_title("Diagonal")
    ax10.legend()
    ### row
    ax11.plot(ix_lam,B12m6_lam[nx//2,:],label="12h - 6h")
    ax11.plot(ix_lam,B24m12_lam[nx//2,:],label="24h - 12h")
    ax11.plot(ix_lam,B48m24_lam[nx//2,:],label="48h - 24h")
    ax11.set_xticks(ix_lam[::(nx//8)])
    ax11.set_title("Row")
    ax11.legend()
    fig.suptitle("NMC in LAM : "+pt+" "+op)
    if dscl:
        fig.savefig("{}_nmcdscl_{}_{}.png".format(model,op,pt))
    else:
        fig.savefig("{}_nmclam_{}_{}.png".format(model,op,pt))
    plt.show()
    plt.close()
    np.save("{}_B12m6_lam.npy".format(model),B12m6_lam)
    np.save("{}_B24m12_lam.npy".format(model),B24m12_lam)
    np.save("{}_B48m24_lam.npy".format(model),B48m24_lam)
    #GM-LAM
    i0 = np.argmin(np.abs(ix_gm - ix_lam[0]))
    i1 = np.argmin(np.abs(ix_gm - ix_lam[-1]))
    n = ix_gm[i0:i1+1].size
    #gm2lam = interp1d(ix_gm,x12m6_gm,axis=1)
    V12m6 = np.dot(x12m6_gm[:,i0:i1+1].T,x12m6_gm[:,i0:i1+1])/float(ne-ns+1)*0.5
    B12m6_gm2lam = np.dot(x12m6_gm[:,i0:i1+1].T,x12m6_lam)/float(ne-ns+1)*0.5
    #gm2lam = interp1d(ix_gm,x24m12_gm,axis=1)
    V24m12 = np.dot(x24m12_gm[:,i0:i1+1].T,x24m12_gm[:,i0:i1+1])/float(ne-ns+1)*0.5
    B24m12_gm2lam = np.dot(x24m12_gm[:,i0:i1+1].T,x24m12_lam)/float(ne-ns+1)*0.5
    V48m24 = np.dot(x48m24_gm[:,i0:i1+1].T,x48m24_gm[:,i0:i1+1])/float(ne-ns+1)*0.5
    B48m24_gm2lam = np.dot(x48m24_gm[:,i0:i1+1].T,x48m24_lam)/float(ne-ns+1)*0.5
    ## plot
    fig = plt.figure(figsize=[12,10],constrained_layout=True)
    gs = gridspec.GridSpec(3,1,figure=fig)
    gs0 = gs[:2].subgridspec(1,3)
    ax00 = fig.add_subplot(gs0[:,0])
    ax01 = fig.add_subplot(gs0[:,1])
    ax02 = fig.add_subplot(gs0[:,2])
    mp0 = ax00.pcolormesh(ix_gm[i0:i1+1], ix_gm[i0:i1+1], V12m6, shading='auto')
    ax00.set_xticks(ix_gm[i0:i1+1:(n//8)])
    ax00.set_yticks(ix_gm[i0:i1+1:(n//8)])
    ax00.set_title("12h - 6h")
    ax00.set_aspect("equal")
    p0 = fig.colorbar(mp0,ax=ax00,shrink=0.6,pad=0.01) #,orientation="horizontal")
    mp1 = ax01.pcolormesh(ix_gm[i0:i1+1], ix_gm[i0:i1+1], V24m12, shading='auto')
    ax01.set_xticks(ix_gm[i0:i1+1:(n//8)])
    ax01.set_yticks(ix_gm[i0:i1+1:(n//8)])
    ax01.set_title("24h - 12h")
    ax01.set_aspect("equal")
    p1 = fig.colorbar(mp1,ax=ax01,shrink=0.6,pad=0.01) #,orientation="horizontal")
    mp2 = ax02.pcolormesh(ix_gm[i0:i1+1], ix_gm[i0:i1+1], V48m24, shading='auto')
    ax02.set_xticks(ix_gm[i0:i1+1:(n//8)])
    ax02.set_yticks(ix_gm[i0:i1+1:(n//8)])
    ax02.set_title("48h - 24h")
    ax02.set_aspect("equal")
    p2 = fig.colorbar(mp2,ax=ax02,shrink=0.6,pad=0.01) #,orientation="horizontal")
    gs1 = gs[2].subgridspec(1,2)
    ax10 = fig.add_subplot(gs1[:,0])
    ax11 = fig.add_subplot(gs1[:,1])
    ### diagonal
    ax10.plot(ix_gm[i0:i1+1],np.diag(V12m6),label="12h - 6h")
    ax10.plot(ix_gm[i0:i1+1],np.diag(V24m12),label="24h - 12h")
    ax10.plot(ix_gm[i0:i1+1],np.diag(V48m24),label="48h - 24h")
    ax10.set_xticks(ix_gm[i0:i1+1:(n//8)])
    ax10.set_title("Diagonal")
    ax10.legend()
    ### row
    ax11.plot(ix_gm[i0:i1+1],V12m6[n//2,:],label="12h - 6h")
    ax11.plot(ix_gm[i0:i1+1],V24m12[n//2,:],label="24h - 12h")
    ax11.plot(ix_gm[i0:i1+1],V48m24[n//2,:],label="48h - 24h")
    ax11.set_xticks(ix_gm[i0:i1+1:(n//8)])
    ax11.set_title("Row")
    ax11.legend()
    fig.suptitle("NMC in GM within LAM : "+pt+" "+op)
    fig.savefig("{}_nmcv_{}_{}.png".format(model,op,pt))
    plt.show()
    plt.close()
    np.save("{}_V12m6.npy".format(model),V12m6)
    np.save("{}_V24m12.npy".format(model),V24m12)
    np.save("{}_V48m24.npy".format(model),V48m24)
    ## plot
    fig, axs = plt.subplots(nrows=2,ncols=2,figsize=[10,10],\
        constrained_layout=True)
    mp0 = axs[0,0].pcolormesh(ix_lam, ix_gm[i0:i1+1], B12m6_gm2lam, shading='auto')
    axs[0,0].set_xticks(ix_lam[::(nx//8)])
    axs[0,0].set_yticks(ix_gm[i0:i1+1:(n//8)])
    axs[0,0].set_title("12h - 6h")
    axs[0,0].set_aspect(n/nx)
    p0 = fig.colorbar(mp0,ax=axs[0,0],orientation="horizontal")
    mp1 = axs[0,1].pcolormesh(ix_lam, ix_gm[i0:i1+1], B24m12_gm2lam, shading='auto')
    axs[0,1].set_xticks(ix_lam[::(nx//8)])
    axs[0,1].set_yticks(ix_gm[i0:i1+1:(n//8)])
    axs[0,1].set_title("24h - 12h")
    axs[0,1].set_aspect(n/nx)
    p1 = fig.colorbar(mp1,ax=axs[0,1],orientation="horizontal")
    mp2 = axs[1,0].pcolormesh(ix_lam, ix_gm[i0:i1+1], B48m24_gm2lam, shading='auto')
    axs[1,0].set_xticks(ix_lam[::(nx//8)])
    axs[1,0].set_yticks(ix_gm[i0:i1+1:(n//8)])
    axs[1,0].set_title("48h - 24h")
    axs[1,0].set_aspect(n/nx)
    p2 = fig.colorbar(mp2,ax=axs[1,0],orientation="horizontal")
    #### diagonal
    #axs[1,0].plot(ix_gm[i0:i1+1],np.diag(B12m6_gm2lam),label="12h - 6h")
    #axs[1,0].plot(ix_gm[i0:i1+1],np.diag(B24m12_gm2lam),label="24h - 12h")
    #axs[1,0].set_xticks(ix_gm[i0:i1+1:(n//8)])
    #axs[1,0].set_title("Diagonal")
    #axs[1,0].legend()
    ### row
    axs[1,1].plot(ix_lam,B12m6_gm2lam[n//2,:],label="12h - 6h")
    axs[1,1].plot(ix_lam,B24m12_gm2lam[n//2,:],label="24h - 12h")
    axs[1,1].plot(ix_lam,B48m24_gm2lam[n//2,:],label="48h - 24h")
    axs[1,1].set_xticks(ix_lam[::(nx//8)])
    axs[1,1].set_title("Row")
    axs[1,1].legend()
    fig.suptitle("NMC in GM x LAM : "+pt+" "+op)
    fig.savefig("{}_nmcgm2lam_{}_{}.png".format(model,op,pt))
    plt.show()
    plt.close()
    np.save("{}_B12m6_gm2lam.npy".format(model),B12m6_gm2lam)
    np.save("{}_B24m12_gm2lam.npy".format(model),B24m12_gm2lam)
    np.save("{}_B48m24_gm2lam.npy".format(model),B48m24_gm2lam)
    