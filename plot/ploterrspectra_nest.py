import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter
from nmc_tools import psd, wnum2wlen, wlen2wnum
sys.path.append(os.path.join(os.path.dirname(__file__),'../analysis'))
from trunc1d import Trunc1d
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
detrend = False
if len(sys.argv)>4:
    detrend = sys.argv[4]=='T'
perts = ["mlef", "envar", "envar_nest", "envar_nestc", \
    "etkf", "po", "srf", "letkf", "kf", "var", "var_nest",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
linecolor = {"mlef":'tab:blue',"envar":'tab:orange',"envar_nest":'tab:green',"envar_nestc":"lime",\
    "etkf":'tab:green', "po":'tab:red',"srf":"tab:pink", "letkf":"tab:purple",\
    "kf":"tab:cyan", "var":"tab:olive","var_nest":"tab:brown",\
    "mlefcw":"tab:green","mlefy":"tab:orange","mlefbe":"tab:red","mlefbm":"tab:pink"}
if model == "z08":
    perts = ["mlef", "grad", "etkf-fh", "etkf-jh"]#, "po", "srf", "letkf"]
    linestyle = {"mlef":"solid", "grad":"dashed",
     "etkf-fh":"solid", "etkf-jh":"dashed"}
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf-fh":'tab:green',"etkf-jh":'tab:red'}
cmap = "coolwarm"
f = "truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[:na,]
print(xt.shape)
nx_t = xt.shape[1]
t = np.arange(na)
ix_t = np.loadtxt("ix_true.txt")
ix_gm = np.loadtxt("ix_gm.txt")
ix_lam = np.loadtxt("ix_lam.txt")
nx_t = ix_t.size
nx_gm = ix_gm.size
nx_lam = ix_lam.size
xt2x = interp1d(ix_t,xt)
xlim = 15.0
nghost = 0 # ghost region for periodicity in LAM
ix_t_rad = ix_t * 2.0 * np.pi / nx_t
ix_gm_rad = ix_gm * 2.0 * np.pi / nx_t
ix_lam_rad = ix_lam * 2.0 * np.pi / nx_t
Lx_gm = 2.0 * np.pi
#dwindow = (1.0 + np.cos(np.pi*np.arange(1,nghost+1)/nghost))*0.5
Lx_lam = 2.0 * np.pi * nx_lam / nx_t
#Lx_lam = 2.0 * np.pi * (nx_lam + 2*nghost) / nx_t
#dx_gm = Lx_gm / nx_gm
#dx_lam = Lx_lam / (nx_lam + 2*nghost)
figall = plt.figure(figsize=[10,10],constrained_layout=True)
axgm = figall.add_subplot(211)
axlam = figall.add_subplot(212,sharex=axgm)
psdgm_dict = {}
psdlam_dict = {}
for pt in perts:
    #GM
    f = "xagm_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xagm = np.load(f)
    print(f"xagm.shape={xagm.shape}")
    f = "xsagm_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xsagm = np.load(f)
    print(f"xsagm.shape={xsagm.shape}")
    if np.isnan(xagm).any():
        print("divergence in {}".format(pt))
        continue
    #LAM
    f = "xalam_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xalam = np.load(f)
    print(f"xalam.shape={xalam.shape}")
    f = "xsalam_{}_{}.npy".format(op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xsalam = np.load(f)
    print(f"xsalam.shape={xsalam.shape}")
    if np.isnan(xalam).any():
        print("divergence in {}".format(pt))
        continue
    fig, axs = plt.subplots(nrows=2,figsize=[10,10],constrained_layout=True)
    xt2xg = interp1d(ix_t,xt)
    xdgm = xagm - xt2xg(ix_gm)
    axs[0].plot(ix_gm,np.sqrt(np.mean(xdgm**2,axis=0)),c='tab:blue',label='GM,err')
    #if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
    #    axs[0].plot(ix_gm,xsagm.mean(axis=0),c='tab:blue',ls='dashed',label='GM,sprd')
    i0 = np.argmin(np.abs(ix_t-ix_lam[0]))
    i1 = np.argmin(np.abs(ix_t-ix_lam[-1]))
    xdlam = xalam - xt[:,i0:i1+1]
    ##xd = xdgm.copy()
    ##xd[:,i0:i1+1] = xdlam
    #xd = np.zeros((xdlam.shape[0],nx_lam+2*nghost))
    #xd[:,nghost:nghost+nx_lam] = xdlam[:,:]
    #xd[:,0:nghost] = xdlam[:,0].reshape(-1,1) * dwindow[None,::-1]
    #xd[:,nghost+nx_lam:] = xdlam[:,-1].reshape(-1,1) * dwindow[None,:]
    #xsa = np.zeros((xsalam.shape[0],nx_lam+2*nghost))
    #xsa[:,nghost:nghost+nx_lam] = xsalam[:,:]
    #xsa[:,0:nghost] = xsalam[:,0].reshape(-1,1) * dwindow[None,::-1]
    #xsa[:,nghost+nx_lam:] = xsalam[:,-1].reshape(-1,1) * dwindow[None,:]
    #ix_lam_ext = ix_t[i0-nghost:i0+nx_lam+nghost]
    axs[0].plot(ix_lam,np.sqrt(np.mean(xdlam**2,axis=0)),c='tab:orange',label='LAM,err')
    #if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
    #    axs[0].plot(ix_lam,xsalam.mean(axis=0),c='tab:orange',ls='dashed',label='LAM,sprd')
    axs[0].vlines([ix_lam[0],ix_lam[-1]],0,1,colors='k',ls='dashdot',transform=axs[0].get_xaxis_transform())
    axs[0].set_xlim(ix_t[0],ix_t[-1])
    axs[0].set_xlabel("grid index")
    #if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
    #    axs[0].set_ylabel("error or spread")
    #else:
    axs[0].set_ylabel("RMSE")
    axs[0].set_title("state space")
    axs[0].grid()
    lines = []
    labels = []
    #espgm = fft(xdgm,axis=1)[:,:nx_gm//2]
    #wnum = fftfreq(nx_gm,dx_gm)[:nx_gm//2]
    ##freq = np.arange(0,nx//2)
    #psdgm = 2.0*np.mean(np.abs(espgm)**2,axis=0)*dx_gm*dx_gm/Lx_gm
    wnum_gm, psdgm = psd(xdgm,ix_gm_rad,axis=1,average=False)
    psdgm_dict[pt] = psdgm
    psdgm = psdgm.mean(axis=0)
    print(f"psdgm.shape={psdgm.shape}")
    print(f"wnum_gm={wnum_gm}")
    axs[1].semilogy(wnum_gm,psdgm,c='tab:blue',marker='x')
    lines.append(Line2D([0],[0],color='tab:blue',lw=2))
    labels.append('GM,err')
    axgm.semilogy(wnum_gm,psdgm,c=linecolor[pt],marker='x',label=pt)
    #nx = xsagm.shape[1]
    #dx = 2.0 * np.pi / nx
    #freq = fftfreq(nx,dx)[:nx//2]
    #if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
    #    espgms = fft(xsagm,axis=1)
    #    psdgm = 2.0*np.mean(np.abs(espgms[:,:nx_gm//2])**2,axis=0)*dx_gm*dx_gm/Lx_gm
    #    axs[1].plot(freq,psdgm,c='tab:blue',ls='dashed',label='GM,sprd')
    #nx = xd.shape[1]
    #esp = fft(xd,axis=1)[:,:nx_lam//2]
    #wnum = fftfreq(nx,dx_lam)[:nx_lam//2]
    ##freq = np.arange(0,nx//2)
    #psd = 2.0*np.mean(np.abs(esp)**2,axis=0)*dx_lam*dx_lam/Lx_lam
    if detrend:
        wnum_lam, psdlam, xdlam_detrend = psd(xdlam,ix_lam_rad,axis=1,cyclic=False,nghost=0,average=False,detrend=detrend)
        axs[0].plot(ix_lam,np.sqrt(np.mean(xdlam_detrend**2,axis=0)),c='tab:orange',ls='dashed',label='LAM,err,detrend')
    else:
        wnum_lam, psdlam = psd(xdlam,ix_lam_rad,axis=1,cyclic=False,nghost=0,average=False,detrend=detrend)
    axs[0].legend()
    psdlam_dict[pt] = psdlam
    psdlam = psdlam.mean(axis=0)
    print(f"psdlam.shape={psdlam.shape}")
    print(f"wnum_lam={wnum_lam}")
    axs[1].semilogy(wnum_lam,psdlam,c='tab:orange',marker='x')
    lines.append(Line2D([0],[0],color='tab:orange',lw=2))
    labels.append('LAM,err')
    axlam.semilogy(wnum_lam,psdlam,c=linecolor[pt],marker='x',label=pt)
    #ax2 = axs[1].twinx()
    #espdiff = esp - espgm
    #ax2.plot(freq,espdiff,c='red')
    #lines.append(Line2D([0],[0],color='red',lw=2))
    #labels.append('(GM+LAM)-GM')
    #ax2.hlines([0],0,1,colors='orange',transform=ax2.get_yaxis_transform(),zorder=0)
    #ax2.tick_params(axis='y',labelcolor='red')
    #if pt != "kf" and pt != "var" and pt != "var_nest" and pt != "4dvar":
    #    esps = fft(xsa,axis=1)
    #    psd = 2.0*np.mean(np.abs(esps[:,:nx_lam//2])**2,axis=0)*dx_lam*dx_lam/Lx_lam
    #    axs[1].plot(freq,psd,c='tab:orange',ls='dashed',label='LAM,sprd')
#    axs[1].set_xlim(wnum[1],wnum[-1])
    axs[1].set_xlabel(r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)")
    axs[1].set_ylabel("power spectral density")
    #axs[1].set_title("spectral space")
    axs[1].legend(lines,labels)
    #axs[1].xaxis.set_major_locator(FixedLocator([180./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi,0.5/np.pi]))
    #axs[1].xaxis.set_major_formatter(FixedFormatter([r'$\frac{180}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$',r'$\frac{1}{2\pi}$']))
    axs[1].xaxis.set_major_locator(FixedLocator([480,240,120,60,30,1]))
    axs[1].xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','1']))
    secax = axs[1].secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
    secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
    secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
    secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
    axs[1].grid()
#    xd2 = ifft(esp,axis=1)
#    axs[0].plot(xs,xd2[0,],label='reconstructed')
    fig.suptitle(f"{op} {pt}")
    if detrend:
        fig.savefig("{}_errspectra_{}_{}_detrend.png".format(model,op,pt),dpi=300)
    else:
        fig.savefig("{}_errspectra_{}_{}.png".format(model,op,pt),dpi=300)
    plt.show(block=False)
    plt.close(fig=fig)
for ax in [axgm,axlam]:
    ax.grid()
    #ax.set_title("spectral space")
    ax.set_xlabel(r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)")
    #ax.xaxis.set_major_locator(FixedLocator([180./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi,0.5/np.pi]))
    #ax.xaxis.set_major_formatter(FixedFormatter([r'$\frac{180}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$',r'$\frac{1}{2\pi}$']))
    ax.xaxis.set_major_locator(FixedLocator([480,240,120,60,30,1]))
    ax.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','1']))
    secax = ax.secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
    secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
    secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
    secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
axgm.legend()
axgm.set_ylabel("GM power spectral density")
axlam.set_ylabel("LAM power spectral density")
figall.suptitle(f"{op}")
if detrend:
    figall.savefig("{}_errspectra_{}_all_detrend.png".format(model,op),dpi=300)
else:
    figall.savefig("{}_errspectra_{}_all.png".format(model,op),dpi=300)
plt.show(block=False)
plt.close(fig=figall)

#t-test
from scipy import stats
##GM
methods = psdgm_dict.keys()
for i,m1 in enumerate(methods):
    for j,m2 in enumerate(methods):
        if j<=i: continue
        print(f"{m1}-{m2}")
        psd1 = psdgm_dict[m1]
        psd2 = psdgm_dict[m2]
        res = stats.ttest_rel(psd1,psd2,axis=0)
        t_stat = res.statistic
        if np.isnan(t_stat).all(): continue
        pvalue = res.pvalue
        t_conf = res.confidence_interval(confidence_level=0.95)
        dmean = (t_conf.high + t_conf.low)/2.
        fig,axs = plt.subplots(figsize=[6,9],nrows=3,sharex=True)
        axs[0].fill_between(wnum_gm,t_conf.high,t_conf.low,color='tab:blue',alpha=0.5)
        axs[0].set_ylabel("95% interval")
        axs[1].plot(wnum_gm,t_stat,lw=2.0)
        axs[1].set_ylabel("t-statistic")
        ymin, ymax = axs[1].get_ylim()
        if ymin*ymax < 0.0:
            axs[1].hlines([0.0],0,1,colors='r',transform=axs[1].get_yaxis_transform(),zorder=1)
        axs[2].plot(wnum_gm,pvalue,marker='^',lw=1.0)
        axs[2].hlines([0.05],0,1,colors='r',transform=axs[2].get_yaxis_transform(),zorder=1)
        axs[2].set_ylabel("p-value")
        axs[2].set_ylim(0.,0.1)
        for ax in axs:
            ax.grid(zorder=0)
        axs[2].set_xlabel(r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)")
        #ax.xaxis.set_major_locator(FixedLocator([180./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi,0.5/np.pi]))
        #ax.xaxis.set_major_formatter(FixedFormatter([r'$\frac{180}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$',r'$\frac{1}{2\pi}$']))
        axs[2].xaxis.set_major_locator(FixedLocator([480,240,120,60,30,1]))
        axs[2].xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','1']))
        secax = axs[0].secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
        secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
        secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
        secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
        fig.suptitle(f"GM {op}: {m1} - {m2}")
        fig.tight_layout()
        fig.savefig("{}_errspectra_gm_t-test_{}_{}-{}.png".format(model,op,m1,m2),dpi=300)
        plt.show()
##LAM
methods = psdlam_dict.keys()
for i,m1 in enumerate(methods):
    for j,m2 in enumerate(methods):
        if j<=i: continue
        print(f"{m1}-{m2}")
        psd1 = psdlam_dict[m1]
        psd2 = psdlam_dict[m2]
        res = stats.ttest_rel(psd1,psd2,axis=0)
        t_stat = res.statistic
        if np.isnan(t_stat).all(): continue
        pvalue = res.pvalue
        t_conf = res.confidence_interval(confidence_level=0.95)
        dmean = (t_conf.high + t_conf.low)/2.
        fig,axs = plt.subplots(figsize=[6,9],nrows=3,sharex=True)
        axs[0].fill_between(wnum_lam,t_conf.high,t_conf.low,color='tab:blue',alpha=0.5)
        axs[0].set_ylabel("95% interval")
        axs[1].plot(wnum_lam,t_stat,lw=2.0)
        axs[1].set_ylabel("t-statistic")
        ymin, ymax = axs[1].get_ylim()
        if ymin*ymax < 0.0:
            axs[1].hlines([0.0],0,1,colors='r',transform=axs[1].get_yaxis_transform(),zorder=1)
        axs[2].plot(wnum_lam,pvalue,marker='^',lw=1.0)
        axs[2].hlines([0.05],0,1,colors='r',transform=axs[2].get_yaxis_transform(),zorder=1)
        axs[2].set_ylabel("p-value")
        axs[2].set_ylim(0.,0.1)
        for ax in axs:
            ax.grid(zorder=0)
        axs[2].set_xlabel(r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)")
        #ax.xaxis.set_major_locator(FixedLocator([180./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi,0.5/np.pi]))
        #ax.xaxis.set_major_formatter(FixedFormatter([r'$\frac{180}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$',r'$\frac{1}{2\pi}$']))
        axs[2].xaxis.set_major_locator(FixedLocator([480,240,120,60,30,1]))
        axs[2].xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','1']))
        secax = axs[0].secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
        secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
        secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
        secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
        fig.suptitle(f"LAM {op}: {m1} - {m2}")
        fig.tight_layout()
        if detrend:
            fig.savefig("{}_errspectra_lam_t-test_{}_{}-{}_detrend.png".format(model,op,m1,m2),dpi=300)
        else:
            fig.savefig("{}_errspectra_lam_t-test_{}_{}-{}.png".format(model,op,m1,m2),dpi=300)
        plt.show()
