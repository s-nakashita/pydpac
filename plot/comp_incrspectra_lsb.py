import sys
import os
import matplotlib.pyplot as plt 
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullFormatter
from nmc_tools import NMC_tools, wnum2wlen, wlen2wnum
from pathlib import Path

plt.rcParams['font.size'] = 16
plt.rcParams['xtick.labelsize'] = 14

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = 'var' #var or envar
if len(sys.argv)>4:
    pt = sys.argv[4]

tc = np.arange(na)+1
t = tc / 4.
ns = 40 # spinup

datadir = Path(f'/Volumes/FF520/nested_envar/data/{model}')
datadir = Path(f'../work/{model}')
preGMpt = 'envar'
ldscl=True
obsloc = ''
if len(sys.argv)>5:
    obsloc = sys.argv[5]
#obsloc = '_partiall'
#obsloc = '_partialc'
#obsloc = '_partialr'
#obsloc = '_partialm'
dscldir = datadir / 'var_vs_envar_dscl_m80obs30'
lsbdir  = datadir / f'var_vs_envar_lsb_preGM{obsloc}_m80obs30'
lamdir  = datadir / f'var_vs_envar_shrink_dct_preGM{obsloc}_m80obs30'
#if ldscl:
#    figdir = datadir
#else:
figdir = lsbdir
figpngdir = Path(os.environ['HOME']+'/Writing/nested_envar/figpng')
figpdfdir = Path(os.environ['HOME']+'/Writing/nested_envar/figpdf')
if obsloc == '':
    figpngdir = figpngdir / 'uniform'
    figpdfdir = figpdfdir / 'uniform'
else:
    figpngdir = figpngdir / obsloc[1:]
    figpdfdir = figpdfdir / obsloc[1:]
if not figpngdir.exists():
    figpngdir.mkdir(parents=True)
if not figpdfdir.exists():
    figpdfdir.mkdir(parents=True)

ptlong = {"envar":"EnVar","var":"3DVar"}
labels = {"dscl":"No LAM DA","conv":"DA", "lsb":"BLSB+DA", "nest":"Nested DA"}
linecolor = {"dscl":"k","conv":"tab:blue","lsb":'tab:orange',"nest":'tab:green'}

ix_t = np.loadtxt(dscldir/"ix_true.txt")
ix_gm = np.loadtxt(dscldir/"ix_gm.txt")
ix_lam = np.loadtxt(dscldir/"ix_lam.txt")[1:-1]
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
nmc_t = NMC_tools(ix_t_rad,cyclic=True,ttype='c')
nmc_gm = NMC_tools(ix_gm_rad,cyclic=True,ttype='c')
nmc_lam = NMC_tools(ix_lam_rad,cyclic=False,ttype='c')

#fig, (axb,axa,axi) = plt.subplots(nrows=3,sharex=True,figsize=[8,8],constrained_layout=True)
fig, axi = plt.subplots(figsize=[8,7],constrained_layout=True)
fig1d, ax1d = plt.subplots(figsize=[12,6],constrained_layout=True)
## GM
#fa = dscldir/"xagm_{}_{}.npy".format(op,preGMpt)
#fb = dscldir/"{}_xfgm_{}_{}.npy".format(model,op,preGMpt)
#if not fa.exists():
#    print("not exist {}".format(fa))
#    exit()
#if not fb.exists():
#    print("not exist {}".format(fb))
#    exit()
#xagm = np.load(fa)
#xbgm = np.load(fb)
#incrgm = xagm[ns:na,:] - xbgm[ns:na,:]
#wnum_gm, psd_gm = nmc_gm.psd(xbgm,axis=1)
#axb.loglog(wnum_gm,psd_gm,c='gray',lw=4.0,label='GM')
#wnum_gm, psd_gm = nmc_gm.psd(xagm,axis=1)
#axa.loglog(wnum_gm,psd_gm,c='gray',lw=4.0,label='GM')
#wnum_gm, psd_gm = nmc_gm.psd(incrgm,axis=1)
#axi.loglog(wnum_gm,psd_gm,c='gray',lw=4.0,label='GM')

psd_incr = {}
psd_mean = {}
psd_sprd = {}
if ldscl:
    key='dscl'
    # downscaling
    fa = dscldir/"xalam_{}_{}.npy".format(op,preGMpt)
    fb = dscldir/"{}_xflam_{}_{}.npy".format(model,op,preGMpt)
    if not fa.exists():
        print("not exist {}".format(fa))
        exit()
    xadscl = np.load(fa)[:,1:-1]
    xbdscl = np.load(fb)[:,1:-1]
    incrdscl = xadscl[ns:na,:] - xbdscl[ns:na,:]
    incrrms = np.sqrt(np.mean(incrdscl**2,axis=1))
    ax1d.plot(t[ns:na],incrrms,c=linecolor[key],label=labels[key]+f'={incrrms.mean():.3f}')
    #wnum, psd_dscl = nmc_lam.psd(xbdscl,axis=1)
    #axb.loglog(wnum,psd_dscl,c=linecolor[key],lw=2.0,label=labels[key])
    #wnum, psd_dscl = nmc_lam.psd(xadscl,axis=1)
    #axa.loglog(wnum,psd_dscl,c=linecolor[key],lw=2.0,label=labels[key])
    wnum, psd_dscl = nmc_lam.psd(incrdscl,axis=1)
    axi.loglog(wnum,psd_dscl,c=linecolor[key],lw=2.0,label=labels[key])
    psd_incr[key] = psd_dscl
    if pt=='envar':
        psdm_dscl = np.zeros_like(psd_dscl)
        psds_dscl = np.zeros_like(psd_dscl)
        for i in range(ns,na):
            f = dscldir/"data/{2}/{0}_gm_ua_{1}_{2}_cycle{3}.npy".format(model,op,preGMpt,i)
            xa1 = np.load(f)
            gm2lam = interp1d(ix_gm,xa1,axis=0)
            xadscl = gm2lam(ix_lam)
            _, psd1 = nmc_lam.psd(xadscl,axis=0,average=False)
            psdm_dscl = psdm_dscl + psd1.mean(axis=1)
            xadscl = xadscl - np.mean(xadscl,axis=1)[:,None]
            _, psd1 = nmc_lam.psd(xadscl,axis=0,average=False)
            psds_dscl = psds_dscl + psd1.mean(axis=1)
        #xsadscl = np.load(f)
        #wnum, psds_dscl = nmc_lam.psd(xsadscl,axis=1)
        psd_mean[key] = psdm_dscl / float(na-ns)
        psd_sprd[key] = psds_dscl / float(na-ns)
    #ax.plot(ix_lam, np.mean(xsadscl,axis=0),\
    #    c='k',ls='dashed')
# LAM
for key in ['conv','lsb','nest']:
    if key=='conv':
        fa = lamdir/"xalam_{}_{}.npy".format(op,pt)
        fb = lamdir/"xflam_{}_{}.npy".format(op,pt)
    elif key=='nest':
        fa = lamdir/"xalam_{}_{}_nest.npy".format(op,pt)
        fb = lamdir/"xflam_{}_{}_nest.npy".format(op,pt)
    else:
        fa = lsbdir/"xalam_{}_{}.npy".format(op,pt)
        fb = lsbdir/"xflam_{}_{}.npy".format(op,pt)
    if not fa.exists():
        print("not exist {}".format(fa))
        exit()
    xalam = np.load(fa)[:,1:-1]
    xblam = np.load(fb)[:,1:-1]
    incrlam = xalam[ns:na,:] - xblam[ns:na,:]
    incrrms = np.sqrt(np.mean(incrlam**2,axis=1))
    ax1d.plot(t[ns:na],incrrms,c=linecolor[key],label=labels[key]+f'={incrrms.mean():.3f}')
    #wnum, psd_lam = nmc_lam.psd(xblam,axis=1)
    #axb.loglog(wnum,psd_lam,c=linecolor[key],lw=2.0,label=labels[key])
    #wnum, psd_lam = nmc_lam.psd(xalam,axis=1)
    #axa.loglog(wnum,psd_lam,c=linecolor[key],lw=2.0,label=labels[key])
    wnum, psd_lam = nmc_lam.psd(incrlam,axis=1)
    if pt!='envar' or key!='lsb':
        axi.loglog(wnum,psd_lam,c=linecolor[key],lw=2.0,label=labels[key])
    psd_incr[key] = psd_lam
    if pt=='envar':
        #if key=='conv':
        #    f = lamdir/"xsalam_{}_{}.npy".format(op,pt)
        #elif key=='nest':
        #    f = lamdir/"xsalam_{}_{}_nest.npy".format(op,pt)
        #else:
        #    f = lsbdir/"xsalam_{}_{}.npy".format(op,pt)
        psdm_lam = np.zeros(psd_lam.size)
        psds_lam = np.zeros(psd_lam.size)
        if key=='lsb':
            incrlsb = []
            incrda = []
        for i in range(ns,na):
            if key=='conv':
                f = lamdir/"data/{2}/{0}_lam_ua_{1}_{2}_cycle{3}.npy".format(model,op,pt,i)
            elif key=='nest':
                f = lamdir/"data/{2}_nest/{0}_lam_ua_{1}_{2}_nest_cycle{3}.npy".format(model,op,pt,i)
            else:
                f = lsbdir/"data/{2}/{0}_lam_ua_{1}_{2}_cycle{3}.npy".format(model,op,pt,i)
                fbb = lsbdir/"data/{2}/{0}_lam_uf_{1}_{2}_cycle{3}.npy".format(model,op,pt,i)
                xbblam = np.load(fbb)
                incrlsb.append(xbblam.mean(axis=1) - xblam[i,])
            xalam = np.load(f)
            _, psd1 = nmc_lam.psd(xalam,axis=0,average=False)
            psdm_lam = psdm_lam + psd1.mean(axis=1)
            xaprtb = xalam - np.mean(xalam,axis=1)[:,None]
            wnum_s, psd1 = nmc_lam.psd(xaprtb,axis=0,average=False)
            psds_lam = psds_lam + psd1.mean(axis=1)
            if key=='lsb':
                incrda.append(xalam.mean(axis=1)-xbblam.mean(axis=1))
        if key=='lsb':
            incrlsb = np.array(incrlsb)
            incrrms = np.sqrt(np.mean(incrlsb**2,axis=1))
            ax1d.plot(t[ns:na],incrrms,ls='dashed',c=linecolor[key],label=labels[key]+f',lsb={incrrms.mean():.3f}')
            wnum, psd = nmc_lam.psd(incrlsb,axis=1)
            axi.loglog(wnum,psd,ls='dashed',c=linecolor['lsb'],lw=2.0,label=labels['lsb']+'(LSB)')
            psd_incr[key+'b'] = psd
            incrda = np.array(incrda)
            incrrms = np.sqrt(np.mean(incrda**2,axis=1))
            ax1d.plot(t[ns:na],incrrms,ls='dotted',c=linecolor[key],label=labels[key]+f',da={incrrms.mean():.3f}')
            wnum, psd = nmc_lam.psd(incrda,axis=1)
            axi.loglog(wnum,psd,ls='dotted',c=linecolor['lsb'],lw=2.0,label=labels['lsb']+'(DA)')
            psd_incr[key+'da'] = psd
        #xsalam = np.load(f)
        #wnum, psds_lam = nmc_lam.psd(xsalam,axis=1)
        psd_mean[key] = psdm_lam / float(na-ns)
        psd_sprd[key] = psds_lam / float(na-ns)
    #if pt == 'var' or pt == 'var_nest':
    #    ax.plot(ix_lam[1:-1], np.mean(xsalam,axis=0),\
    #    c=linecolor[pt],ls='dashed')
    #else:
    #    ax.plot(ix_lam, np.mean(xsalam,axis=0),\
    #    c=linecolor[pt],ls='dashed')

for i,ax in enumerate([axi]): #[axb,axa,axi]):
    ax.grid()
    #ax.xaxis.set_major_locator(FixedLocator([180./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi,0.5/np.pi]))
    #ax.xaxis.set_major_formatter(FixedFormatter([r'$\frac{180}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$',r'$\frac{1}{2\pi}$']))
    ax.xaxis.set_major_locator(FixedLocator([480,240,120,60,30,12,2]))
    secax = ax.secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
    secax.xaxis.set_major_locator(FixedLocator([np.pi,np.pi/6.,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
    if i==0:
        ax.set_xlabel(r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)")
        ax.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','12','2']))
    else:
        ax.xaxis.set_major_formatter(NullFormatter())
    if i==0:
        secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
        secax.xaxis.set_major_formatter(FixedFormatter([r'$\pi$',r'$\frac{\pi}{6}$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
    else:
        secax.xaxis.set_major_formatter(NullFormatter())
#ybmin, ybmax = axb.get_ylim()
#yamin, yamax = axa.get_ylim()
yimin, yimax = axi.get_ylim()
#ymin = min(ybmin,yamin)
#ymax = max(ybmax,yamax)
#axb.set_ylim(ymin,ymax)
#axa.set_ylim(ymin,ymax)
#axb.legend(loc='upper left',bbox_to_anchor=(1.0,1.0))
#axb.set_ylabel('first guess')
#axa.set_ylabel('analysis')
axi.vlines([24],0,1,colors='magenta',ls='dashed',zorder=0,transform=ax.get_xaxis_transform())
axi.set_ylim(4.0e-7,3.0e-2) #1e-8,2e-2)
axi.legend() #loc='upper left',bbox_to_anchor=(1.0,1.0))
axi.set_ylabel('increment')
fig.suptitle(ptlong[pt])
fig.savefig(figpngdir/f"{model}_incrspectra_{op}_{pt}.png",dpi=300)
fig.savefig(figpdfdir/f"{model}_incrspectra_{op}_{pt}.pdf")
plt.show(block=False)

ax1d.set_xlabel('day')
ax1d.set_ylabel('spatial averaged rms increment')
ax1d.legend(loc='upper left',bbox_to_anchor=(0.82,1.0),title=ptlong[pt])
#fig1d.savefig(figpngdir/f"{model}_incr1d_{op}_{pt}.png")
plt.show()
plt.close()

if pt=='envar':
    #fig, (axs,axi) = plt.subplots(nrows=2,sharex=True,figsize=[8,6],constrained_layout=True)
    fig, axs = plt.subplots(figsize=[8,7],constrained_layout=True)
    figr, axrs = plt.subplots(ncols=3,figsize=[12,6],constrained_layout=True)
    rlabels = ['mean','prtb']
    rlines = [Line2D([0],[0],lw=2.0,ls='dashed',c='k'),Line2D([0],[0],lw=2.0,ls='dotted',c='k')]
    for key in psd_sprd.keys():
        if key!='lsbb' and key!='lsbda':
            if key=='dscl':
                axs.loglog(wnum,psd_sprd[key],c=linecolor[key],lw=2.0,label=labels[key])
                axrs[0].loglog(wnum,psd_mean[key],c=linecolor[key],lw=2.0,ls='dashed',label=labels[key])
                axrs[1].loglog(wnum,psd_sprd[key],c=linecolor[key],lw=2.0,ls='dotted',label=labels[key])
            else:
                axs.loglog(wnum_s,psd_sprd[key],c=linecolor[key],lw=2.0,label=labels[key])
                axrs[0].loglog(wnum_s,psd_mean[key],c=linecolor[key],lw=2.0,ls='dashed',label=labels[key])
                axrs[1].loglog(wnum_s,psd_sprd[key],c=linecolor[key],lw=2.0,ls='dotted',label=labels[key])
                r = (psd_mean[key] - psd_mean['dscl'])/psd_mean[key]
                axrs[2].semilogx(wnum_s, r*100, c=linecolor[key],lw=2.0,ls='dashed') #,label=labels[key])
                r = (psd_sprd[key] - psd_sprd['dscl'])/psd_sprd[key]
                axrs[2].semilogx(wnum_s, r*100, c=linecolor[key],lw=2.0,ls='dotted') #,label=labels[key])
            #if key!='lsb':
            #    axi.loglog(wnum,psd_incr[key],c=linecolor[key],lw=2.0,label=labels[key])
        #elif key=='lsbb':
        #    axi.loglog(wnum,psd_incr[key],ls='dashed',c=linecolor['lsb'],lw=2.0,label=labels['lsb']+'(LSB)')
        #else:
        #    axi.loglog(wnum,psd_incr[key],ls='dotted',c=linecolor['lsb'],lw=2.0,label=labels['lsb']+'(DA)')
    #ymin, ymax = axs.get_ylim()
    #ymin=1e-8; ymax=1e-1
    ymin=4e-7; ymax=3e-2
    for i,ax in enumerate([axs]+axrs.tolist()):
        if i==0:
            ax.set_ylim(ymin,ymax)
        ax.grid()
        ax.vlines([24],0,1,colors='magenta',ls='dashed',zorder=0,transform=ax.get_xaxis_transform())
        #ax.xaxis.set_major_locator(FixedLocator([180./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi,0.5/np.pi]))
        #ax.xaxis.set_major_formatter(FixedFormatter([r'$\frac{180}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$',r'$\frac{1}{2\pi}$']))
        ax.xaxis.set_major_locator(FixedLocator([480,240,120,60,30,12,2]))
        secax = ax.secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
        secax.xaxis.set_major_locator(FixedLocator([np.pi,np.pi/6.,np.pi/15.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
        #if i==1:
        ax.set_xlabel(r"wave number ($\omega_k=\frac{2\pi}{\lambda_k}$)")
        ax.xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','30','12','2']))
        #else:
        #    ax.xaxis.set_major_formatter(NullFormatter())
        #if i==0:
        secax.set_xlabel(r'wave length ($\lambda_k=\frac{2\pi}{\omega_k}$)')
        secax.xaxis.set_major_formatter(FixedFormatter([r'$\pi$',r'$\frac{\pi}{6}$',r'$\frac{\pi}{15}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
        #else:
        #    secax.xaxis.set_major_formatter(NullFormatter())
    #axi.legend(loc='upper left',bbox_to_anchor=(1.0,1.0))
    axs.legend() #loc='upper left',bbox_to_anchor=(1.0,1.0))
    axs.set_ylabel('spread')
    #axi.set_ylabel('increment')
    fig.suptitle(ptlong[pt])
    #fig.savefig(figdir/f"{model}_incr+sprdspectra_{op}_{pt}.png",dpi=300)
    #fig.savefig(figdir/f"{model}_incr+sprdspectra_{op}_{pt}.pdf")
    fig.savefig(figpngdir/f"{model}_sprdspectra_{op}_{pt}.png",dpi=300)
    fig.savefig(figpdfdir/f"{model}_sprdspectra_{op}_{pt}.pdf")
    axrs[0].legend(fontsize=12)
    axrs[0].set_title('mean',fontsize=14)
    axrs[1].set_title('prtb',fontsize=14)
    axrs[1].sharey(axrs[0])
    axrs[2].set_title('(LAM DA - No LAM DA)/(LAM DA)',fontsize=14)
    axrs[2].legend(rlines,rlabels,fontsize=12)
    axrs[2].set_ylabel('%')
    figr.suptitle(ptlong[pt])
    figr.savefig(figpngdir/f"{model}_compspectra_{op}_{pt}.png",dpi=300)
    figr.savefig(figpdfdir/f"{model}_compspectra_{op}_{pt}.pdf")
    plt.show() #block=False)
    plt.close()