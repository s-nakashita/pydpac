import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from pathlib import Path
from matplotlib.patches import FancyArrowPatch
from scipy import fft

def mbs(x,y,lx,nbx,ly,nby,ntest=4999,seed=None):
    nx = x.size
    ny = y.size
    rng = np.random.default_rng(seed=seed)
    sref = y.mean() - x.mean()
    # moving block resampling
    xbs = []
    ybs = []
    sbs = []
    for itest in range(ntest):
        isx = rng.choice(x.size-lx+1,size=nbx,replace=True)
        xtmp = np.concatenate([x[i:i+lx] for i in isx])
        xbs.append(xtmp)
        isy = rng.choice(y.size-ly+1,size=nby,replace=True)
        ytmp = np.concatenate([y[i:i+ly] for i in isy])
        ybs.append(ytmp)
        sbs.append(ytmp.mean() - xtmp.mean())
    # jackknife variance estimation
    xbs = np.array(xbs)
    ybs = np.array(ybs)
    nlenx = lx*nbx
    nleny = ly*nby
    xmb = []
    for iblk in range(nbx):
        xtmp = 0.0
        for jblk in range(nbx):
            if iblk==jblk: continue
            xtmp = xtmp + np.sum(np.sum(xbs[:,jblk*lx:(jblk+1)*lx],axis=1))
        xmb.append(xtmp/(nlenx-lx))
    xmb = np.array(xmb)
    xmbm = np.mean(xmb,axis=0)
    varjx = np.sum((xmb-xmbm)**2)*(nbx-1)*nlenx/nbx/nx
    ymb = []
    for iblk in range(nby):
        ytmp = 0.0
        for jblk in range(nby):
            if iblk==jblk: continue
            ytmp = ytmp + np.sum(np.sum(ybs[:,jblk*ly:(jblk+1)*ly],axis=1))
        ymb.append(ytmp/(nleny-ly))
    ymb = np.array(ymb)
    ymbm = np.mean(ymb,axis=0)
    varjy = np.sum((ymb-ymbm)**2)*(nby-1)*nleny/nby/ny
    varbs = varjx + varjy

    dstat = (np.array(sbs) - sref)/np.sqrt(varbs)
    return np.sort(dstat)

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = 'var' #var or envar
if len(sys.argv)>4:
    pt = sys.argv[4]
anl = True
#if len(sys.argv)>5:
#    anl = (sys.argv[5]=='T')

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
if pt=='envar':
    lsbdir = datadir / f'envar_noinfl_lsb_preGM{obsloc}_m80obs30'
lamdir  = datadir / f'var_vs_envar_shrink_dct_preGM{obsloc}_m80obs30'
#if ldscl:
#    figdir = datadir
#else:
figdir = lsbdir

ptlong = {"envar":"EnVar","var":"3DVar"}
labels = {"dscl":"Dscl","conv":"DA", "lsb":"DA+LSB", "nest":"Nested DA"}
linecolor = {"dscl":"k","conv":"tab:blue","lsb":'tab:orange',"nest":'tab:green'}

ns = 40 # spinup
tc = np.arange(ns,na)+1 # cycles
t = tc / 4. # days
ntime = na - ns
nt1 = ntime // 3
errors = {}
ar1s = {}
ar2s = {}
if ldscl:
    keys = ['dscl','conv','lsb','nest']
else:
    keys = ['conv','lsb','nest']
for key in keys:
    if key=='dscl':
        if anl:
            f = dscldir / f"e_lam_{op}_{preGMpt}.txt"
        else:
            f = dscldir / f"ef_lam_{op}_{preGMpt}.txt"
    elif key=='conv':
        if anl:
            f = lamdir / f"e_lam_{op}_{pt}.txt"
            fs = lamdir / f"stda_lam_{op}_{pt}.txt"
        else:
            f = lamdir / f"ef_lam_{op}_{pt}.txt"
            fs = lamdir / f"stdf_lam_{op}_{pt}.txt"
    elif key=='nest':
        if anl:
            f = lamdir / f"e_lam_{op}_{pt}_nest.txt"
            fs = lamdir / f"stda_lam_{op}_{pt}_nest.txt"
        else:
            f = lamdir / f"ef_lam_{op}_{pt}_nest.txt"
            fs = lamdir / f"stdf_lam_{op}_{pt}_nest.txt"
    else:
        if anl:
            f = lsbdir / f"e_lam_{op}_{pt}.txt"
            fs = lsbdir / f"stda_lam_{op}_{pt}.txt"
        else:
            f = lsbdir / f"ef_lam_{op}_{pt}.txt"
            fs = lsbdir / f"stdf_lam_{op}_{pt}.txt"
    if not f.exists():
        print("not exist {}".format(f))
        continue
    e = np.loadtxt(f)[ns:]
    fig, axs = plt.subplots(nrows=3,constrained_layout=True)
    axs[0].plot(t,e)
    axs[0].set_xlabel('days')
    axs[0].set_ylabel('RMSE')
    dt = 6
    x = e - e.mean()
    y = fft.rfft(x)
    f = fft.rfftfreq(x.size,1./dt)
    p = 2.0 / y.size * np.abs(y)
    pra = [np.mean(p[max(0,i-4):min(p.size-1,i+1)]) for i in range(p.size)]
    axs[1].plot(f,p,lw=0.0,marker='.')
    axs[1].plot(f,pra)
    axs[1].set_xlabel('frequency (1/h)')
    axs[1].set_ylabel('Power')
    varx = np.sum(x**2)
    c = np.correlate(x,x,mode='full')
    r = c / varx
    r1 = r[r.size//2-1]
    print(r1)
    # autocorrelation model for 1st and 2nd orders (Zwiers and von Storch 1995; Wilks 1997)
    tau = np.arange(-x.size+1,x.size)
    ar1 = r1**np.abs(tau)
    xm2 = e[:-2] - e[:-2].mean()
    xp2 = e[2:] - e[2:].mean()
    varxm2 = np.sum(xm2**2)
    varxp2 = np.sum(xp2**2)
    r2 = np.correlate(xm2,xp2)/np.sqrt(varxm2*varxp2)
    phi1 = r1 * (1-r2) / (1-r1**2)
    phi2 = (r2-r1**2) / (1-r1**2)
    ar2 = np.zeros_like(ar1)
    ar2[0] = 1.0
    ar2[1] = r1; ar2[-1] = r1
    ar2[2] = r2; ar2[-2] = r2
    for k in range(x.size-1):
        ar2[k+2] = phi1*ar2[k+1]+phi2*ar2[k]
        ar2[-(k+2)] = phi1*ar2[-(k+1)]+phi2*ar2[-k]
    axs[2].plot(tau,r,label='raw')
    axs[2].plot(tau,ar1,ls='dashed',label='AR(1)')
    axs[2].plot(tau,np.roll(ar2,ar2.size//2),ls='dotted',label='AR(2)')
    axs[2].set_ylabel('autocorrelation')
    axs[2].set_xlabel('lag')
    axs[2].set_xlim(-30,30)
    axs[2].legend(loc='upper right') #,bbox_to_anchor=(0.8,0.9))
    fig.suptitle(labels[key])
    fig.savefig(figdir/f'acorr_rmse_{pt}_{key}.png')
    #plt.show()
    plt.close()
    errors[key] = e
    ar1s[key] = ar1
    ar2s[key] = ar2
#exit()

cmap = plt.get_cmap('tab10')
from scipy.stats import t as statt
for ii,k1 in enumerate(errors.keys()):
    if ii==len(errors)-1: break
    x = errors[k1]
    ar1x = ar1s[k1]
    ar2x = ar2s[k1]
    xp = x - x.mean()
    varx = np.sum(xp**2)
    f = open(figdir/f'htest_{pt}_{k1}.txt',mode='w')
    for jj,k2 in enumerate(errors.keys()):
        if k1==k2 or ii>jj: continue
        print(f"{k2}-{k1}")
        f.write(f"{k2}-{k1}\n")
        y = errors[k2]
        ar1y = ar1s[k2]
        ar2y = ar2s[k2]
        yp = y - y.mean()
        vary = np.sum(yp**2)
        # adjusted t-test (Zwiers and von Storch 1995; Wilks 2011, Section 5.2)
        d = y - x
        dmean = d.mean()
        vard = np.sum((d-dmean)**2)
        dm = d[:-1] - d[:-1].mean()
        dp = d[1:] - d[1:].mean()
        r1 = np.sum(dm*dp)/np.sqrt(np.sum(dm*dm))/np.sqrt(np.sum(dp*dp))
        ne = d.size * (1-r1) / (1+r1)
        """
        fig, axs = plt.subplots(nrows=3,constrained_layout=True)
        axs[0].plot(t,d)
        axs[0].set_xlabel('days')
        axs[0].set_ylabel('RMSE diff')
        dt = 6
        d_f = fft.rfft(d)
        freq = fft.rfftfreq(d.size,1./dt)
        p = 2.0 / d_f.size * np.abs(d_f)
        pra = [np.mean(p[max(0,i-4):min(p.size-1,i+1)]) for i in range(p.size)]
        axs[1].plot(freq,p,lw=0.0,marker='.')
        axs[1].plot(freq,pra)
        axs[1].set_xlabel('frequency (1/h)')
        axs[1].set_ylabel('Power')
        d_c = np.correlate(d-dmean,d-dmean,mode='full')
        rall = d_c / vard
        # autocorrelation model for 1st and 2nd orders (Zwiers and von Storch 1995; Wilks 1997)
        tau = np.arange(-d.size+1,d.size)
        ar1d = r1**np.abs(tau)
        dm2 = e[:-2] - e[:-2].mean()
        dp2 = e[2:] - e[2:].mean()
        vardm2 = np.sum(dm2**2)
        vardp2 = np.sum(dp2**2)
        r2 = np.correlate(dm2,dp2)/np.sqrt(vardm2*vardp2)
        phi1 = r1 * (1-r2) / (1-r1**2)
        phi2 = (r2-r1**2) / (1-r1**2)
        ar2d = np.zeros_like(ar1d)
        ar2d[0] = 1.0
        ar2d[1] = r1; ar2d[-1] = r1
        ar2d[2] = r2; ar2d[-2] = r2
        for k in range(d.size-1):
            ar2d[k+2] = phi1*ar2d[k+1]+phi2*ar2d[k]
            ar2d[-(k+2)] = phi1*ar2d[-(k+1)]+phi2*ar2d[-k]
        axs[2].plot(tau,rall,label='raw')
        axs[2].plot(tau,ar1d,ls='dashed',label='AR(1)')
        axs[2].plot(tau,np.roll(ar2d,ar2d.size//2),ls='dotted',label='AR(2)')
        axs[2].set_ylabel('autocorrelation')
        axs[2].set_xlabel('lag')
        axs[2].set_xlim(-30,30)
        axs[2].legend(loc='upper right') #,bbox_to_anchor=(0.8,0.9))
        fig.suptitle(f'{ptlong[pt]} ({labels[k2]}) - ({labels[k1]})')
        fig.savefig(figdir/f'acorr_rmse_{pt}_{k2}-{k1}.png')
        #plt.show()
        plt.close()
        """
        std = np.sqrt(vard/(d.size-1))
        tval = dmean / (std / np.sqrt(ne))
        df = ne
        if tval > 0.0:
            pval = 1.0 - statt.cdf(tval,df)
        else:
            pval = statt.cdf(tval,df)
        print(f"dmean={dmean:.3e} std={std:.3e} r1={r1:.2f} ne={ne:.2f}")
#        vix = 1.0 + 2.0 * np.sum((1.0-np.arange(1,x.size)/x.size)*ar1x[:x.size-1])
#        nex = x.size / vix
#        viy = 1.0 + 2.0 * np.sum((1.0-np.arange(1,y.size)/y.size)*ar1y[:y.size-1])
#        ney = y.size / viy
#        if (nex + ney) < 30.0: continue
#        rmod1 = (np.sum(xp[1:]*xp[:-1])+np.sum(yp[1:]*yp[:-1]))/(varx+vary)
#        nex = x.size * (1-rmod1) / (1+rmod1)
#        nex = max(2,nex)
#        nex = min(x.size,nex)
#        ney = y.size * (1-rmod1) / (1+rmod1)
#        ney = max(2,ney)
#        ney = min(y.size,ney)
#        std = np.sqrt((varx + vary)/(x.size+y.size-2))
#        tval = (y.mean() - x.mean()) / std / (1.0/np.sqrt(nex) + 1.0/np.sqrt(ney))
#        df = nex + ney
        res=f"ZvS test: ne={df:.2f} r1={r1:.2f} tval={tval:.3e} pval={pval:.3e}"+\
            f" 10%={statt.ppf(0.95,df):.3e}"+\
            f" 5%={statt.ppf(0.975,df):.3e}"+\
            f" 1%={statt.ppf(0.995,df):.3e}"
        print(res)
        f.write(res+'\n')
        continue
        """
        # moving block bootstrap test for AR(1) (Wilks 1997)
        vmodx = vix * np.exp(2.0*vix/x.size)
        vmody = viy * np.exp(2.0*viy/y.size)
        alp = 2.0*(1-1/vmodx)/3.0
        l0 = np.sqrt(float(x.size))
        niter = 0
        while niter < 10:
            l = (x.size-l0+1)**alp
            diff = np.abs(l - l0)
            dj = 1.0 + alp*(x.size-l0+1)**(alp-1)
            if np.abs(dj)<1.0e-5: break
            niter += 1
            #print(f"iter{niter} l0={l0:.2f} l={l:.2f} d={diff:.2e} dj={dj:.2e}")
            l0 = l - dj
        print(f"l={l0}, floor(l)={np.floor(l0)}")
        l0x = max(int(np.floor(l0)),1)
        nbx = int(x.size / l0x)

        alp = 2.0*(1-1/vmody)/3.0
        l0 = np.sqrt(float(y.size))
        niter = 0
        while niter < 10:
            l = (y.size-l0+1)**alp
            diff = np.abs(l - l0)
            dj = 1.0 + alp*(y.size-l0+1)**(alp-1)
            if np.abs(dj)<1.0e-5: break
            niter += 1
            #print(f"iter{niter} l0={l0:.2f} l={l:.2f} d={diff:.2e} dj={dj:.2e}")
            l0 = l - dj
        print(f"l={l0}, floor(l)={np.floor(l0)}")
        l0y = max(int(np.floor(l0)),1)
        nby = int(y.size / l0y)

        std = np.sqrt(vmodx*varx/x.size + vmody*vary/y.size)
        dval = (y.mean() - x.mean()) / std
        dstat = mbs(x,y,l0x,nbx,l0y,nby)
        du = np.sum(dstat>=dval)/(dstat.size+1)
        dl = np.sum(dstat<=dval)/(dstat.size+1)
        res = f"mbbs(1) test: upper={du:.3e} lower={dl:.3e}"
        print(res)
        f.write(res+'\n')
        nb1p = int(5.0e-3 * (dstat.size+1))
        nb5p = int(1.0e-2 * (dstat.size+1))
        nb10p = int(5.0e-2 * (dstat.size+1))
        conf = [[dstat[nb1p],dstat[-nb1p]],\
            [dstat[nb5p],dstat[-nb5p]],\
            [dstat[nb10p],dstat[-nb10p]]]

        title=f'{ptlong[pt]} ({labels[k2]}) - ({labels[k1]})'
        fig, ax = plt.subplots(figsize=[6,4],constrained_layout=True)
        ax.hist(dstat,bins=20,histtype='bar',color=cmap(0),alpha=0.3)
        ax.vlines([dval],0,1,colors=cmap(0),transform=ax.get_xaxis_transform(),zorder=0)
        mscale = 30
        zy = 0.9
        for ii in range(len(conf)):
            c1 = conf[ii]
            ax.vlines(c1,0,1,colors=cmap(ii+1),ls='dashed',transform=ax.get_xaxis_transform(),zorder=0)
            interval = FancyArrowPatch((c1[0],zy),(c1[1],zy),\
            arrowstyle='<|-|>',mutation_scale=mscale,color=cmap(ii+1),transform=ax.get_xaxis_transform())
            ax.add_patch(interval)
            zy-=0.03
            mscale-=10
        ax.set_title(title,fontsize=17)
        #ax.legend(loc='upper left',bbox_to_anchor=(0.85,1.0),fontsize=14,\
        #    title='confidence interval',title_fontsize=14)
        fig.savefig(figdir/'{}_ediff{}-{}_lam_mbbs1_{}_{}.png'.format(model,k2,k1,op,pt),dpi=300)
        plt.show()
        plt.close()
        """

        # moving block bootstrap test for AR(2) (Wilks 1997)
        vix = 1.0 + 2.0 * np.sum((1.0-np.arange(1,x.size)/x.size)*ar2x[:x.size-1])
        viy = 1.0 + 2.0 * np.sum((1.0-np.arange(1,y.size)/y.size)*ar2y[:y.size-1])
        vmodx = vix * np.exp(3.0*vix/x.size)
        vmody = viy * np.exp(3.0*viy/y.size)
        #alp = 2.0*(1-1/np.sqrt(4*vmodx))/3.0
        alp = 2.0*(1-1/vix)/3.0
        l0 = np.sqrt(float(x.size))
        niter = 0
        while niter < 10:
            l = (x.size-l0+1)**alp
            diff = np.abs(l - l0)
            dj = 1.0 + alp*(x.size-l0+1)**(alp-1)
            if np.abs(dj)<1.0e-5: break
            niter += 1
            #print(f"iter{niter} l0={l0:.2f} l={l:.2f} d={diff:.2e} dj={dj:.2e}")
            l0 = l - dj
        print(f"l={l0}, floor(l)={np.floor(l0)}")
        l0x = max(int(np.floor(l0)),1)
        nbx = int(x.size / l0x)
        print(f"lx={l0x}, nby={nbx}")

        #alp = 2.0*(1-1/np.sqrt(4*vmody))/3.0
        alp = 2.0*(1-1/viy)/3.0
        l0 = np.sqrt(float(y.size))
        niter = 0
        while niter < 10:
            l = (y.size-l0+1)**alp
            diff = np.abs(l - l0)
            dj = 1.0 + alp*(y.size-l0+1)**(alp-1)
            if np.abs(dj)<1.0e-5: break
            niter += 1
            #print(f"iter{niter} l0={l0:.2f} l={l:.2f} d={diff:.2e} dj={dj:.2e}")
            l0 = l - dj
        print(f"l={l0}, floor(l)={np.floor(l0)}")
        l0y = max(int(np.floor(l0)),1)
        nby = int(y.size / l0y)
        print(f"ly={l0y}, nby={nby}")

        std = np.sqrt(vmodx*varx/x.size + vmody*vary/y.size)
        dval = (y.mean() - x.mean()) / std
        dstat = mbs(x,y,l0x,nbx,l0y,nby)
        du = np.sum(dstat>=dval)/(dstat.size+1)
        dl = np.sum(dstat<=dval)/(dstat.size+1)
        res = f"mbbs(2) test: upper={du:.3e} lower={dl:.3e}"
        print(res)
        f.write(res+'\n')
        nb1p = int(5.0e-3 * (dstat.size+1))
        nb5p = int(1.0e-2 * (dstat.size+1))
        nb10p = int(5.0e-2 * (dstat.size+1))
        conf = [[dstat[nb1p],dstat[-nb1p]],\
            [dstat[nb5p],dstat[-nb5p]],\
            [dstat[nb10p],dstat[-nb10p]]]

        title=f'{ptlong[pt]} AR(2) ({labels[k2]}) - ({labels[k1]})'
        fig, ax = plt.subplots(figsize=[6,4],constrained_layout=True)
        ax.hist(dstat,bins=20,histtype='bar',color=cmap(0),alpha=0.3)
        ax.vlines([dval],0,1,colors=cmap(0),transform=ax.get_xaxis_transform(),zorder=0)
        mscale = 30
        zy = 0.9
        for ii in range(len(conf)):
            c1 = conf[ii]
            ax.vlines(c1,0,1,colors=cmap(ii+1),ls='dashed',transform=ax.get_xaxis_transform(),zorder=0)
            interval = FancyArrowPatch((c1[0],zy),(c1[1],zy),\
            arrowstyle='<|-|>',mutation_scale=mscale,color=cmap(ii+1),transform=ax.get_xaxis_transform())
            ax.add_patch(interval)
            zy-=0.03
            mscale-=10
        ax.set_title(title,fontsize=17)
        #ax.legend(loc='upper left',bbox_to_anchor=(0.85,1.0),fontsize=14,\
        #    title='confidence interval',title_fontsize=14)
        fig.savefig(figdir/'{}_ediff{}-{}_lam_mbbs2_{}_{}.png'.format(model,k2,k1,op,pt),dpi=300)
        plt.show()
        plt.close()
    f.close()
#        alp = 2.0*(1-1/np.sqrt(4*vmod))/3.0
#        l0 = np.sqrt(float(e.size))
#        niter = 0
#        while niter < 10:
#            l = (e.size-l0+1)**alp
#            diff = np.abs(l - l0)
#            dj = 1.0 + alp*(e.size-l0+1)**(alp-1)
#            if np.abs(dj)<1.0e-5: break
#            niter += 1
#            #print(f"iter{niter} l0={l0:.2f} l={l:.2f} d={diff:.2e} dj={dj:.2e}")
#            l0 = l - dj
#        print(f"l={l0}, floor(l)={np.floor(l0)}")
#        l0 = np.floor(l0)
#        vinfls[key] = vmod
#        nes[key] = ne
#        bls[key] = l0
