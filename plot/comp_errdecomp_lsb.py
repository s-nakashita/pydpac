import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.title_fontsize'] = 24
from matplotlib.colors import Normalize
from pathlib import Path
from scipy.interpolate import interp1d
sys.path.append(os.path.join(os.path.dirname(__file__),'../analysis'))
from trunc1d import Trunc1d

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
lamdir  = datadir / f'var_vs_envar_shrink_dct_preGM{obsloc}_m80obs30'
#if ldscl:
#    figdir = datadir
#else:
figdir = lsbdir

ptlong = {"envar":"EnVar","var":"3DVar"}
labels = {"dscl":"No LAM DA","conv":"LAM DA", "lsb":"DA+LSB", "nest":"Nested DA"}
linecolor = {"dscl":"k","conv":"tab:blue","lsb":'tab:orange',"nest":'tab:green'}
captions = {"envar":"(b)","var":"(a)"}

tc = np.arange(na)+1 # cycles
t = tc / 4. # days
ns = 40 # spinup
ntime = na - ns
nt1 = ntime // 3

ix_t = np.loadtxt(dscldir/"ix_true.txt")
ix_gm = np.loadtxt(dscldir/"ix_gm.txt")
ix_lam = np.loadtxt(dscldir/"ix_lam.txt")
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
truncope_t =Trunc1d(ix_t_rad,cyclic=True,ttype='c',resample=False)
truncope_gm = Trunc1d(ix_gm_rad,cyclic=True,ttype='c',resample=False)
truncope_lam = Trunc1d(ix_lam_rad,cyclic=False,ttype='c',resample=False)
kthres = [24.,60.]

# nature
f = dscldir/"truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[:na,:]
xt2x = interp1d(ix_t,xt)

xds={}
xddecomps={}
if ldscl:
    key='dscl'
    # downscaling
    f = dscldir / f"xalam_{op}_{preGMpt}.npy"
    if not f.exists():
        print("not exist {}".format(f))
        exit()
    xadscl = np.load(f)
    xd = xadscl - xt2x(ix_lam)
    _, xddecomp = truncope_lam.scale_decomp(xd.T,kthres=kthres)
    xds[key] = xd
    xddecomps[key] = xddecomp
# LAM
for key in ['conv','lsb','nest']:
    if key=='conv':
        f = lamdir/"xalam_{}_{}.npy".format(op,pt)
    elif key=='nest':
        f = lamdir/"xalam_{}_{}_nest.npy".format(op,pt)
    else:
        f = lsbdir/"xalam_{}_{}.npy".format(op,pt)
    if not f.exists():
        print("not exist {}".format(f))
        continue
    xalam = np.load(f)
    xd = xalam - xt2x(ix_lam)
    _, xddecomp = truncope_lam.scale_decomp(xd.T,kthres=kthres)
    xds[key] = xd
    xddecomps[key] = xddecomp

vlim=3.0
for i in range(3):
    t0 = ns-1+i*nt1
    t1 = min(na,t0+nt1)
    na1 = t1 - t0
    print(f"day{t[t0]}-{t[t1-1]}")
    vlims=[]
    for key in xddecomps.keys():
        xddecomp = xddecomps[key]
        if len(vlims)<len(xddecomp):
            for xdd in xddecomp:
                vlims.append(max(np.max(xdd.T[t0:t1]),-np.min(xdd.T[t0:t1])))
        else:
            continue
            #for k,xdd in enumerate(xddecomp):
            #    vlim0 = vlims[k]
            #    vlim1 = max(np.max(xdd.T[t0:t1]),-np.min(xdd.T[t0:t1]))
            #    vlims[k] = max(vlim0,vlim1)
    fig, axs = plt.subplots(ncols=5,sharey=True,figsize=[13,4],constrained_layout=True)
    figp, axsp = plt.subplots(nrows=3,ncols=4,sharey=True,sharex=True,figsize=[12,10],constrained_layout=True)
    fig1d, axs1d = plt.subplots(nrows=3,ncols=1,sharex=True,figsize=[10,10],constrained_layout=True)
    p0 = axs[0].pcolormesh(ix_lam,t[t0:t1],xt2x(ix_lam)[t0:t1],shading='auto',\
        cmap='gnuplot',norm=Normalize(-15,15))
    axs[0].set_xticks(ix_lam[::(nx_lam//6)])
    axs[0].set_yticks(t[t0:t1:(na1//8)])
    axs[0].set_ylabel("days")
    axs[0].set_title('nature')
    fig.colorbar(p0,ax=axs[0],shrink=0.6,pad=0.01)#,orientation='horizontal'
    plist=[]
    ylimlist=[]
    for j,key in enumerate(xds.keys()):
        xd = xds[key]
        xddecomp = xddecomps[key]
        mse = np.mean(xd[t0:t1]**2,axis=1)
        p1 = axs[j+1].pcolormesh(ix_lam,t[t0:t1],xd[t0:t1],shading='auto',\
        cmap='coolwarm',norm=Normalize(-vlim,vlim))
        axs[j+1].set_xticks(ix_lam[::(nx_lam//6)])
        axs[j+1].set_title(labels[key]+f': MSE={np.mean(mse):.3f}',fontsize=14)
        msesum = np.zeros_like(mse)
        for k, xdd in enumerate(xddecomp):
            p2 = axsp[k,j].pcolormesh(ix_lam,t[t0:t1],xdd.T[t0:t1],shading='auto',\
                cmap='coolwarm',norm=Normalize(-vlims[k],vlims[k]))
            plist.append(p2)
            if k==0:
                axsp[k,j].set_yticks(t[t0:t1:(na1//8)])
                axsp[k,j].set_title(labels[key])
            if k==2:
                axsp[k,j].set_xticks(ix_lam[::(nx_lam//6)])
            mse1 = np.mean(xdd.T[t0:t1]**2,axis=1)
            axs1d[k].plot(t[t0:t1],mse1,c=linecolor[key],label=labels[key]+f':{np.mean(mse1):.3e}')
            msesum = msesum + mse1
            if j==0:
                ymin,ymax=axs1d[k].get_ylim()
                ylimlist.append((ymin,ymax))
            #for kk in range(k+1,len(xddecomp)):
            #    xdd2 = xddecomp[kk]
            #    cxd = xdd.T*xdd2.T
            #    cmse1 = 2.0*np.mean(cxd[t0:t1],axis=1)
            #    axs1d[k+kk-1,1].plot(t[t0:t1],cmse1,c=linecolor[key],label=labels[key]+f':{np.mean(cmse1):.3f}')
            #    msesum = msesum + cmse1
        #rmsesum = np.sqrt(msesum)
        print("{}, analysis MSE = {} ({})".format(key,np.mean(mse),np.mean(msesum)))
    fig.colorbar(p1,ax=axs[-1],shrink=0.6,pad=0.01)
    axsp[0,0].set_ylabel(r"large: $k <$"+f"{kthres[0]}")
    axsp[1,0].set_ylabel(f"middle: {kthres[0]}"+r"$\leq k <$"+f"{kthres[1]}")
    axsp[2,0].set_ylabel(f"small: {kthres[1]}"+r"$\leq k$")
    figp.colorbar(plist[0],ax=axsp[0,-1],shrink=0.6,pad=0.01)
    figp.colorbar(plist[1],ax=axsp[1,-1],shrink=0.6,pad=0.01)
    figp.colorbar(plist[2],ax=axsp[2,-1],shrink=0.6,pad=0.01)
    axs1d[0].set_title('(large)^2')
    axs1d[1].set_title('(middle)^2')
    axs1d[2].set_title('(small)^2')
    #axs1d[0,1].set_title('2(large)x(middle)')
    #axs1d[1,1].set_title('2(large)x(small)')
    #axs1d[2,1].set_title('2(middle)x(small)')
    for k,ax in enumerate(axs1d.flatten()):
        #ymin, ymax = ax.get_ylim()
        #if ymax>5.0:
        #    ymax = 5.0
        #if ymin<-5.0:
        #    ymin = -5.0
        ymin, ymax = ylimlist[k]
        ax.set_ylim(ymin,ymax)
        ax.grid()
        ax.legend(loc='upper left',bbox_to_anchor=(1.0,1.0))
    fig.suptitle(ptlong[pt]+f' analysis error, day{t[t0]:.0f}-')
    figp.suptitle(ptlong[pt]+f' analysis error, day{t[t0]:.0f}-')
    fig1d.suptitle(ptlong[pt]+f' analysis MSE, day{t[t0]:.0f}-')
    fig.savefig(figdir/'{}_xdfull{}_lam_{}_{}.png'.format(model,i+1,op,pt),dpi=300)
    figp.savefig(figdir/'{}_xddecomp{}_lam_{}_{}.png'.format(model,i+1,op,pt),dpi=300)
    fig1d.savefig(figdir/'{}_edecomp{}_lam_{}_{}.png'.format(model,i+1,op,pt),dpi=300)
    plt.show(block=False)
    plt.close()

for key in xds.keys():
    xd = xds[key]
    xddecomp = xddecomps[key]
    mse = np.mean(xd[ns:]**2,axis=1)
    rmse = np.sqrt(mse)
    msemean = np.mean(mse)
    msesum = np.zeros_like(mse)
    scalelist=['large','middle','small']
    mselist = [mse]
    for k, xdd in enumerate(xddecomp):
        mse1 = np.mean(xdd.T[ns:]**2,axis=1)
        mse1mean = np.mean(mse1)
        ratio = mse1mean / msemean
        print("{}, {} analysis MSE = {:.3e} ({:.3f})".format(key,scalelist[k],mse1mean,ratio))
        msesum = msesum + mse1
        mselist.append(mse1)
    rmsesum = np.sqrt(msesum)
    print("{}, analysis MSE = {:.3e} ({:.3e})".format(key,msemean,np.mean(msesum)))
    print("{}, analysis RMSE = {:.3e} ({:.3e})".format(key,np.mean(rmse),np.mean(rmsesum)))
    np.savetxt(figdir/f"msedecomp_{pt}_{key}.txt",np.array(mselist))