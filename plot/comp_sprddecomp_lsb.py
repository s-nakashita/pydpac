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
pt = 'envar' 
anl = True
#if len(sys.argv)>5:
#    anl = (sys.argv[5]=='T')

datadir = Path(f'/Volumes/FF520/nested_envar/data/{model}')
datadir = Path(f'../work/{model}')
preGMpt = 'envar'
ldscl=True
obsloc = ''
if len(sys.argv)>4:
    obsloc = sys.argv[4]
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
labels = {"dscl":"Downscaling","conv":"DA", "lsb":"DA+LSB", "nest":"Nested DA"}
linecolor = {"dscl":"k","conv":"tab:blue","lsb":'tab:orange',"nest":'tab:green'}

ns = 40 # spinup
tc = np.arange(ns,na) # cycles
t = tc / 4. # days
ntime = na - ns
nt1 = ntime // 3

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
    xsa = []
    xsd1 = []
    xsd2 = []
    xsd3 = []
    for i in range(ns,na):
        f = dscldir/"data/{2}/{0}_gm_ua_{1}_{2}_cycle{3}.npy".format(model,op,preGMpt,i)
        if not f.exists():
            print("not exist {}".format(f))
            exit()
        xagm = np.load(f)
        gm2lam = interp1d(ix_gm,xagm,axis=0)
        xadscl = gm2lam(ix_lam)
        xadscl = xadscl - np.mean(xadscl,axis=1)[:,None]
        _, xddecomp = truncope_lam.scale_decomp(xadscl,kthres=kthres)
        xsa1 = np.sqrt(np.diag(xadscl@xadscl.T/(xadscl.shape[1]-1)))
        xsa.append(xsa1)
        for xd, xsdlist in zip(xddecomp,[xsd1,xsd2,xsd3]):
            xsa1 = np.sqrt(np.diag(xd@xd.T/(xd.shape[1]-1)))
            xsdlist.append(xsa1)
    xds[key] = np.array(xsa)
    xddecomp = [np.array(xsd1),np.array(xsd2),np.array(xsd3)]
    xddecomps[key] = xddecomp
# LAM
for key in ['conv','lsb','nest']:
    xsa = []
    xsd1 = []
    xsd2 = []
    xsd3 = []
    for i in range(ns,na):
        if key=='conv':
            f = lamdir/"data/{2}/{0}_lam_ua_{1}_{2}_cycle{3}.npy".format(model,op,pt,i)
        elif key=='nest':
            f = lamdir/"data/{2}_nest/{0}_lam_ua_{1}_{2}_nest_cycle{3}.npy".format(model,op,pt,i)
        else:
            f = lsbdir/"data/{2}/{0}_lam_ua_{1}_{2}_cycle{3}.npy".format(model,op,pt,i)
        if not f.exists():
            print("not exist {}".format(f))
            continue
        xalam = np.load(f)
        xalam = xalam - np.mean(xalam,axis=1)[:,None]
        _, xddecomp = truncope_lam.scale_decomp(xalam,kthres=kthres)
        xsa1 = np.sqrt(np.diag(xalam@xalam.T/(xalam.shape[1]-1)))
        xsa.append(xsa1)
        for xd, xsdlist in zip(xddecomp,[xsd1,xsd2,xsd3]):
            xsa1 = np.sqrt(np.diag(xd@xd.T/(xd.shape[1]-1)))
            xsdlist.append(xsa1)
    xds[key] = np.array(xsa)
    xddecomp = [np.array(xsd1),np.array(xsd2),np.array(xsd3)]
    xddecomps[key] = xddecomp

vlim=0.8
for i in range(3):
    t0 = i*nt1
    t1 = min(na,t0+nt1)
    na1 = t1 - t0
    print(f"day{t[t0]}-{t[t1-1]}")
    vlims=[]
    for key in xddecomps.keys():
        xddecomp = xddecomps[key]
        if len(vlims)<len(xddecomp):
            for xdd in xddecomp:
                vlims.append(max(np.max(xdd[t0:t1]),-np.min(xdd[t0:t1])))
        else:
            continue
            #for k,xdd in enumerate(xddecomp):
            #    vlim0 = vlims[k]
            #    vlim1 = max(np.max(xdd[t0:t1]),-np.min(xdd[t0:t1]))
            #    vlims[k] = max(vlim0,vlim1)
    fig, axs = plt.subplots(ncols=4,sharey=True,figsize=[10,4],constrained_layout=True)
    figp, axsp = plt.subplots(nrows=3,ncols=4,sharey=True,sharex=True,figsize=[12,10],constrained_layout=True)
    plist=[]
    ylimlist=[]
    for j,key in enumerate(xds.keys()):
        xd = xds[key]
        xddecomp = xddecomps[key]
        p1 = axs[j].pcolormesh(ix_lam,t[t0:t1],xd[t0:t1],shading='auto',\
        cmap='viridis',norm=Normalize(0.0,vlim))
        axs[j].set_xticks(ix_lam[::(nx_lam//6)])
        axs[j].set_title(labels[key])
        for k, xdd in enumerate(xddecomp):
            p2 = axsp[k,j].pcolormesh(ix_lam,t[t0:t1],xdd[t0:t1],shading='auto',\
                cmap='viridis',norm=Normalize(0.0,vlims[k]))
            plist.append(p2)
            if k==0:
                axsp[k,j].set_yticks(t[t0:t1:(na1//8)])
                axsp[k,j].set_title(labels[key])
            if k==2:
                axsp[k,j].set_xticks(ix_lam[::(nx_lam//6)])
    fig.colorbar(p1,ax=axs[-1],shrink=0.6,pad=0.01)
    axsp[0,0].set_ylabel(r"large: $k <$"+f"{kthres[0]}")
    axsp[1,0].set_ylabel(f"middle: {kthres[0]}"+r"$\leq k <$"+f"{kthres[1]}")
    axsp[2,0].set_ylabel(f"small: {kthres[1]}"+r"$\leq k$")
    figp.colorbar(plist[0],ax=axsp[0,-1],shrink=0.6,pad=0.01)
    figp.colorbar(plist[1],ax=axsp[1,-1],shrink=0.6,pad=0.01)
    figp.colorbar(plist[2],ax=axsp[2,-1],shrink=0.6,pad=0.01)
    fig.suptitle(ptlong[pt]+f' analysis spread, day{t[t0]:.0f}-')
    figp.suptitle(ptlong[pt]+f' analysis spread, day{t[t0]:.0f}-')
    fig.savefig(figdir/'{}_xsafull{}_lam_{}_{}.png'.format(model,i+1,op,pt),dpi=300)
    figp.savefig(figdir/'{}_xsadecomp{}_lam_{}_{}.png'.format(model,i+1,op,pt),dpi=300)
    plt.show(block=False)
    plt.close()

for key in xds.keys():
    xd = xds[key]
    xddecomp = xddecomps[key]
    var = np.mean(xd**2,axis=1)
    sprd = np.sqrt(var)
    varmean = var.mean()
    varsum = np.zeros_like(var)
    scalelist=['large','middle','small']
    varlist = [var]
    for k, xdd in enumerate(xddecomp):
        var1 = np.mean(xdd**2,axis=1)
        var1mean = np.mean(var1)
        ratio = var1mean / varmean
        print("{}, {} analysis variance = {:.3e} ({:.3f})".format(key,scalelist[k],var1mean,ratio))
        varsum = varsum + var1
        varlist.append(var1)
    sprdsum = np.sqrt(varsum)
    print("{}, analysis variance = {:.3e} ({:.3e})".format(key,varmean,np.mean(varsum)))
    print("{}, analysis spread = {:.3e} ({:.3e})".format(key,np.mean(sprd),np.mean(sprdsum)))
    np.savetxt(figdir/f"vardecomp_{pt}_{key}.txt",np.array(varlist))