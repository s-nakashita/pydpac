import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from pathlib import Path
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['legend.title_fontsize'] = 24
from matplotlib.patches import FancyArrowPatch

# autocorrelation estimation using jackknife method
def forward(x):
    x = np.array(x, int)
    near_zero = np.isclose(x, 0)
    x[near_zero] = 0
    x[~near_zero] = 960 / x[~near_zero]
    return x
backward = forward
def jackknife(e,t,key):
    nws = [2,3,4,6,8,10,12,15,16,20,24,30,32,40,48,64,80,96,120,160,240,320,480]
    fall = np.mean(e[ns:])
    fest = []
    jerr = []
    ngroups = []
    for nw in nws:
        ngroup = e[ns:].size // nw
        ngroups.append(ngroup)
        ftmp = np.array([np.sum(e[ns+l*nw:ns+(l+1)*nw])/nw for l in range(ngroup)])
        fest.append(np.mean(ftmp))
        jerr.append(np.sqrt(np.sum((ftmp - fall)**2)/ngroup/(ngroup-1)))
    figj, axj = plt.subplots(figsize=[8,6],constrained_layout=True)
    axj.errorbar(np.arange(1,len(fest)+1),fest,yerr=jerr)
    axj.set_xlim(0,len(fest)+1)
    axj.set_xlabel('w')
    axj.xaxis.set_major_locator(FixedLocator(np.arange(1,len(fest)+1,2)))
    axj.xaxis.set_major_formatter(FixedFormatter([f'{nw}\n{ng}' for nw, ng in zip(nws[::2],ngroups[::2])]))
    axj.set_ylabel('<e>')
    if key=='dscl':
        axj.set_title(labels[key])
    else:
        axj.set_title(ptlong[pt]+' '+labels[key])
    axj.grid()
    #sax = axj.secondary_xaxis('top', functions=(forward,backward))
    #sax.set_xlabel('n')
    #sax.xaxis.set_major_locator(FixedLocator(forward(nws[::2])))
    if key=='dscl':
        figj.savefig(figdir/f'e_lam_jk_{key}.png')
    else:
        figj.savefig(figdir/f'e_lam_jk_{pt}_{key}.png')
    plt.close(fig=figj)
    # resampling
    nwc = 32
    n_resample = e[ns:].size // nwc
    e_resample = np.array([np.sum(e[ns+l*nwc:ns+(l+1)*nwc])/nwc for l in range(n_resample)])
    t_resample = np.array([t[ns+nwc//2+nwc*l] for l in range(n_resample)])
    return nwc, e_resample, t_resample

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
#if pt=='envar':
#    lsbdir = datadir / f'envar_noinfl_lsb_preGM{obsloc}_m80obs30'
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
labels = {"dscl":"No LAM DA","conv":"LAM DA", "lsb":"BLSB+DA", "nest":"Nested DA"}
linecolor = {"dscl":"k","conv":"tab:blue","lsb":'tab:orange',"nest":'tab:green'}

fig, ax = plt.subplots(figsize=[12,6],constrained_layout=True)
figs, sax = plt.subplots(figsize=[12,6],constrained_layout=True)
fig0, (ax00,ax01,ax02) = plt.subplots(nrows=3,figsize=[12,12],constrained_layout=True)
#fig1, (ax10,ax11,ax12) = plt.subplots(nrows=3,figsize=[12,12],constrained_layout=True)
figs0, (sax00,sax01,sax02) = plt.subplots(nrows=3,figsize=[12,12],constrained_layout=True)

tc = np.arange(na)+1 # cycles
t = tc / 4. # days
ns = 40 # spinup
ntime = na - ns
nt1 = ntime // 3
errors = {}
if ldscl:
    keys = ['dscl','conv','lsb','nest']
else:
    keys = ['conv','lsb','nest']
for key in keys:
    if key=='dscl':
        # downscaling
        if anl:
            f = dscldir / f"e_lam_{op}_{preGMpt}.txt"
            fs = dscldir / f"stda_lam_{op}_{preGMpt}.txt"
        else:
            f = dscldir / f"ef_lam_{op}_{preGMpt}.txt"
            fs = dscldir / f"stdf_lam_{op}_{preGMpt}.txt"
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
    e = np.loadtxt(f)
    if anl:
        print("{}, analysis RMSE = {}".format(key,np.mean(e[ns:])))
    else:
        print("{}, forecast RMSE = {}".format(key,np.mean(e[ns:])))
    stda = np.loadtxt(fs)
    ax.plot(t[ns:],e[ns:],c=linecolor[key],label=labels[key]+f'={np.mean(e[ns:]):.3f}')
    ax00.plot(t[ns:ns+nt1],e[ns:ns+nt1],c=linecolor[key],label=labels[key]+f'={np.mean(e[ns:]):.3f}')
    ax01.plot(t[ns+nt1:ns+2*nt1],e[ns+nt1:ns+2*nt1],c=linecolor[key])#,label=labels[key]+f'={np.mean(e[ns:]):.3f}')
    ax02.plot(t[ns+2*nt1:],e[ns+2*nt1:],c=linecolor[key])#,label=labels[key]+f'={np.mean(e[ns:]):.3f}')
    sax.plot(t[ns:],stda[ns:],c=linecolor[key],label=labels[key]+f'={np.mean(stda[ns:]):.3f}')
    sratio = stda/e
    sax00.plot(t[ns:ns+nt1],sratio[ns:ns+nt1],c=linecolor[key],label=labels[key]+f'={np.mean(sratio[ns:]):.3f}')
    sax01.plot(t[ns+nt1:ns+2*nt1],sratio[ns+nt1:ns+2*nt1],c=linecolor[key]) #,label=labels[key]+f'={np.mean(sratio):.3f}')
    sax02.plot(t[ns+2*nt1:],sratio[ns+2*nt1:],c=linecolor[key]) #,label=labels[key]+f'={np.mean(sratio):.3f}')
    errors[key] = e[ns:]
    ## autocorrelation length estimation using jackknife method
    #nwc, e_resample, t_resample = jackknife(e,t,key)
    #nt2 = t_resample.size // 3
    #ax10.plot(t_resample[:nt2],e_resample[:nt2],c=linecolor[key],label=labels[key]+f'={np.mean(e_resample):.3f}')
    #ax11.plot(t_resample[nt2:2*nt2],e_resample[nt2:2*nt2],c=linecolor[key])#,label=labels[key]+f'={np.mean(e[ns:]):.3f}')
    #ax12.plot(t_resample[2*nt2:],e_resample[2*nt2:],c=linecolor[key])#,label=labels[key]+f'={np.mean(e[ns:]):.3f}')
    #errors[key] = e_resample #e[ns:]
for ax1 in [ax,ax00,ax01,ax02]: #,ax10,ax11,ax12]:
    ax1.hlines([1.0],0,1,colors='k',ls='dotted',lw=2.5,transform=ax1.get_yaxis_transform())
for ax1 in [ax,ax00,ax01,ax02]: #,ax10,ax11,ax12]:
    ax1.set_ylim(0.0,2.0)
    ax1.set_ylabel('RMSE') #,title=op)
ax.set_xlim(10,250)
ax.set_xticks([10,40,70,100,130,160,190,220,250])
ax.set_xticks(t[ns-1:na:40],minor=True)
ax.grid()
ax.set_xlabel('days')
ax.legend(loc='upper left',bbox_to_anchor=(0.82,1.0),\
    title=f'{ptlong[pt]} time average')
ax02.set_xlabel('days')
ax00.legend(loc='upper left',bbox_to_anchor=(1.01,0.95),\
    title=f'{ptlong[pt]} time average')
#ax12.set_xlabel('days (resampling)')
#ax10.legend(loc='upper left',bbox_to_anchor=(1.01,0.95),\
#    title=f'{ptlong[pt]} time average\n(resampling)')
sax.set_xlim(t[ns],t[-1])
sax.set_ylabel('spread')
sax.set_xlabel('days')
sax.legend(loc='upper left',bbox_to_anchor=(0.82,1.0),\
    title=f'{ptlong[pt]} time average')
#sax.set_xlim(t[ns],t[-1])
for sax0 in [sax00,sax01,sax02]:
    sax0.hlines([1],0,1,colors='gray',transform=sax0.get_yaxis_transform(),zorder=0)
    sax0.set_ylim(0.0,2.0)
    sax0.set(ylabel='STD/RMSE') #,title=op)
sax02.set_xlabel('days')
sax00.legend(loc='upper left',bbox_to_anchor=(1.01,0.95),\
    title=ptlong[pt]+' time average \nof STD/RMSE')
if anl:
    fig.savefig(figpngdir/'{}_e_lam_{}_{}.png'.format(model,op,pt),dpi=300)
    fig0.savefig(figpngdir/'{}_e3_lam_{}_{}.png'.format(model,op,pt),dpi=300)
    #fig1.savefig(figpngdir/'{}_e_lam_resample_{}_{}.png'.format(model,op,pt),dpi=300)
    figs.savefig(figpngdir/'{}_stda_lam_{}_{}.png'.format(model,op,pt),dpi=300)
    figs0.savefig(figpngdir/'{}_ss3_lam_{}_{}.png'.format(model,op,pt),dpi=300)
    fig.savefig(figpdfdir/'{}_e_lam_{}_{}.pdf'.format(model,op,pt))
    fig0.savefig(figpdfdir/'{}_e3_lam_{}_{}.pdf'.format(model,op,pt))
    #fig1.savefig(figpdfdir/'{}_e_lam_resample_{}_{}.pdf'.format(model,op,pt))
    figs.savefig(figpdfdir/'{}_stda_lam_{}_{}.pdf'.format(model,op,pt))
    figs0.savefig(figpdfdir/'{}_ss3_lam_{}_{}.pdf'.format(model,op,pt))
else:
    fig0.savefig(figpngdir/'{}_ef_lam_{}_{}.png'.format(model,op,pt),dpi=300)
#    fig1.savefig(figpngdir/'{}_stdf_lam_{}_{}.png'.format(model,op,pt),dpi=300)
    fig0.savefig(figpdfdir/'{}_ef_lam_{}_{}.pdf'.format(model,op,pt))
#    fig1.savefig(figpdfdir/'{}_stdf_lam_{}_{}.pdf'.format(model,op,pt))
#plt.show()
plt.close(fig=fig)
plt.close(fig=fig0)
#plt.close(fig=fig1)
plt.close(fig=figs)
plt.close(fig=figs0)

# error statistics
fig, ax = plt.subplots(figsize=[6,8],constrained_layout=True)
for i,key in enumerate(errors.keys()):
    e = errors[key]
    ax.boxplot(e,positions=[i+1],meanline=True,showmeans=True)
    nout = np.sum(e>2.0)
    if nout>0:
        ax.text(i+0.9,0.95,f'{nout}',transform=ax.get_xaxis_transform(),\
            ha='center',size='x-small',weight='bold',color='r')
    emean = np.mean(e)
    emin = np.min(e)
    emax = np.max(e)
    estd = np.std(e)
    nbust = np.sum(e>1.0)
    print(f"{key}: min={emin:.3e} mean={emean:.3e} max={emax:.3e} std={estd:.3e}")
    print(f"# of bust = {nbust}")
    if key=='dscl':
        eref = e.copy()
    corr = np.correlate(eref,e)/np.linalg.norm(eref,ord=2)/np.linalg.norm(e,ord=2)
    print(f"corr between dscl = {corr[0]:.3e}")
ax.set_xticks(np.arange(1,len(errors.keys())+1))
ax.set_xticklabels(errors.keys())
ax.set_ylabel('RMSE')
ax.set_ylim(0.0,2.0)
ax.grid(axis='y')
ax.set_title(ptlong[pt])
fig.savefig(figpngdir/'{}_ebox_lam_{}_{}.png'.format(model,op,pt),dpi=300)
plt.show()
exit()

# bootstrap
fig, ax = plt.subplots(figsize=[10,6],constrained_layout=True)
cmap = plt.get_cmap('tab10')
i=0
thetas = {}
barlabels = {}
height = 0.9
for key in errors.keys():
    e = errors[key]
    ### calculate autocorrelation
    #ax.acorr(e,maxlags=e.size-1,usevlines=False,label=key)
    ## bootstrap sampling
    nb = 2000
    nparts = e.size
    nsample = e.size
    theta = []
    for n in range(nb):
        index = np.sort(np.random.choice(nparts,size=nsample,replace=True))
        esample = [e[i] for i in index]
        theta.append(np.mean(np.array(esample)))
    ## estimate confidence interval
    theta = np.sort(np.array(theta))
    alphalist = [5.0e-3,2.5e-2,5.0e-2] # 99, 95, 90%
    conf = []
    for alpha in alphalist:
        nlow = int(nb*alpha)+1
        nhigh = int(nb*(1-alpha))
        conf.append([theta[nlow],theta[nhigh]])
    label = labels[key] + f'\n99[{conf[0][0]:.3f},{conf[0][1]:.3f}]'\
        + f'\n95[{conf[1][0]:.3f},{conf[1][1]:.3f}]' \
        + f'\n90[{conf[2][0]:.3f},{conf[2][1]:.3f}]'
    ax.hist(theta,bins=20,histtype='bar',color=cmap(i),alpha=0.3,label=label)
    ax.vlines(conf[0],0,1,colors=cmap(i),ls='dashed',transform=ax.get_xaxis_transform(),zorder=0)
    mscale = 30
    y = height
    for j in range(len(conf)):
        c1 = conf[j]
        interval = FancyArrowPatch((c1[0],y),(c1[1],y),\
        arrowstyle='<|-|>',mutation_scale=mscale,color=cmap(i),transform=ax.get_xaxis_transform())
        ax.add_patch(interval)
        y-=0.03
        mscale-=10
    ax.text(np.mean(conf[0]),height+0.01,labels[key],ha='center',va='bottom',fontsize=14,transform=ax.get_xaxis_transform())
    thetas[key] = theta
    barlabels[key] = label
    i+=1
    height -= 0.15
ax.set_title(f'{ptlong[pt]} #replace={nb} #sample={e.size}')
ax.legend(loc='upper left',bbox_to_anchor=(1.01,1.0),fontsize=16,\
    title='confidence interval',title_fontsize=16)
fig.savefig(figdir/'{}_e_lam_bs_{}_{}.png'.format(model,op,pt),dpi=300)
#plt.show()
plt.close()

# bootstrap for difference
plt.rcParams['xtick.labelsize'] = 'medium'
plt.rcParams['ytick.labelsize'] = 'medium'
for i,k1 in enumerate(thetas.keys()):
    for j,k2 in enumerate(thetas.keys()):
        if k1==k2 or i>j: continue
        th1 = thetas[k1]
        th2 = thetas[k2]
        nb = 2000
        dth = []
        for n in range(nb):
            ii = np.random.choice(th1.size,size=1)
            jj = np.random.choice(th2.size,size=1)
            dth.append(th1[ii]-th2[jj])
        dth = np.sort(np.array(dth).ravel())
        conf=[]
        for alpha in alphalist:
            nlow = int(nb*alpha)+1
            nhigh = int(nb*(1-alpha))
            conf.append([dth[nlow],dth[nhigh]])
        title=f'{ptlong[pt]} ({labels[k1]}) - ({labels[k2]})'
        label = f'99[{conf[0][0]:.3f},{conf[0][1]:.3f}]'\
        + f'\n95[{conf[1][0]:.3f},{conf[1][1]:.3f}]' \
        + f'\n90[{conf[2][0]:.3f},{conf[2][1]:.3f}]'
        fig, ax = plt.subplots(figsize=[6,4],constrained_layout=True)
        ax.hist(dth,bins=20,histtype='bar',color=cmap(0),alpha=0.3,label=label)
        ax.vlines(conf[0],0,1,colors=cmap(0),ls='dashed',transform=ax.get_xaxis_transform(),zorder=0)
        xmin, xmax = ax.get_xlim()
        if xmin*xmax<0.0:
            ax.vlines([0],0,1,colors='k',ls='dotted',transform=ax.get_xaxis_transform(),zorder=0)
        mscale = 30
        y = 0.9
        for ii in range(len(conf)):
            c1 = conf[ii]
            interval = FancyArrowPatch((c1[0],y),(c1[1],y),\
            arrowstyle='<|-|>',mutation_scale=mscale,color=cmap(0),transform=ax.get_xaxis_transform())
            ax.add_patch(interval)
            y-=0.03
            mscale-=10
        ax.set_title(title,fontsize=17)
        ax.legend(loc='upper left',bbox_to_anchor=(0.85,1.0),fontsize=14,\
            title='confidence interval',title_fontsize=14)
        fig.savefig(figdir/'{}_ediff{}-{}_lam_bs_{}_{}.png'.format(model,k1,k2,op,pt),dpi=300)
        #plt.show()
        plt.close()
exit()

# t-test
import matplotlib as mpl
from scipy import stats
from plot_heatmap import heatmap, annotate_heatmap
methods = errors.keys()
nmethods = len(methods)
pmatrix = np.eye(nmethods)
tmatrix = np.eye(nmethods)
for i, m1 in enumerate(methods):
    for j, m2 in enumerate(methods):
        if m1==m2:
            tmatrix[i,j] = np.nan
            pmatrix[i,j] = np.nan
            continue
        e1 = errors[m1]
        e2 = errors[m2]
        res_ttest = stats.ttest_rel(e1,e2)
        tmatrix[i,j] = res_ttest.statistic
        pmatrix[i,j] = res_ttest.pvalue
        if pmatrix[i,j] < 1e-16:
            pmatrix[i,j] = 1e-10
        if tmatrix[i,j]>0.0:
            pmatrix[i,j] = pmatrix[i,j] * -1.0
        print(f"{m1},{m2} t-stat:{tmatrix[i,j]:.3f} pvalue:{pmatrix[i,j]:.3e}")
print("")
fig, ax = plt.subplots(figsize=[8,6])
cmap=plt.get_cmap("PiYG_r")
cl0 = cmap(cmap.N//2)
cl1 = cmap(cmap.N//4*3)
cl2a = cmap(cmap.N-1)
cl2b = cmap(0)
cl3 = cmap(cmap.N//4)
cl4 = cmap(cmap.N//2-1)
cdict = {
    'red':(
        (0.0, cl0[0], cl0[0]),
        (0.25, cl1[0], cl1[0]),
        (0.5, cl2a[0], cl2b[0]),
        (0.75, cl3[0], cl3[0]),
        (1.0, cl4[0], cl4[0]),
    ),
    'green':(
        (0.0, cl0[1], cl0[1]),
        (0.25, cl1[1], cl1[1]),
        (0.5, cl2a[1], cl2b[1]),
        (0.75, cl3[1], cl3[1]),
        (1.0, cl4[1], cl4[1]),
    ),
    'blue':(
        (0.0, cl0[2], cl0[2]),
        (0.25, cl1[2], cl1[2]),
        (0.5, cl2a[2], cl2b[2]),
        (0.75, cl3[2], cl3[2]),
        (1.0, cl4[2], cl4[2]),
    ),
    'alpha':(
        (0.0, cl0[3], cl0[3]),
        (0.25, cl1[3], cl1[3]),
        (0.5, cl2a[3], cl2b[3]),
        (0.75, cl3[3], cl3[3]),
        (1.0, cl4[3], cl4[3]),
    ),
}
mycmap = mpl.colors.LinearSegmentedColormap('GrPi',cdict)
norm = mpl.colors.BoundaryNorm([-0.1,-0.05,-0.01,0.0,0.01,0.05,0.1],8,extend="both")
sigrates = ['<90%','>90%','>95%','>99%','>99%','>95%','>90%','<90%']
fmt = mpl.ticker.FuncFormatter(lambda x, pos: sigrates[norm(x)])
im, _ = heatmap(pmatrix,methods,methods,ax=ax,\
    cmap=mycmap.resampled(8), norm=norm,\
    cbar_kw=dict(ticks=[-0.075,-0.03,-0.005,0.005,0.03,0.075],format=fmt,extend="both"),\
    cbarlabel="significance")
annotate_heatmap(im,data=tmatrix,thdata=np.abs(pmatrix),\
    valfmt="{x:.2f}",fontweight="bold",\
    threshold=0.05,textcolors=("white","black"))
ax.set_title(f"t-test for {ptlong[pt]} {op}: RMSE row-col")
fig.tight_layout()
if anl:
    fig.savefig(figdir/"{}_e_t-test_for_lam_{}_{}.png".format(model, op, pt),dpi=300)
else:
    fig.savefig(figdir/"{}_ef_t-test_for_lam_{}_{}.png".format(model, op, pt),dpi=300)
plt.show()