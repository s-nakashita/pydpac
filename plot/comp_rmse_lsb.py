import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['legend.title_fontsize'] = 24

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
pt = 'var' #var or envar
if len(sys.argv)>4:
    pt = sys.argv[4]
anl = True
if len(sys.argv)>5:
    anl = (sys.argv[5]=='T')

datadir = Path(f'/Volumes/FF520/nested_envar/data/{model}')
datadir = Path(f'../work/{model}')
preGMpt = 'envar'
ldscl=True
#obsloc = ''
#obsloc = '_partiall'
#obsloc = '_partialc'
#obsloc = '_partialr'
obsloc = '_partialm'
dscldir = datadir / 'var_vs_envar_dscl_m80obs30'
lsbdir  = datadir / f'var_vs_envar_lsb_preGM{obsloc}_m80obs30'
lamdir  = datadir / f'var_vs_envar_shrink_dct_preGM{obsloc}_m80obs30'
#if ldscl:
#    figdir = datadir
#else:
figdir = lsbdir

ptlong = {"envar":"EnVar","var":"3DVar"}
labels = {"conv":"DA", "lsb":"DA+LSB", "nest":"Nested DA"}
linecolor = {"conv":"tab:blue","lsb":'tab:orange',"nest":'tab:green'}

fig0, ax0 = plt.subplots(figsize=[12,6],constrained_layout=True)
fig1, ax1 = plt.subplots(figsize=[12,6],constrained_layout=True)

tc = np.arange(na)+1 # cycles
t = tc / 4. # days
ns = 40 # spinup
errors = {}
if ldscl:
    # downscaling
    if anl:
        f = dscldir / f"e_lam_{op}_{preGMpt}.txt"
    else:
        f = dscldir / f"ef_lam_{op}_{preGMpt}.txt"
    if not f.exists():
        print("not exist {}".format(f))
        exit()
    e_dscl = np.loadtxt(f)
    if anl:
        print("dscl, analysis RMSE = {}".format(np.mean(e_dscl[ns:])))
        f = dscldir / f"stda_lam_{op}_{preGMpt}.txt"
    else:
        print("dscl, forecast RMSE = {}".format(np.mean(e_dscl[ns:])))
        f = dscldir / f"stdf_lam_{op}_{preGMpt}.txt"
    stda_dscl = np.loadtxt(f)
    ax0.plot(t,e_dscl,c='k',label=f'downscaling={np.mean(e_dscl[ns:]):.3f}')
    ax1.plot(t,stda_dscl,c='k',label=f'downscaling={np.mean(stda_dscl[ns:]):.3f}')
    errors['dscl'] = e_dscl[ns:]
for key in ['conv','lsb','nest']:
    if key=='conv':
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
    ax0.plot(t,e,c=linecolor[key],label=labels[key]+f'={np.mean(e[ns:]):.3f}')
    ax1.plot(t,stda,c=linecolor[key],label=labels[key]+f'={np.mean(stda[ns:]):.3f}')
    errors[key] = e[ns:]
for ax in [ax0,ax1]:
    ax.hlines([1.0],0,1,colors='gray',ls='dotted',transform=ax.get_yaxis_transform())
    ax.set_xlim(t[ns],t[-1])
ax0.set_ylim(0.0,1.5)
ax1.set_ylim(0.0,1.5)
ax0.set(xlabel='days',ylabel='RMSE') #,title=op)
ax1.set(xlabel='days',ylabel='STD') #,title=op)
ax0.legend(loc='upper left',bbox_to_anchor=(1.01,0.95),\
    title=f'{ptlong[pt]} time average')
ax1.legend(loc='upper left',bbox_to_anchor=(1.01,0.95),\
    title=ptlong[pt])
if anl:
    fig0.savefig(figdir/'{}_e_lam_{}_{}.png'.format(model,op,pt),dpi=300)
    fig1.savefig(figdir/'{}_stda_lam_{}_{}.png'.format(model,op,pt),dpi=300)
    fig0.savefig(figdir/'{}_e_lam_{}_{}.pdf'.format(model,op,pt))
    fig1.savefig(figdir/'{}_stda_lam_{}_{}.pdf'.format(model,op,pt))
else:
    fig0.savefig(figdir/'{}_ef_lam_{}_{}.png'.format(model,op,pt),dpi=300)
    fig1.savefig(figdir/'{}_stdf_lam_{}_{}.png'.format(model,op,pt),dpi=300)
    fig0.savefig(figdir/'{}_ef_lam_{}_{}.pdf'.format(model,op,pt))
    fig1.savefig(figdir/'{}_stdf_lam_{}_{}.pdf'.format(model,op,pt))
plt.show()
plt.close()

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