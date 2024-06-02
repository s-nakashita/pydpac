import os
import sys
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])

datadir = Path(f'/Volumes/FF520/nested_envar/data/{model}')
preGMpt = 'envar'
dscldir = datadir / 'var_vs_envar_dscl_m80obs30'
lamdir  = datadir / 'var_vs_envar_preGM_m80obs30'

perts = ["envar", "envar_nest","var","var_nest"]
labels = {"envar":"EnVar", "envar_nest":"Nested EnVar", "var":"3DVar", "var_nest":"Nested 3DVar"}
linecolor = {"envar":'tab:orange',"envar_nest":'tab:green',"var":"tab:olive","var_nest":"tab:brown"}

fig0, ax0 = plt.subplots(figsize=[12,6],constrained_layout=True)
fig1, ax1 = plt.subplots(figsize=[12,6],constrained_layout=True)

tc = np.arange(na)+1 # cycles
t = tc / 4. # days
ns = 40 # spinup
errors = {}
# downscaling
f = dscldir / f"e_lam_{op}_{preGMpt}.txt"
if not f.exists():
    print("not exist {}".format(f))
    exit()
e_dscl = np.loadtxt(f)
print("dscl, analysis RMSE = {}".format(np.mean(e_dscl[ns:])))
f = dscldir / f"stda_lam_{op}_{preGMpt}.txt"
stda_dscl = np.loadtxt(f)
ax0.plot(t,e_dscl,c='k',label=f'downscaling={np.mean(e_dscl[ns:]):.3f}')
ax1.plot(t,stda_dscl,c='k',label=f'downscaling={np.mean(stda_dscl[ns:]):.3f}')
errors['dscl'] = e_dscl[ns:]
for pt in perts:
    f = lamdir / f"e_lam_{op}_{pt}.txt"
    if not f.exists():
        print("not exist {}".format(f))
        continue
    e = np.loadtxt(f)
    print("{}, analysis RMSE = {}".format(pt,np.mean(e[ns:])))
    f = lamdir / f"stda_lam_{op}_{pt}.txt"
    stda = np.loadtxt(f)
    ax0.plot(t,e,c=linecolor[pt],label=labels[pt]+f'={np.mean(e[ns:]):.3f}')
    ax1.plot(t,stda,c=linecolor[pt],label=labels[pt]+f'={np.mean(stda[ns:]):.3f}')
    errors[pt] = e[ns:]
for ax in [ax0,ax1]:
    ax.hlines([1.0],0,1,colors='gray',ls='dotted',transform=ax.get_yaxis_transform())
    ax.set_xlim(t[ns],t[-1])
ax0.set_ylim(0.0,1.5)
ax1.set_ylim(0.0,1.5)
ax0.set(xlabel='days',ylabel='RMSE') #,title=op)
ax1.set(xlabel='days',ylabel='STD') #,title=op)
ax0.legend(loc='upper left',bbox_to_anchor=(1.01,0.95),\
    title='Time average')
ax1.legend(loc='upper left',bbox_to_anchor=(1.01,0.95))
fig0.savefig(datadir/'{}_e_lam_{}.png'.format(model,op),dpi=300)
fig1.savefig(datadir/'{}_stda_lam_{}.png'.format(model,op),dpi=300)
fig0.savefig(datadir/'{}_e_lam_{}.pdf'.format(model,op))
fig1.savefig(datadir/'{}_stda_lam_{}.pdf'.format(model,op))
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
        print(f"{m1},{m2} t-stat:{res_ttest.statistic:.3f} pvalue:{res_ttest.pvalue:.3e}")
        tmatrix[i,j] = res_ttest.statistic
        pmatrix[i,j] = res_ttest.pvalue
        if tmatrix[i,j]>0.0:
            if pmatrix[i,j] < 1e-16:
                pmatrix[i,j] = 1e-16
            pmatrix[i,j] = pmatrix[i,j] * -1.0
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
ax.set_title(f"t-test for LAM {op}: RMSE row-col")
fig.tight_layout()
fig.savefig(datadir/"{}_e_t-test_for_lam_{}.png".format(model, op),dpi=300)
plt.show()