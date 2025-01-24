import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path 
import argparse
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.labelsize'] = 14

datadir = Path('/Volumes/FF520/pyesa/adata/l96')
figdir = Path('fig/res_hess')

cmap = plt.get_cmap('tab10')
enasas = ['minnorm','diag','ridge','pcr','pls']
colors = {'asa':cmap(0),'minnorm':cmap(1),'diag':cmap(2),'pcr':cmap(3),'ridge':cmap(4),'pls':cmap(5)}
markers = {'asa':'*','minnorm':'o','diag':'v','pcr':'s','ridge':'P','pls':'X'}
ms = {'asa':8,'minnorm':5,'diag':5,'pcr':5,'ridge':5,'pls':5}
marker_style=dict(markerfacecolor='none')

parser = argparse.ArgumentParser()
parser.add_argument("-e","--ens",action='store_true',\
    help="ensemble estimated A")
parser.add_argument("-l","--loc",type=int,default=None,\
    help="localization radius for ensemble A")
argsin = parser.parse_args()
lens = argsin.ens
rloc = argsin.loc
if lens:
    if rloc is not None:
        csvname = f'res_hessens_loc{rloc}'
    else:
        csvname = 'res_hessens'
else:
    csvname = 'res_hess'

methods = ['asa','minnorm','ridge','pcr','pls'] #
nmethod = len(methods)

# load data
keys = ['estp_mean','estm_mean','calcp_mean','calcm_mean','rmsd_p','rmsd_m']
datadict = dict()
for key in keys:
    dataref = pd.read_csv(datadir/f'res_hess_{key}.csv')
    data = pd.read_csv(datadir/f'{csvname}_{key}.csv',index_col=['FT','member'])
    ft = data.index.get_level_values('FT')
    ft = ft[~ft.duplicated()]
    member = data.index.get_level_values('member')
    member = member[~member.duplicated()]
    print(ft,len(ft))
    print(member,len(member))
    datadict[key] = dict()
    for method in methods:
        if method=='asa':
            datadict[key][method] = dataref[method].values.reshape(len(ft),len(member))
        else:
            datadict[key][method] = data[method].values.reshape(len(ft),len(member))
marker_style=dict(markerfacecolor='none')
# mean
figm, axsm = plt.subplots(ncols=2,nrows=2,figsize=[8,6],constrained_layout=True)
# rmsd
figr, axsr = plt.subplots(ncols=2,nrows=2,figsize=[8,6],constrained_layout=True)
# rmsd / mean
fig, axs = plt.subplots(ncols=2,nrows=2,figsize=[8,6],constrained_layout=True)
lines = []
labels = []
for method in methods:
    for j in range(len(member)-1):
        mem = member[j]
        ax = axs.flatten()[j]
        ax.set_title(f'mem{mem}')
        axm = axsm.flatten()[j]
        axm.set_title(f'mem{mem}')
        axr = axsr.flatten()[j]
        axr.set_title(f'mem{mem}')
        # mean
        #lin_p = datadict['estp_mean'][method][j,]
        nln_p = datadict['calcp_mean'][method][:,j]
        #lin_m = datadict['estm_mean'][method][j,]
        nln_m = datadict['calcm_mean'][method][:,j]
        # rmsd
        rmsd_p = datadict['rmsd_p'][method][:,j]
        rmsd_m = datadict['rmsd_m'][method][:,j]
        marker_style.update(markerfacecolor=colors[method],markeredgecolor='k',markeredgewidth=0.5)
        axm.plot(ft,nln_p,c=colors[method],marker='$+$',**marker_style)
        axr.plot(ft,rmsd_p,c=colors[method],marker='$+$',**marker_style)
        ax.plot(ft,rmsd_p/nln_p,c=colors[method],marker='$+$',**marker_style)
        marker_style.update(markerfacecolor=colors[method],markeredgecolor='gray',markeredgewidth=0.3)
        axm.plot(ft,-1.0*nln_m,c=colors[method],ls='dotted',marker='$-$',**marker_style)
        axr.plot(ft,rmsd_m,c=colors[method],ls='dotted',marker='$-$',**marker_style)
        ax.plot(ft,-1.0*rmsd_m/nln_m,c=colors[method],ls='dotted',marker='$-$',**marker_style)
        if j==0:
            lines.append(Line2D([0],[0],c=colors[method]))
            if method=='asa' and lens:
                labels.append(method+r' ($\mathbf{A}_\mathrm{est}$)')
            else:
                labels.append(method)
for j, ax in enumerate(axs.flatten()):
    axm = axsm.flatten()[j]
    axr = axsr.flatten()[j]
    ymin, ymax = axm.get_ylim()
    axr.set_ylim(ymin,ymax)
    for ax1 in [ax,axm,axr]:
        if j>=len(member)-1:
            #ax1.remove()
            ax1.set_frame_on(False)
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.legend(lines,labels,loc='center')
        else:
            ax1.set_xticks(ft)
            ax1.set_xticklabels([f'FT{f}' for f in ft])
            ax1.grid()
            if j==0:
                ax1.legend(lines,labels,fontsize=14)

if lens:
    title2=r' with $\mathbf{A}_\mathrm{ens}$'
    if rloc is not None:
        title2=title2+r' $r_\mathrm{loc}$='+f'{rloc}'
else:
    title2=''
figm.suptitle(r'$\overline{|\Delta J(\Delta x_{0}^{*})|}$ (nonlinear)'+title2,fontsize=18)
figr.suptitle(r'RMSD of $\Delta J(\Delta x_{0}^{*})$ between linear and nonlinear'+title2,fontsize=18)
fig.suptitle(r'RMSD/$\overline{|\Delta J(\Delta x_{0}^{*})|}$'+title2,fontsize=18)
figm.savefig(figdir/f'{csvname}_calc_mean.png',dpi=300)
figr.savefig(figdir/f'{csvname}_rmsd.png',dpi=300)
fig.savefig(figdir/f'{csvname}_rmsd_o_calc_mean.png',dpi=300)
plt.show()
plt.close()
