import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from pathlib import Path 
import argparse
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['axes.labelsize'] = 14

datadir = Path('/Volumes/FF520/pyesa/adata/l96')
figdir = Path('fig/res')

cmap = plt.get_cmap('tab10')
enasas = ['minnorm','diag','ridge','pcr','pls']
colors = {'asa':cmap(0),'minnorm':cmap(1),'diag':cmap(2),'pcr':cmap(3),'ridge':cmap(4),'pls':cmap(5)}
markers = {'asa':'*','minnorm':'o','diag':'v','pcr':'s','ridge':'P','pls':'X'}
ms = {'asa':8,'minnorm':5,'diag':5,'pcr':5,'ridge':5,'pls':5}
marker_style=dict(markerfacecolor='none')

parser = argparse.ArgumentParser()
parser.add_argument("-s","--sample",type=str,default="all",help="sample (all, near, far)")
argsin = parser.parse_args()
sample = argsin.sample

methods = ['minnorm','ridge','pcr','pls']
nmethod = len(methods)

# load data
keys = ['rmsd'] #'aspect','slope',
for key in keys:
    data = pd.read_csv(datadir/f'{sample}_mul-mul_{key}.csv',index_col=['FT','member'])
    ft = data.index.get_level_values('FT')
    ft = ft[~ft.duplicated()]
    member = data.index.get_level_values('member')
    member = member[~member.duplicated()]
    print(ft,len(ft))
    print(member,len(member))
    fig, axs = plt.subplots(ncols=4,nrows=1,figsize=[12,4],sharex=True,sharey=True,constrained_layout=True)
    for method in methods:
        d2d = data[method].values.reshape(len(ft),len(member))
        for j in range(len(ft)):
            f = ft[j]
            ax = axs.flatten()[j]
            ax.set_title(f'FT{f}')
            if key=='slope':
                y = d2d[j,] - 45.0
                if j==0:
                    ax.set_ylabel('slope - 45.0 (degree)')
            else:
                y = d2d[j,]
                if j==0:
                    if key=='rmsd':
                        ax.set_ylabel(key.upper())
                    else:
                        ax.set_ylabel(key)
            ax.plot(member,y,c=colors[method],marker=markers[method],ms=ms[method],label=method,**marker_style)
    for j, ax in enumerate(axs.flatten()):
        if j>=len(ft):
            ax.remove()
        else:
            ax.set_xticks(member)
            ax.set_xticklabels([f'mem{m}' if m==8 else f'{m}' for m in member])
            ax.grid()
            if j==0:
                ax.legend()
    fig.suptitle(f'{sample} multi-multi',fontsize=18)
    fig.savefig(figdir/f'{sample}_mul-mul_{key}.png',dpi=300)
    plt.show()
    plt.close()
