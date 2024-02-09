import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
import pandas as pd
from pathlib import Path

datadir1 = Path('work/baxter11')
datadir2 = Path('work/baxter11_en')

nobslist = [31, 16, 11, 8]
nenslist = [40,80,120,160,320,640,960]

plt.rcParams['font.size'] = 16
cmap = plt.get_cmap('tab10')
for k in range(19):
    if k==0:
        title='RMSE'
    else:
        title=f'Absolute error for wave number = {k}'
    fig, [ax1,ax2] = plt.subplots(nrows=2,figsize=[12,8],\
        sharex=True,constrained_layout=True)
    lines = []
    legends = []
    for i,nobs in enumerate(nobslist):
        print(f"nobs={nobs}")
        lamlist = []
        lam_nestlist = []
        difflist = []
        markers = []
        # 3DVar
        csvfile = f't-test_nobs{nobs}.csv'
        df = pd.read_csv(datadir1/csvfile,header=1,comment='#')
        #print(df.loc[k,'95%'])
        diff = df.loc[k,'LAM'] - df.loc[k,'LAM_nest']
        lamlist.append(df.loc[k,'LAM'])
        lam_nestlist.append(df.loc[k,'LAM_nest'])
        difflist.append(diff)
        mk = ''
        if df.loc[k,'95%']:
            mk = '^'
        if df.loc[k,'99%']:
            mk = 'o'
        markers.append(mk)
        #EnVar
        for nens in nenslist:
            print(f"nens={nens}")
            csvfile = f't-test_nobs{nobs}nmem{nens}.csv'
            df = pd.read_csv(datadir2/csvfile,header=1,comment='#')
            #print(df.loc[k,'95%'])
            diff = df.loc[k,'LAM'] - df.loc[k,'LAM_nest']
            lamlist.append(df.loc[k,'LAM'])
            lam_nestlist.append(df.loc[k,'LAM_nest'])
            difflist.append(diff)
            mk = ''
            if df.loc[k,'95%']:
                mk = '^'
            if df.loc[k,'99%']:
                mk = 'o'
            markers.append(mk)
        ax1.plot(np.arange(len(lamlist)),lamlist,c=cmap(i),ls='dashed')
        ax1.plot(np.arange(len(lam_nestlist)),lam_nestlist,c=cmap(i))
        ax2.plot(np.arange(len(difflist)),difflist)
        legends.append(f'nobs={nobs}')
        lines.append(Line2D([0],[0],color=cmap(i),lw=2.))
        for j, d in enumerate(difflist):
            mk = markers[j]
            ax2.plot([j],[d],marker=mk,lw=0.0,c=cmap(i))
    legends.append('95%')
    lines.append(Line2D([0],[0],color='k',lw=0.,marker='^'))
    legends.append('99%')
    lines.append(Line2D([0],[0],color='k',lw=0.,marker='o'))
    xlabels = ['3DVar']
    for nens in nenslist:
        xlabels.append(f'EnVar{nens}')
    ax2.set_xticks(np.arange(len(xlabels)))
    ax2.set_xticklabels(xlabels)
    #ax1.set_title('LAM, LAM_nest')
    ax1.legend(\
       [Line2D([0],[0],color='k',ls='dashed',lw=2),\
        Line2D([0],[0],color='k',lw=2)],\
       ['LAM','LAM_nest'],loc='upper left',bbox_to_anchor=(1.01,1.0))
    ax2.set_title('LAM - LAM_nest')
    ax2.legend(lines,legends,loc='upper left',bbox_to_anchor=(1.01,1.0))
    fig.suptitle(title)
    fig.savefig(datadir1/f'comp_k{k}.png',dpi=300)
    fig.savefig(datadir1/f'comp_k{k}.pdf')
    #plt.show()
    plt.close()