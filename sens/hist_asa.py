import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 16
import xarray as xr
from scipy.stats import norm, entropy
from pathlib import Path
import argparse

datadir = Path('data')

cmap = plt.get_cmap('tab10')
enasas = ['minnorm','diag','ridge','pcr','pls']
colors = {'asa':cmap(0),'minnorm':cmap(1),'diag':cmap(2),'pcr':cmap(3),'ridge':cmap(4),'pls':cmap(5)}
markers = {'asa':'*','minnorm':'o','diag':'v','pcr':'s','ridge':'P','pls':'X'}
marker_style=dict(markerfacecolor='none')
vtlist = [24,48,72,96]
nelist = [8, 16,24,32,40]

parser = argparse.ArgumentParser()
parser.add_argument("-m","--metric",type=str,default="",\
    help="forecast metric type")
argsin = parser.parse_args()
metric = argsin.metric
figdir = Path(f"fig{metric}")
if not figdir.exists(): figdir.mkdir()

# load results
nensbase = 8
for vt in vtlist:
    success = dict()
    failure = dict()
    ds_asa = xr.open_dataset(datadir/f'asa{metric}_vt{vt}nens{nensbase}.nc')
    print(ds_asa)
    f = open(figdir/f'nsucess_failure_vt{vt}.txt',mode='w')
    nsuccess = np.sum(ds_asa.res_nl.values<=-0.5)
    nfailure = np.sum(ds_asa.res_nl.values>0.0)
    f.write(f"asa: #success={nsuccess} #failure={nfailure}\n")
    success['asa'] = nsuccess / ds_asa.res_nl.size
    failure['asa'] = nfailure / ds_asa.res_nl.size

    kld_dict = {}
    kld_gauss_dict = {}
    for key in enasas:
        success[key] = []
        failure[key] = []
        kld_dict[key] = []
        kld_gauss_dict[key] = []
    for ne in nelist:
        figr, axr = plt.subplots(figsize=[8,6],constrained_layout=True)
        figc, axc = plt.subplots(figsize=[8,6],constrained_layout=True)
        key = 'asa'
        r = ds_asa.res_nl.values
        rmean1 = r.mean()
        rstd1 = r.std()
        label=key #+'\n'+r'$\mu$='+f'{rmean1:.2f}, '+r'$\sigma$='+f'{rstd1:.2f}'
        if metric=='':
            x = np.linspace(-1.0,1.0,51)
        else:
            x = np.linspace(-0.08,0.02,51)
        n_asa, bins, _ = axr.hist(r,bins=x,histtype='step',density=True,color=colors[key],label=label) #,alpha=0.3)
        print(np.sum(n_asa*np.diff(x))) # \int p(x)dx
        e = entropy(n_asa*np.diff(x)) # Shannon entropy
        print(e)
        #axr.plot(x,norm.pdf(x,loc=rmean1,scale=rstd1),c=colors[key],label=label)
        for key in enasas:
            ds = xr.open_dataset(datadir/f'{key}{metric}_vt{vt}nens{ne}.nc')
            r = ds.res_nl.values
            rmean2 = r.mean()
            rstd2 = r.std()
            label=key #+'\n'+r'$\mu$='+f'{rmean2:.2f}, '+r'$\sigma$='+f'{rstd2:.2f}'
            n_enasa, bins, _ = axr.hist(r,bins=x,histtype='step',density=True,color=colors[key],label=label) #,alpha=0.3)
            #axr.plot(x,norm.pdf(x,loc=rmean2,scale=rstd2),c=colors[key],label=label)
            kld = entropy(n_asa*np.diff(x),n_enasa*np.diff(x))
            print(f"KLD(asa||{key})={kld}")
            kld_dict[key].append(kld)
            kld_g = np.log(rstd2/rstd1) - 0.5 + (rstd1**2 + rmean1**2 + rmean2**2 - 2*rmean1*rmean2)/2/rstd2/rstd2
            print(f"KLD_gauss(asa||{key})={kld_g}")
            kld_gauss_dict[key].append(kld_g)
            nsuccess = np.sum(r<=-0.5)
            nfailure = np.sum(r>0.0)
            f.write(f"{key} ne{ne}: #success={nsuccess} #failure={nfailure}\n")
            success[key].append(nsuccess/r.size)
            failure[key].append(nfailure/r.size)
            c = ds.corrdJ.values
            cmean = c.mean()
            cstd = c.std()
            label=key+'\n'+r'$\mu$='+f'{cmean:.2f}, '+r'$\sigma$='+f'{cstd:.2f}'
            axc.hist(c,bins=50,histtype='bar',density=True,color=colors[key],alpha=0.3)
            x_c = np.linspace(-1.0,1.0,51)
            axc.plot(x_c,norm.pdf(x_c,loc=cmean,scale=cstd),c=colors[key],label=label)
        axr.set_xlabel('nonlinear forecast response')
        axc.set_xlabel('spatial correlation')
        if metric=='':
            axr.legend(loc='upper right') #,bbox_to_anchor=(1.0,1.0))
        else:
            axr.legend(loc='upper left') #,bbox_to_anchor=(1.0,1.0))
        axc.legend(loc='upper left') #,bbox_to_anchor=(1.0,1.0))
        figr.suptitle(f'FT{vt}, {ne} member')
        figc.suptitle(f'FT{vt}, {ne} member')
        figr.savefig(figdir/f'hist_resnl_vt{vt}ne{ne}.png',dpi=300)
        figc.savefig(figdir/f'hist_corr_vt{vt}ne{ne}.png',dpi=300)
        #plt.show()
        plt.close(fig=figr)
        plt.close(fig=figc)
    f.close()

    fig, (ax1,ax2) = plt.subplots(ncols=2,sharey=True,figsize=[10,6],constrained_layout=True)
    #ax2 = ax1.twinx()
    ax1.hlines([success['asa']],0,1,colors=colors['asa'],\
        ls='solid',transform=ax1.get_yaxis_transform(),label='asa')
    ax2.hlines([failure['asa']],0,1,colors=colors['asa'],\
        ls='solid',transform=ax2.get_yaxis_transform(),label='asa')
    for key in enasas:
        ax1.plot(nelist,success[key],c=colors[key],marker=markers[key],label=key)
        ax2.plot(nelist,failure[key],c=colors[key],marker=markers[key],label=key)#,\
            #markerfacecolor='none',markeredgecolor=colors[key],ls='dashed')
    ax2.legend(loc='upper left',bbox_to_anchor=(1.0,1.0))
    ax1.set_title('success ratio')
    ax2.set_title('failure ratio')
    for ax in [ax1,ax2]:
        ax.set_xticks(nelist)
        ax.set_xticklabels([f'mem{ne}' for ne in nelist],fontsize=14)
    fig.suptitle(f'FT{vt}')
    fig.savefig(figdir/f'nsuccess_failure_vt{vt}.png',dpi=300)
    plt.show()
    plt.close(fig=fig)

    fig, ax = plt.subplots(figsize=[8,6],constrained_layout=True)
    for key in kld_dict.keys():
        ax.plot(nelist,kld_dict[key],c=colors[key],marker=markers[key],label=key)
        ax.plot(nelist,kld_gauss_dict[key],c=colors[key],marker=markers[key],\
            markerfacecolor='none',markeredgecolor=colors[key],ls='dashed')#,label=key+', gauss')
    ax.legend()
    ax.set_yscale('log')
    ax.set_ylabel('KL divergence of forecast response')
    ax.set_xticks(nelist)
    ax.set_xticklabels([f'mem{ne}' for ne in nelist])
    fig.suptitle(f'FT{vt}')
    fig.savefig(figdir/f'kld_resnl_vt{vt}.png',dpi=300)
    #plt.show()
    plt.close(fig=fig)
    #exit()