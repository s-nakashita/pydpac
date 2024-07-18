import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
from pathlib import Path
import sys
plt.rcParams['font.size'] = 16

model = 'l96'
pt = 'letkf'
wdir = Path(f'/Volumes/dandelion/pyesa/data/{model}')
nens = 8
laglist = [0,1,2,3]
ftlist = [0,24,48,72] #,96]
offsets = [0,4,8,12] #,16]
nft = len(ftlist)
icyc = 50
ncyc = 1000

datadir = wdir / f'extfcst_m{nens}'
xa = np.load(datadir/f"{model}_xf00_linear_{pt}.npy")
xfall = np.load(datadir/f"{model}_ufext_linear_{pt}.npy")
print(xa.shape)
print(xfall.shape)

fedict = {}
fsdict = {}
for lag in laglist:
    felist = []
    fslist = []
    for i,ft in enumerate(ftlist):
        xf = xfall[icyc:ncyc,offsets[i],:,:]
        for j in range(1,lag+1):
            xf = np.concatenate([xf,xfall[icyc-2*j:ncyc-2*j,offsets[i]+2*j,:,:]],axis=-1)
        print(xf.shape)
        xd = xf - xa[icyc+offsets[i]:ncyc+offsets[i],:,:].mean(axis=2)[:,:,None]
        felist.append(np.std(np.sqrt(np.mean(xd**2,axis=1)),axis=1))
        fslist.append(np.mean(np.std(xf,axis=2),axis=1))
    fedict[lag] = felist
    fsdict[lag] = fslist

cmap = plt.get_cmap('tab20')
fig, (axfe, axfs) = plt.subplots(ncols=2,sharey=True,figsize=[10,6])
nbox = len(fedict)
width = 0.75 / nbox
xoffset = 0.5 * width * (nbox - 1)
xaxis = np.arange(1,nft+1) - xoffset
for lag in laglist:
    icol = laglist.index(lag)
    fe = np.array(fedict[lag]).reshape(nft,-1)
    fs = np.array(fsdict[lag])
    bplot1 = axfe.boxplot(fe.T,positions=xaxis,widths=width,patch_artist=True,whis=(0,100),zorder=0)
    for patch in bplot1['boxes']:
        patch.set_facecolor(cmap(2*icol+1))
    axfe.plot(np.arange(1,nft+1),fe.mean(axis=1),c=cmap(2*icol),label=f'{nens}*{lag+1} (-{lag*12}h)')
    bplot2 = axfs.boxplot(fs.T,positions=xaxis,widths=width,patch_artist=True,whis=(0,100),zorder=0)
    for patch in bplot2['boxes']:
        patch.set_facecolor(cmap(2*icol+1))
    axfs.plot(np.arange(1,nft+1),fs.mean(axis=1),c=cmap(2*icol),label=f'{nens}*{lag+1} (-{lag*12}h)')
    xaxis = xaxis + width
for ax in [axfe,axfs]:
    ax.set_xticks(np.arange(1,nft+1))
    ax.set_xticklabels(ftlist)
    ax.grid()
axfe.set(xlabel='forecast hours',title='Ensemble standard deviation of \nforecast error against analysis')
axfe.legend()
axfs.set(xlabel='forecast hours',title='Ensemble spread')
axfs.legend()
fig.savefig(wdir/f"{model}_fe+fs_tlag12h_linear_{pt}.png")
plt.show()