import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
from pathlib import Path
import sys
plt.rcParams['font.size'] = 16

model = 'l96'
pt = 'letkf'
wdir = Path(f'/Volumes/dandelion/pyesa/data/{model}')
nenslist = [8,12,16,20,24,28,32,36] #,40,80]
ftlist = [0,24,48,72,96,120]
offsets = [0,4,8,12,16,20]
nft = len(ftlist)
icyc = 50
ncyc = 1000

fedict = {}
fsdict = {}
for nens in nenslist:
    datadir = wdir / f'extfcst_m{nens}'
    e = np.loadtxt(datadir/f"e_linear_{pt}.txt")
    print(f"{nens} member, averaged RMSE against nature = {e[icyc:ncyc].mean():.4f}")
    xa = np.load(datadir/f"{model}_xf00_linear_{pt}.npy")
    felist = []
    fslist = []
    for ft in ftlist:
        xf = np.load(datadir/f"{model}_xf{ft:02d}_linear_{pt}.npy")
        xd = xf[icyc:ncyc,:,:].mean(axis=2) - xa[icyc:ncyc,:,:].mean(axis=2)
        felist.append(np.sqrt(np.mean(xd**2,axis=1)))
        fslist.append(np.mean(np.std(xf[icyc:ncyc],axis=2),axis=1))
    fedict[nens] = felist
    fsdict[nens] = fslist

cmap = plt.get_cmap('tab20')
fig, (axfe, axfs) = plt.subplots(ncols=2,sharey=True,figsize=[10,6])
nbox = len(fedict)
width = 0.75 / nbox
xoffset = 0.5 * width * (nbox - 1)
xaxis = np.arange(1,nft+1) - xoffset
for nens in nenslist:
    icol = nenslist.index(nens)
    fe = np.array(fedict[nens])
    fs = np.array(fsdict[nens])
    bplot1 = axfe.boxplot(fe.T,positions=xaxis,widths=width,patch_artist=True,whis=(0,100))
    for patch in bplot1['boxes']:
        patch.set_facecolor(cmap(2*icol+1))
    axfe.plot(np.arange(1,nft+1),fe.mean(axis=1),c=cmap(2*icol),label=f'{nens}')
    bplot2 = axfs.boxplot(fs.T,positions=xaxis,widths=width,patch_artist=True,whis=(0,100))
    for patch in bplot2['boxes']:
        patch.set_facecolor(cmap(2*icol+1))
    axfs.plot(np.arange(1,nft+1),fs.mean(axis=1),c=cmap(2*icol),label=f'{nens}')
    xaxis = xaxis + width
for ax in [axfe,axfs]:
    ax.set_xticks(np.arange(1,nft+1))
    ax.set_xticklabels(ftlist)
    ax.grid()
axfe.set(xlabel='forecast hours',title='Forecast error against analysis')
axfe.legend()
axfs.set(xlabel='forecast hours',title='Ensemble spread')
axfs.legend()
fig.savefig(wdir/f"{model}_fe+fs_mem_linear_{pt}.png")
plt.show()