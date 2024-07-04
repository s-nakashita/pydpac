import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
from pathlib import Path
import sys

model = 'l96'
pt = 'letkf'
nens = 20
if len(sys.argv)>1:
    nens = int(sys.argv[1])
datadir = Path(f'/Volumes/dandelion/pyesa/data/{model}/extfcst_m{nens}')

xf00 = np.load(datadir/f"{model}_xf00_linear_{pt}.npy")
xf24 = np.load(datadir/f"{model}_xf24_linear_{pt}.npy")
xf48 = np.load(datadir/f"{model}_xf48_linear_{pt}.npy")
xf72 = np.load(datadir/f"{model}_xf72_linear_{pt}.npy")
xf96 = np.load(datadir/f"{model}_xf96_linear_{pt}.npy")
xf120 = np.load(datadir/f"{model}_xf120_linear_{pt}.npy")
print(xf00.shape)
print(xf24.shape)
print(xf48.shape)
print(xf72.shape)
print(xf96.shape)
print(xf120.shape)
xflist = [xf00,xf24,xf48,xf72,xf96,xf120]
ftlist = [0,24,48,72,96,120]
offsets = [0,4,8,12,16,20]

ix = np.arange(xf00.shape[1])
t = np.arange(xf00.shape[0]) # 6-hourly
d = t / 4 # days

fig, axs = plt.subplots(figsize=[10,4],ncols=6,sharey=True,constrained_layout=True)
mplist = []
for i in range(len(xflist)):
    xf = xflist[i]
    ft = ftlist[i]
    ioffset = offsets[i]
    #t = np.arange(xf.shape[0]) # 6-hourly
    #d = t / 4 # days
    axs[i].set_title(f'FT{ft:02d}')
    mp = axs[i].pcolormesh(ix,d[ioffset:],xf[ioffset:xf.shape[0]-ioffset,:,0],norm=Normalize(-10,10),cmap='coolwarm')
    mplist.append(mp)
fig.colorbar(mplist[-1],ax=axs[-1],shrink=0.6,pad=0.01)
fig.savefig(datadir/f"{model}_xf_linear_{pt}.png")
plt.show()

fig, axs = plt.subplots(figsize=[10,4],ncols=6,sharey=True,constrained_layout=True)
mplist = []
for i in range(len(xflist)):
    xf = xflist[i]
    ft = ftlist[i]
    ioffset = offsets[i]
    if i==0:
        t = np.arange(xf.shape[0]) # 6-hourly
        d = t / 4 # days
        axs[i].set_title(f'FT{ft:02d}')
        mp = axs[i].pcolormesh(ix,d,xf[ioffset:,:,0],norm=Normalize(-10,10),cmap='coolwarm')
    else:
        xd = xf[ioffset:-ioffset,:,0] - xflist[0][ioffset:,:,0]
        axs[i].set_title(f'FT{ft:02d} - FT00')
        mp = axs[i].pcolormesh(ix,d[ioffset:],xd,norm=Normalize(-10,10),cmap='coolwarm')
    mplist.append(mp)
fig.colorbar(mplist[0],ax=axs[0],shrink=0.6,pad=0.01)
fig.colorbar(mplist[-1],ax=axs[-1],shrink=0.6,pad=0.01)
fig.savefig(datadir/f"{model}_xfd_linear_{pt}.png")
plt.show()

fig, axs = plt.subplots(figsize=[10,4],ncols=6,sharey=True,constrained_layout=True)
mplist = []
for i in range(len(xflist)):
    xf = xflist[i]
    ft = ftlist[i]
    ioffset = offsets[i]
    xs = np.std(xf,axis=2)
    #t = np.arange(xf.shape[0]) # 6-hourly
    #d = t / 4 # days
    axs[i].set_title(f'FT{ft:02d}')
    mp = axs[i].pcolormesh(ix,d[ioffset:],xs[ioffset:xs.shape[0]-ioffset,:],norm=Normalize(0,2.5),cmap='viridis')
    mplist.append(mp)
fig.colorbar(mplist[-1],ax=axs[-1],shrink=0.6,pad=0.01)
fig.savefig(datadir/f"{model}_xfs_linear_{pt}.png")
plt.show()