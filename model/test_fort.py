import numpy as np 
import matplotlib.pyplot as plt 
import sys
from pathlib import Path

model="l05II"
if len(sys.argv)>1:
    model=sys.argv[1]
if model=="l05II":
    #nx=240
    #nk=8
    #F=15.0
    nx = 40
    nk = 1
    F = 1.0
    nt=401
    pyfile=f"lorenz/l05II/euler_n{nx}k{nk}F{int(F)}.npy"
    pytendfile=f"lorenz/l05II/euler_tend_n{nx}k{nk}F{int(F)}.npy"
    fortfile=f"lorenz05/build/l05II_euler_n{nx}k{nk}F{int(F)}.grd"
    forttendfile=f"lorenz05/build/l05II_euler_tend_n{nx}k{nk}F{int(F)}.grd"
    figname=f"n{nx}k{nk}F{int(F)}"
elif model=="l05IIm":
    nx=240
    nks=[8,16,32,64]
    F=15.0
    pyfile=f"lorenz/l05IIm/n{nx}k{'+'.join([str(n) for n in nks])}F{int(F)}.npy"
    fortfile=f"lorenz05/build/l05IIm_n{nx}k{'+'.join([str(n) for n in nks])}F{int(F)}.grd"
    figname=f"n{nx}k{'+'.join([str(n) for n in nks])}F{int(F)}"
elif model=="l05III":
    nx=960
    nk=32
    ni=12
    b=10.0
    c=0.6
    F=15.0
    pyfile=f"lorenz/l05III/n{nx}k{nk}i{ni}F{int(F)}c{c:.1f}.npy"
    fortfile=f"lorenz05/build/l05III_n{nx}k{nk}i{ni}F{int(F)}c{c:.1f}.grd"
    figname=f"n{nx}k{nk}i{ni}F{int(F)}c{c:.1f}"
    filmat=np.fromfile("lorenz05/build/filmat.grd",dtype=">f4").reshape(-1,nx)
    plt.matshow(filmat)
    plt.colorbar()
    plt.show()
    plt.close()
elif model=="l05IIIm":
    nx=960
    nks=[32,64,128,256]
    ni=12
    b=10.0
    c=0.6
    F=15.0
    pyfile=f"lorenz/l05IIIm/n{nx}k{'+'.join([str(n) for n in nks])}i{ni}F{int(F)}b{b:.1f}c{c:.1f}.npy"
    fortfile=f"lorenz05/build/l05IIIm_n{nx}k{'+'.join([str(n) for n in nks])}i{ni}F{int(F)}b{b:.1f}c{c:.1f}.grd"
    figname=f"n{nx}k{'+'.join([str(n) for n in nks])}i{ni}F{int(F)}b{b:.1f}c{c:.1f}"
xpy=np.load(pyfile)
xtpy=np.load(pytendfile)
#xfort=np.fromfile(fortfile,dtype=">f4").reshape(-1,nx)
xfort = np.fromfile(fortfile,'<d',nt*nx)
xfort = xfort.reshape((nt,nx),order='F')
xtfort = np.fromfile(forttendfile,'<d',nt*nx)
xtfort = xtfort.reshape((nt,nx),order='F')

figdir=Path(f"lorenz05/test/{model}")
if not figdir.exists(): figdir.mkdir(parents=True)

ntmax = 100
taxis = np.arange(xpy.shape[0])
xaxis = np.arange(xpy.shape[1])
vmax = max(np.max(xpy[:ntmax,:]),np.max(xfort[:ntmax,:]))
vmin = min(np.min(xpy[:ntmax,:]),np.min(xfort[:ntmax,:]))
fig, axs = plt.subplots(ncols=3,figsize=[8,6],sharey=True,constrained_layout=True)
p0=axs[0].pcolormesh(xaxis,taxis[:ntmax],xpy[:ntmax,:],shading='auto',cmap='coolwarm',vmin=vmin,vmax=vmax)
p1=axs[1].pcolormesh(xaxis,taxis[:ntmax],xfort[:ntmax,:],shading='auto',cmap='coolwarm',vmin=vmin,vmax=vmax)
diff=xfort[:ntmax]-xpy[:ntmax]
print(f"diff max={np.max(diff)} min={np.min(diff)}")
p2=axs[2].pcolormesh(xaxis,taxis[:ntmax],diff,shading='auto',cmap='coolwarm')
fig.colorbar(p1,ax=axs[1],pad=0.01,shrink=0.6)
fig.colorbar(p2,ax=axs[2],pad=0.01,shrink=0.6)
axs[0].set_title('Python')
axs[1].set_title('Fortran')
axs[2].set_title('Diff')
fig.suptitle('state')
fig.savefig(figdir/f"test_{figname}.png",dpi=300)
plt.show()
plt.close()

vmax = max(np.max(xtpy[:ntmax,:]),np.max(xtfort[:ntmax,:]))
vmin = min(np.min(xtpy[:ntmax,:]),np.min(xtfort[:ntmax,:]))
fig, axs = plt.subplots(ncols=3,figsize=[8,6],sharey=True,constrained_layout=True)
p0=axs[0].pcolormesh(xaxis,taxis[:ntmax],xtpy[:ntmax,:],shading='auto',cmap='coolwarm',vmin=vmin,vmax=vmax)
p1=axs[1].pcolormesh(xaxis,taxis[:ntmax],xtfort[:ntmax,:],shading='auto',cmap='coolwarm',vmin=vmin,vmax=vmax)
diff=xtfort[:ntmax]-xtpy[:ntmax]
print(f"diff tend max={np.max(diff)} min={np.min(diff)}")
p2=axs[2].pcolormesh(xaxis,taxis[:ntmax],diff,shading='auto',cmap='coolwarm')
fig.colorbar(p1,ax=axs[1],pad=0.01,shrink=0.6)
fig.colorbar(p2,ax=axs[2],pad=0.01,shrink=0.6)
axs[0].set_title('Python')
axs[1].set_title('Fortran')
axs[2].set_title('Diff')
fig.suptitle('tendency')
fig.savefig(figdir/f"test_tend_{figname}.png",dpi=300)
plt.show()
plt.close()

figdir = figdir/figname
if not figdir.exists(): figdir.mkdir()
for i in range(ntmax):
    fig, axs = plt.subplots(nrows=2,sharex=True)
    axs[0].plot(xaxis,xpy[i,])
    axs[0].plot(xaxis,xfort[i,])
    diff1 = xpy[i,] - xfort[i,]
    axs[1].plot(xaxis,xpy[i,]-xfort[i,])
    for ax in axs:
        if np.sqrt(np.dot(diff1,diff1))>1.0e-16:
            imax = np.argmax(np.abs(diff1))
            ax.vlines([imax],0,1,colors='r',transform=ax.get_xaxis_transform(),zorder=0)
        ax.grid()
    fig.suptitle(f"i={i}")
    fig.savefig(figdir/f"i{i}.png")
    #plt.show()
    plt.close()

    fig, axs = plt.subplots(nrows=2,sharex=True)
    axs[0].plot(xaxis,xtpy[i,])
    axs[0].plot(xaxis,xtfort[i,])
    diff1 = xtpy[i,] - xtfort[i,]
    axs[1].plot(xaxis,xtpy[i,]-xtfort[i,])
    for ax in axs:
        if np.sqrt(np.dot(diff1,diff1))>1.0e-16:
            imax = np.argmax(np.abs(diff1))
            ax.vlines([imax],0,1,colors='r',transform=ax.get_xaxis_transform(),zorder=0)
        ax.grid()
    fig.suptitle(f"i={i} tend")
    fig.savefig(figdir/f"i{i}tend.png")
    #plt.show()
    plt.close()