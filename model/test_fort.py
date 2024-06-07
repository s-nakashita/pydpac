import numpy as np 
import matplotlib.pyplot as plt 
import sys

model="l05II"
if len(sys.argv)>1:
    model=sys.argv[1]
if model=="l05II":
    nx=240
    nk=8
    F=15.0
    pyfile=f"lorenz/l05II/n{nx}k{nk}F{int(F)}.npy"
    fortfile=f"lorenz05/build/l05II_n{nx}k{nk}F{int(F)}.grd"
elif model=="l05IIm":
    nx=240
    nks=[8,16,32,64]
    F=15.0
    pyfile=f"lorenz/l05IIm/n{nx}k{'+'.join([str(n) for n in nks])}F{int(F)}.npy"
    fortfile=f"lorenz05/build/l05IIm_n{nx}k{'+'.join([str(n) for n in nks])}F{int(F)}.grd"
elif model=="l05III":
    nx=960
    nk=32
    ni=12
    b=10.0
    c=0.6
    F=15.0
    pyfile=f"lorenz/l05III/n{nx}k{nk}i{ni}F{int(F)}c{c:.1f}.npy"
    fortfile=f"lorenz05/build/l05III_n{nx}k{nk}i{ni}F{int(F)}c{c:.1f}.grd"
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
xpy=np.load(pyfile)
xfort=np.fromfile(fortfile,dtype=">f4").reshape(-1,nx)

taxis = np.arange(xpy.shape[0])
xaxis = np.arange(xpy.shape[1])
fig, axs = plt.subplots(ncols=3,figsize=[8,6],sharey=True,constrained_layout=True)
p0=axs[0].pcolormesh(xaxis,taxis,xpy,shading='auto',cmap='coolwarm',vmin=-15.0,vmax=15.0)
p1=axs[1].pcolormesh(xaxis,taxis,xfort,shading='auto',cmap='coolwarm',vmin=-15.0,vmax=15.0)
p2=axs[2].pcolormesh(xaxis,taxis,xfort-xpy,shading='auto',cmap='coolwarm')
fig.colorbar(p1,ax=axs[1],pad=0.01,shrink=0.6)
fig.colorbar(p2,ax=axs[2],pad=0.01,shrink=0.6)
axs[0].set_title('Python')
axs[1].set_title('Fortran')
axs[2].set_title('Diff')
fig.savefig(f"lorenz05/build/test_{model}.png",dpi=300)
plt.show()

for i in range(0,101,10):
    fig, axs = plt.subplots(nrows=2,sharex=True)
    axs[0].plot(xaxis,xpy[i,])
    axs[0].plot(xaxis,xfort[i,])
    axs[1].plot(xaxis,xpy[i,]-xfort[i,])
    fig.suptitle(f"i={i}")
    plt.show()