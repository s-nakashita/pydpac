import numpy as np 
import matplotlib.pyplot as plt 
from pathlib import Path
plt.rcParams['font.size'] = 12

npdir = Path('../../model/qg/test')
jaxdir = Path('test')
figdir = Path('test/fig')
if not figdir.exists():
    figdir.mkdir()

ts = 1000
te = 10000
dt = 100
cmap_q = plt.get_cmap('viridis')(np.linspace(0,1,int((te-ts)/dt)+1))
#figq1d, axs1d = plt.subplots(nrows=2)
for i,t in enumerate(range(ts, te+dt, dt)):
    qn = np.load(npdir/f"q{t:06d}.npy").T
    qj = np.load(jaxdir/f"q{t:06d}.npy").T
    psin = np.load(npdir/f"p{t:06d}.npy").T
    psij = np.load(jaxdir/f"p{t:06d}.npy").T
    n = qn.shape[0]
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,n)
    fig, axs = plt.subplots(2,3,figsize=[12,6],constrained_layout=True)
    # psi
    pd = (psij - psin) #/np.sqrt(np.mean(psin**2))*100
    vmax = max(np.max(psin),np.max(psij),np.max(pd))
    vmin = min(np.min(psin),np.min(psij),np.min(pd))
    vlim = max(vmax,-vmin)
    c = axs[0,0].pcolormesh(x,y,psin,vmin=-vlim,vmax=vlim)
    fig.colorbar(c, ax=axs[0,0], shrink=0.6, pad=0.01)
    axs[0,0].set_title('NumPy')
    axs[0,0].set_ylabel(r'$\psi$')
    c = axs[0,1].pcolormesh(x,y,psij,vmin=-vlim,vmax=vlim)
    fig.colorbar(c,ax=axs[0,1],shrink=0.6,pad=0.01)
    axs[0,1].set_title('JAX')
    c = axs[0,2].pcolormesh(x,y,pd,cmap='RdBu',vmin=-vlim,vmax=vlim)
    fig.colorbar(c,ax=axs[0,2],shrink=0.6,pad=0.01)
    axs[0,2].set_title('JAX - NumPy')
    # q
    qd = (qj - qn) #/np.sqrt(np.mean(qn**2))*100
    vmax = max(np.max(qn),np.max(qj),np.max(qd))
    vmin = min(np.min(qn),np.min(qj),np.min(qd))
    vlim = max(vmax,-vmin)
    c = axs[1,0].pcolormesh(x,y,qn,vmin=-vlim,vmax=vlim)
    fig.colorbar(c,ax=axs[1,0],shrink=0.6,pad=0.01)
    axs[1,0].set_ylabel(r'$q$')
    c = axs[1,1].pcolormesh(x,y,qj,vmin=-vlim,vmax=vlim)
    fig.colorbar(c,ax=axs[1,1],shrink=0.6,pad=0.01)
    c = axs[1,2].pcolormesh(x,y,qd,cmap='RdBu',vmin=-vlim,vmax=vlim)
    fig.colorbar(c,ax=axs[1,2],shrink=0.6,pad=0.01)
    for ax in axs.flatten():
        ax.set_aspect(1.0)
    fig.suptitle(r"$t=$"+f"{t}")
    fig.savefig(figdir/f"pq{t:06d}.png")
    #plt.show()
    plt.close(fig=fig)

#    axs1d[0].plot(y,qn[32,],c=cmap_q[i],label=f't={t}')
#    axs1d[0].plot(y,qj[32,],c=cmap_q[i],ls='dashed')
#    axs1d[1].plot(y,qj[32,]-qn[32,],c=cmap_q[i])
#axs1d[0].legend(ncol=2)
#axs1d[0].set_title('solid=NumPy, dashed=JAX')
#axs1d[1].set_title('JAX - NumPy')
#figq1d.suptitle(r'$q$ at y=0.25')
#plt.show()
