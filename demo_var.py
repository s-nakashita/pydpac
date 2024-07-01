import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt 
from pathlib import Path
plt.rcParams['font.size'] = 18

xb = 27.0
sigb = 1.5
y = 23.5
sigo = 2.0
figdir = Path(f'demo_var/b3o4')
if not figdir.exists(): figdir.mkdir(parents=True)
title=r'$\sigma^\mathrm{b}$:$\sigma^\mathrm{o}$=3:4'

labels=[r'$x_\mathrm{b}$',r'$y$',r'$x_\mathrm{a}$']
sigmas={r'$x_\mathrm{b}$':r'$\sigma^\mathrm{b}$',r'$y$':r'$\sigma^\mathrm{o}$',r'$x_\mathrm{a}$':r'$\sigma^\mathrm{a}$'}
colors={r'$x_\mathrm{b}$':'b',r'$y$':'r',r'$x_\mathrm{a}$':'g'}

corrlist = np.linspace(-0.9,0.9,19).tolist()
xalist = []
sigalist=[]
for corr in corrlist:
    
    tmp = 1.0 / sigb / sigb + 1.0 / sigo / sigo - 2.0 * corr / sigb / sigo
    siga = np.sqrt((1.0-corr*corr)/tmp)
    xa = siga*siga*(xb/sigb/sigb + y/sigo/sigo - corr*(xb+y)/sigb/sigo)/(1.0-corr*corr)
    xalist.append(xa)
    sigalist.append(siga)

    fig, ax = plt.subplots(figsize=[8,6])
    x = np.linspace(20.0,32.0,101)
    pb=norm.pdf(x,loc=xb,scale=sigb)
    po=norm.pdf(x,loc=y,scale=sigo)
    pa=norm.pdf(x,loc=xa,scale=siga)
    for p, xc, sig, label in zip(
        [pb,po,pa],
        [xb,y,xa],
        [sigb,sigo,siga],
        labels
    ):
        c=colors[label]
        ax.plot(x,p,c=c,lw=3.0)
        ax.plot([xc],np.max(p),lw=0.0,marker='*',ms=18,c=c,label=label+f'={xc:.1f}')
        ax.vlines([xc],0,np.max(p),colors=c,ls='dashed')
        #ax.vlines([xc-sig,xc+sig],0,np.max(p)*np.exp(-0.5),colors=c,ls='dotted')
        ax.annotate("",xy=(xc,np.max(p)*np.exp(-0.5)),xytext=(xc+sig,np.max(p)*np.exp(-0.5)),arrowprops=dict(arrowstyle="<->",color=c,lw=1.5))
        ax.annotate(sigmas[label],xy=(xc+sig*0.5,np.max(p)*np.exp(-0.5)+0.01),color=c,ha='center',va='bottom')
    ax.legend()
    ax.grid()
    ax.set_title(title+f', corr={corr:.1f}')
    fig.savefig(figdir/f"c{corr:.1f}.png",dpi=300)
    #plt.show()
    plt.close()

fig, axs = plt.subplots(nrows=2,figsize=[8,6],sharex=True,constrained_layout=True)
axs[0].plot(corrlist,xalist,c='g')
axs[0].hlines([xb,y],0,1,colors=('b','r'),ls='dashed',transform=axs[0].get_yaxis_transform())
axs[0].set_title(r'$x^\mathrm{a}$')
axs[1].plot(corrlist,sigalist,c='g')
axs[1].hlines([sigb,sigo],0,1,colors=('b','r'),ls='dashed',transform=axs[1].get_yaxis_transform())
axs[1].set_title(r'$\sigma^\mathrm{a}$')
for ax in axs:
    ax.set_xticks(corrlist[::2])
    ax.set_xticks(corrlist,minor=True)
    ax.grid()
fig.suptitle(title)
fig.savefig(figdir/"corr.png",dpi=300)
plt.show()