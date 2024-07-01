import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 18

xb = 27.0
sigb = 2.0
y = 23.5
sigo = 1.5

siga = np.sqrt(1.0/(1.0/sigb/sigb + 1.0/sigo/sigo))
xa = siga*siga*(xb/sigb/sigb + y/sigo/sigo)

fig, ax = plt.subplots(figsize=[8,6])
x = np.linspace(20.0,32.0,101)
pb=norm.pdf(x,loc=xb,scale=sigb)
po=norm.pdf(x,loc=y,scale=sigo)
pa=norm.pdf(x,loc=xa,scale=siga)
labels=[r'$x_\mathrm{b}$',r'$y$',r'$x_\mathrm{a}$']
sigmas={r'$x_\mathrm{b}$':r'$\sigma^\mathrm{b}$',r'$y$':r'$\sigma^\mathrm{o}$',r'$x_\mathrm{a}$':r'$\sigma^\mathrm{a}$'}
colors={r'$x_\mathrm{b}$':'b',r'$y$':'r',r'$x_\mathrm{a}$':'g'}
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
fig.savefig("demo_var.png",dpi=300)
plt.show()