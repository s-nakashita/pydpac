import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 14
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from pathlib import Path
from lorenz import L96
from lorenz2 import L05II
from lorenz3 import L05III

model = 1
if model == 1:
    # Lorenz I
    nx = 40
    F = 8.0
    h = 0.05
    step = L96(nx,h,F)
elif model == 2:
    # Lorenz II
    nx = 240
    nk = 8
    h = 0.05
    F = 10.0
    step = L05II(nx,nk,h,F)
elif model==3:
    # Lorenz III
    nx = 960
    nk = 32
    ni = 12
    b = 10.0
    c = 0.6
    h = 0.05 / b
    F = 15.0
    step = L05III(nx,nk,ni,b,c,h,F)

figdir = Path(f'chaos/lorenz{model}')
if not figdir.exists():
    figdir.mkdir(parents=True)

z0 = np.zeros(nx)
z0[nx//2] += F*0.01
for i in range(100*int(0.05/h)):
    z0 = step(z0)
ne = 30
z0e = np.zeros((nx,ne+1))
z0e[:,0] = z0[:] # truth
for j in range(1,ne+1):
    z0e[:,j] = z0[:]
    z0e[:,j] = z0[:] + np.random.randn(nx)*0.1
zb = np.vstack((z0e,z0e[0].reshape(1,-1)))
t = 0.0

phi = np.linspace(0,2.0*np.pi,nx+1)
x = np.cos(phi)
y = np.sin(phi)

cmap = plt.get_cmap('tab10')
fig = plt.figure(figsize=[12,9],constrained_layout=True)
gs = gridspec.GridSpec(1,2,figure=fig)
gs0 = gs[0].subgridspec(5,1)
ax0 = fig.add_subplot(gs0[:2,:])
ax1 = fig.add_subplot(gs0[2:,:],projection='3d')
ax0.set_xlabel(r'$\phi$')
ax0.set_ylabel('X')
ax1.plot(x,y,np.zeros_like(zb[:,0]),c='gray',alpha=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
#ax1.set_xticklabels([])
#ax1.set_yticklabels([])
ax1.set_zlabel('X')
labels = []
lines = []
ax0.plot(phi,zb[:,0],c='b',lw=1.5)
ax1.plot(x,y,zb[:,0],c='b',lw=1.5)
labels.append('truth')
lines.append(Line2D([0],[0],color='b',lw=1.5))
for i in range(1,zb.shape[1]):
    #c = cmap(i)
    c = 'r'
    ax0.plot(phi,zb[:,i],c=c,lw=0.5,alpha=0.5)
    ax1.plot(x,y,zb[:,i],c=c,lw=0.5,alpha=0.5)
    if i==1:
        labels.append('ensemble forecast')
        lines.append(Line2D([0],[0],color=c,lw=0.5,alpha=0.5))
ax1.set_title(f't={int(t/0.05/4)}d')

nt6h = int(0.05/h)
nt = nt6h * 4 * 30
ze = []
ze.append(z0e)
for i in range(nt):
    t += h 
    z0e = step(z0e)
    if i % nt6h == 0:
        ze.append(z0e)
zb = np.vstack((z0e,z0e[0].reshape(1,-1)))

gs1 = gs[1].subgridspec(5,1)
ax2 = fig.add_subplot(gs1[:2,:])
ax3 = fig.add_subplot(gs1[2:,:],projection='3d')
ax2.set_xlabel(r'$\phi$')
ax2.set_ylabel('X')
ax3.plot(x,y,np.zeros_like(zb[:,0]),c='gray',alpha=0.5)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
#ax3.set_xticklabels([])
#ax3.set_yticklabels([])
ax3.set_zlabel('X')
ax2.plot(phi,zb[:,0],c='b',lw=1.5)
ax3.plot(x,y,zb[:,0],c='b',lw=1.5)
for i in range(1,zb.shape[1]):
    #c = cmap(i)
    c = 'r'
    ax2.plot(phi,zb[:,i],c=c,lw=0.5,alpha=0.5)
    ax3.plot(x,y,zb[:,i],c=c,lw=0.5,alpha=0.5)
#    if i==0:
#        labels.append(f't={int(t/0.05/4)}d')
#        lines.append(Line2D([0],[0],color=c))
ax3.set_title(f't={int(t/0.05/4)}d')

ax0.legend(lines,labels)
fig.suptitle(f"Lorenz {model}")
fig.savefig(figdir/"chaos.png",dpi=300)
plt.show()
plt.close()

# animation
import matplotlib.animation as animation
zmin = np.min(np.array(ze))
zmax = np.max(np.array(ze))
fig= plt.figure(figsize=[6,6])
ax = fig.add_subplot(projection='3d',\
    autoscale_on=False,xlim=(-1.2,1.2),ylim=(-1.2,1.2),zlim=(zmin-0.1,zmax+0.1))
ax.set_zlabel('X')
ax.set_title(f'Lorenz {model}')
lines = []
truth, = ax.plot([],[],[],c='b',lw=1.5)
lines.append(truth)
for j in range(1,ne+1):
    ensfcst, = ax.plot([],[],[],c='r',lw=0.5,alpha=0.5)
    lines.append(ensfcst)
time_template = 't = %d d'
time_text = ax.text(0.05,0.9,0.9,'',transform=ax.transAxes)

def animate(i, ze, lines):
    t=int(i*nt6h*h/0.05/4)
    ze0 = ze[i]
    zb = np.vstack((ze0,ze0[0,:].reshape(1,-1)))
    for j, line in enumerate(lines):
        line.set_data(x,y)
        line.set_3d_properties(zb[:,j])
    time_text.set_text(time_template % t)
    outlist = lines + [time_text]
    return outlist

ani = animation.FuncAnimation(fig, animate, len(ze),\
    fargs=(ze,lines), interval=50)
writer = animation.PillowWriter(fps=15,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
ani.save(figdir/'chaos.gif', writer=writer)

plt.show()