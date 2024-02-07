import numpy as np 
import matplotlib.pyplot as plt 
plt.rcParams['font.size'] = 14
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from pathlib import Path
from lorenz_nest import L05nest
from lorenz3 import L05III
from scipy.interpolate import interp1d

# true (Lorenz III)
nx = 960
nk = 32
ni = 12
b = 10.0
c = 0.6
h = 0.05 / b
F = 15.0
step = L05III(nx,nk,ni,b,c,h,F)
# GM (Lorenz II)
nx_gm = 240
intgm = int(nx / nx_gm)
nk_gm = nk // intgm
h_gm = 0.05 / 36
# LAM (Lorenz III)
nx_lam = 240
nk_lam = nk
ni_lam = ni
ist_lam = 520
nsp = 10
lamstep = 1
step_nest = L05nest(nx,nx_gm,nx_lam,nk_gm,nk_lam,ni_lam,b,c,h_gm,F,intgm,ist_lam,nsp,lamstep)

figdir = Path(f'chaos/lorenz_nest')
if not figdir.exists():
    figdir.mkdir(parents=True)

# truth
z0 = np.zeros(nx)
z0[nx//2] += F*0.01
for i in range(100*int(0.05/h)):
    z0 = step(z0)
ne = 30
t2gm = interp1d(step_nest.ix_true,z0)
z0_gm = t2gm(step_nest.ix_gm)
z0e_gm = np.zeros((nx_gm,ne))
z0e_gm[:,0] = z0_gm[:]
for j in range(ne):
    z0e_gm[:,j] = z0_gm[:]
    z0e_gm[:,j] = z0_gm[:] + np.random.randn(nx_gm)*0.1
gm2lam = interp1d(step_nest.ix_gm,z0e_gm,axis=0)
z0e_lam = gm2lam(step_nest.ix_lam)
t = 0.0

phi = np.linspace(0,2.0*np.pi,nx+1)
x = np.cos(phi)
y = np.sin(phi)
phi_gm = np.linspace(0,2.0*np.pi,nx_gm+1)
x_gm = np.cos(phi_gm)
y_gm = np.sin(phi_gm)
phi_lam = 2.0*np.pi * step_nest.ix_lam / nx
x_lam = np.cos(phi_lam)
y_lam = np.sin(phi_lam)

cmap = plt.get_cmap('tab10')
fig = plt.figure(figsize=[12,9],constrained_layout=True)
gs = gridspec.GridSpec(1,2,figure=fig)
gs0 = gs[0].subgridspec(5,1)
ax0 = fig.add_subplot(gs0[:2,:])
ax1 = fig.add_subplot(gs0[2:,:],projection='3d')
ax0.set_xlabel(r'$\phi$')
ax0.set_ylabel('X')
ax1.plot(x,y,np.zeros_like(x),c='gray',alpha=0.5)
ax1.plot(x_lam,y_lam,np.zeros_like(x_lam),c='r',alpha=0.7)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
#ax1.set_xticklabels([])
#ax1.set_yticklabels([])
ax1.set_zlabel('X')
labels = []
lines = []
zb = np.hstack((z0,[z0[0]]))
zb_gm = np.vstack((z0e_gm,z0e_gm[0].reshape(1,-1)))
zb_lam = z0e_lam.copy()
ax0.plot(phi,zb,c='b',lw=1.5)
ax1.plot(x,y,zb,c='b',lw=1.5)
labels.append('truth')
lines.append(Line2D([0],[0],color='b',lw=1.5))
for i in range(zb_gm.shape[1]):
    #c = cmap(i)
    c = 'orange'
    ax0.plot(phi_gm,zb_gm[:,i],c=c,lw=0.5,alpha=0.5)
    ax1.plot(x_gm,y_gm,zb_gm[:,i],c=c,lw=0.5,alpha=0.5)
    if i==0:
        labels.append('GM ensemble')
        lines.append(Line2D([0],[0],color=c,lw=0.5,alpha=0.5))
    c = 'r'
    ax0.plot(phi_lam,zb_lam[:,i],c=c,lw=0.5,alpha=0.5)
    ax1.plot(x_lam,y_lam,zb_lam[:,i],c=c,lw=0.5,alpha=0.5)
    if i==0:
        labels.append('LAM ensemble')
        lines.append(Line2D([0],[0],color=c,lw=0.5,alpha=0.5))
ax1.set_title(f't={int(t/0.05/4)}d')
ax1.view_init(elev=30.,azim=225.)

nt6h = int(0.05/h)
nt = nt6h * 4 * 30
zt = []
zt.append(z0)
ze_gm = []
ze_gm.append(z0e_gm)
ze_lam = []
ze_lam.append(z0e_lam)
for i in range(nt):
    t += h 
    z0 = step(z0)
    if i % nt6h == 0:
        zt.append(z0)
        z0e_gm, z0e_lam = step_nest(z0e_gm,z0e_lam)
        ze_gm.append(z0e_gm)
        ze_lam.append(z0e_lam)

gs1 = gs[1].subgridspec(5,1)
ax2 = fig.add_subplot(gs1[:2,:])
ax3 = fig.add_subplot(gs1[2:,:],projection='3d')
ax2.set_xlabel(r'$\phi$')
ax2.set_ylabel('X')
ax3.plot(x,y,np.zeros_like(x),c='gray',alpha=0.5)
ax3.plot(x_lam,y_lam,np.zeros_like(x_lam),c='r',alpha=0.7)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
#ax3.set_xticklabels([])
#ax3.set_yticklabels([])
ax3.set_zlabel('X')
zb = np.hstack((z0,[z0[0]]))
zb_gm = np.vstack((z0e_gm,z0e_gm[0].reshape(1,-1)))
zb_lam = z0e_lam.copy()
ax2.plot(phi,zb,c='b',lw=1.5)
ax3.plot(x,y,zb,c='b',lw=1.5)
for i in range(zb_gm.shape[1]):
    #c = cmap(i)
    c = 'orange'
    ax2.plot(phi_gm,zb_gm[:,i],c=c,lw=0.5,alpha=0.5)
    ax3.plot(x_gm,y_gm,zb_gm[:,i],c=c,lw=0.5,alpha=0.5)
    c = 'r'
    ax2.plot(phi_lam,zb_lam[:,i],c=c,lw=0.5,alpha=0.5)
    ax3.plot(x_lam,y_lam,zb_lam[:,i],c=c,lw=0.5,alpha=0.5)
#    if i==0:
#        labels.append(f't={int(t/0.05/4)}d')
#        lines.append(Line2D([0],[0],color=c))
ax3.set_title(f't={int(t/0.05/4)}d')
ax3.view_init(elev=30.,azim=225.)

zmin = np.min(np.array(zt)) #min(np.min(np.array(ze_gm)),np.min(np.array(ze_lam)))
zmax = np.max(np.array(zt)) #max(np.max(np.array(ze_gm)),np.max(np.array(ze_lam)))
ax1.set_zlim(zmin-0.1,zmax+0.1)
ax3.set_zlim(zmin-0.1,zmax+0.1)

ax0.legend(lines,labels)
fig.suptitle(f"Nested Lorenz")
fig.savefig(figdir/"chaos.png",dpi=300)
plt.show()
plt.close()
#exit()

# animation
import matplotlib.animation as animation
fig= plt.figure(figsize=[6,6])
ax = fig.add_subplot(projection='3d',\
    autoscale_on=False,xlim=(-1.2,1.2),ylim=(-1.2,1.2),zlim=(zmin-0.1,zmax+0.1))
ax.set_zlabel('X')
ax.set_title(f'Nested Lorenz')
ax.view_init(elev=30.,azim=225.)
lines = []
truth, = ax.plot([],[],[],c='b',lw=1.5)
lines.append(truth)
for j in range(1,ne+1):
    ensfcst_gm, = ax.plot([],[],[],c='orange',lw=0.5,alpha=0.5)
    lines.append(ensfcst_gm)
for j in range(1,ne+1):
    ensfcst_lam, = ax.plot([],[],[],c='r',lw=0.5,alpha=0.5)
    lines.append(ensfcst_lam)
time_template = 't = %d d'
time_text = ax.text(0.05,0.9,0.9,'',transform=ax.transAxes)

def animate(i, zt, ze_gm, ze_lam, lines):
    t=int(i*nt6h*h/0.05/4)
    zt0 = zt[i]
    ze0_gm = ze_gm[i]
    ze0_lam = ze_lam[i]
    zb = np.hstack((zt0,[zt0[0]]))
    zb_gm = np.vstack((ze0_gm,ze0_gm[0,:].reshape(1,-1)))
    zb_lam = ze0_lam.copy()
    for j, line in enumerate(lines):
        if j==0:
            line.set_data(x,y)
            line.set_3d_properties(zb)
        elif j<ne+1:
            line.set_data(x_gm,y_gm)
            line.set_3d_properties(zb_gm[:,j-1])
        else:
            line.set_data(x_lam,y_lam)
            line.set_3d_properties(zb_lam[:,j-(ne+1)])
    time_text.set_text(time_template % t)
    outlist = lines + [time_text]
    return outlist

ani = animation.FuncAnimation(fig, animate, len(zt),\
    fargs=(zt,ze_gm,ze_lam,lines), interval=50)
writer = animation.PillowWriter(fps=15,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
ani.save(figdir/'chaos.gif', writer=writer)

plt.show()