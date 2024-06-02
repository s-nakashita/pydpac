import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['font.size'] = 16
from pathlib import Path

## nature
nx_true = 960
nk_true = 32
nks_true = [256,128,64,32]
ni_true = 12
b_true = 10.0
c_true = 0.6
F_true = 15.0
lamstep = 1
dt_true = 0.05 / 36 / lamstep
## GM
intgm = 4
nx_gm = nx_true // intgm
nk_gm = nk_true // intgm
nks_gm = np.array(nks_true,dtype=np.int64) // intgm
## LAM
nx_lam = 240
ist_lam = 240
lamstep = 1
nsp = 10
po = 1
intrlx = 1
nk_lam = nk_true
nks_lam = nks_true
ni_lam = ni_true
b_lam = b_true
c_lam = c_true
F = F_true
dt = dt_true * lamstep
datadir = Path('/Volumes/FF520/nested_envar/data')
#figdir = datadir/Path(f'l05nest/nsp{nsp}p{po}intrlx{intrlx}')
figdir = datadir/Path(f'l05nestm/m{"+".join([str(n) for n in nks_gm])}/nsp{nsp}p{po}intrlx{intrlx}')

ix_t = np.loadtxt(figdir/'ix_t.txt')
ix_gm = np.loadtxt(figdir/'ix_gm.txt')
ix_lam = np.loadtxt(figdir/'ix_lam.txt')
z_t = np.load(figdir/'x_t.npy')
z_gm = np.load(figdir/'x_gm.npy')
z_lam = np.load(figdir/'x_lam.npy')
print(z_t.shape)

nx = ix_t.size
nx_gm = ix_gm.size
nx_lam = ix_lam.size
phi = np.linspace(0.0,2.0*np.pi,nx+1)
x = np.cos(phi)
y = np.sin(phi)
phi_gm = np.linspace(0.0,2.0*np.pi,nx_gm+1)
x_gm = np.cos(phi_gm)
y_gm = np.sin(phi_gm)
phi_lam = 2.0*np.pi*ix_lam / nx
x_lam = np.cos(phi_lam)
y_lam = np.sin(phi_lam)

nt = z_t.shape[0]
t = np.arange(nt)
days = t * 0.25

zmin = np.min(np.array(z_t))
zmax = np.max(np.array(z_t))
# snapshots
for d in range(0,65,5):
    i = np.argmin(np.abs(days - d))
    zt0 = z_t[i]
    zg0 = z_gm[i]
    zl0 = z_lam[i]
    fig = plt.figure(figsize=[6,6])
    ax = fig.add_subplot(projection='3d',autoscale_on=False,\
        xlim=(-1.2,1.2),ylim=(-1.2,1.2),zlim=(zmin-0.1,zmax+0.1))
    ax.set_zlabel('X')
    ax.view_init(elev=30.,azim=135.)
    ax.plot(x,y,np.zeros_like(x),c='gray',alpha=0.5)
    # nature
    zb = np.hstack((zt0,[zt0[0]]))
    ax.plot(x,y,zb,c='b',lw=1.5)
    # GM
    zbg = np.hstack((zg0,[zg0[0]]))
    ax.plot(x_gm,y_gm,zbg,c='orange',lw=1.5)
    # LAM
    ax.plot(x_lam,y_lam,zl0,c='red',lw=1.5)
    time_template = 't = %.0f d'
    time_text = ax.text(0.05,0.9,20.0,time_template % days[i],\
        va='bottom',ha='left',transform=ax.transAxes)
    title = ax.set_title('Nature',color='b',loc='left',fontsize=18,pad=3.0)
    title = ax.annotate(' GM',color='orange',fontsize=18,\
        xycoords=title, xy=(1, 0), verticalalignment="bottom",)
    title = ax.annotate(' LAM',color='r',fontsize=18,\
        xycoords=title, xy=(1, 0), verticalalignment="bottom",)
    fig.savefig(figdir/f'nature+GM+LAM_d{d}.png',dpi=300)
    plt.close()

# animation
fig = plt.figure(figsize=[6,6])
ax = fig.add_subplot(projection='3d',autoscale_on=False,\
    xlim=(-1.2,1.2),ylim=(-1.2,1.2),zlim=(zmin-0.1,zmax+0.1))
ax.set_zlabel('X')
ax.view_init(elev=30.,azim=135.)
ax.plot(x,y,np.zeros_like(x),c='gray',alpha=0.5)
lines = []
l, = ax.plot([],[],[],c='b',lw=1.5)
lines.append(l)
l_gm, = ax.plot([],[],[],c='orange',lw=1.5)
lines.append(l_gm)
l_lam, = ax.plot([],[],[],c='r',lw=1.5)
lines.append(l_lam)
time_template = 't = %.0f d'
time_text = ax.text(0.05,0.9,20.0,'',\
    va='bottom',ha='left',transform=ax.transAxes)
title = ax.set_title('',color='b',loc='left',fontsize=18,pad=3.0)
title1 = ax.annotate('',color='orange',fontsize=18,\
        xycoords=title, xy=(1, 0), verticalalignment="bottom",)
title2 = ax.annotate('',color='r',fontsize=18,\
        xycoords=title1, xy=(1, 0), verticalalignment="bottom",)

def animate(i,days,zt,zg,zl,lines):
    t0=days[i]
    zt0 = zt[i]
    zg0 = zg[i]
    zl0 = zl[i]
    # nature
    line = lines[0]
    zb = np.hstack((zt0,[zt0[0]]))
    line.set_data(x,y)
    line.set_3d_properties(zb)
    # GM
    line = lines[1]
    zbg = np.hstack((zg0,[zg0[0]]))
    line.set_data(x_gm,y_gm)
    line.set_3d_properties(zbg)
    # LAM
    line = lines[2]
    line.set_data(x_lam,y_lam)
    line.set_3d_properties(zl0)
    #
    time_text.set_text(time_template % t0)
    outlist = lines + [time_text]
    #
    title.set_text('Nature')
    title1.set_text(' GM')
    title2.set_text(' LAM')
    outlist = outlist + [title,title1,title2]
    return outlist

ani = animation.FuncAnimation(fig, animate, nt,\
    fargs=(days,z_t,z_gm,z_lam,lines), interval=50)
writer = animation.PillowWriter(fps=15,metadata=dict(artist='Me'),bitrate=1800)
ani.save(figdir/'nature+GM+LAM.gif',writer=writer)
plt.show()