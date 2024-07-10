import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
from asa import ASA
from enasa import EnASA
import sys
sys.path.append('../model')
from lorenz import L96
plt.rcParams['font.size'] = 16

# forecast model
nx = 40
dt = 0.05 / 6 # 1 hour
F = 8.0
model = L96(nx,dt,F)

# SA settings
vt = 48 # hours
if len(sys.argv)>1:
    vt = int(sys.argv[1])
ioffset = vt // 6
def cost(x,*args):
    xa, ic, hwidth = args
    nxh = x.size // 2
    i0 = nxh - hwidth
    i1 = nxh + hwidth + 1
    xd = np.roll(x,nxh-ic)[i0:i1] - np.roll(xa,nxh-ic)[i0:i1]
    return 0.5*np.dot(xd,xd)
def jac(x,*args):
    xa, ic, hwidth = args
    nxh = x.size // 2
    i0 = nxh - hwidth
    i1 = nxh + hwidth + 1
    dJdxtmp = np.zeros_like(x)
    dJdxtmp[i0:i1] = np.roll(x,nxh-ic)[i0:i1] - np.roll(xa,nxh-ic)[i0:i1]
    dJdx = np.roll(dJdxtmp,-nxh+ic)
    return dJdx

# load data
modelname = 'l96'
pt = 'letkf'
nens = 20
if len(sys.argv)>2:
    nens=int(sys.argv[2])
datadir = Path(f'/Volumes/dandelion/pyesa/data/{modelname}/extfcst_m{nens}')
xf00 = np.load(datadir/f"{modelname}_xf00_linear_{pt}.npy")
xfv  = np.load(datadir/f"{modelname}_xf{vt:02d}_linear_{pt}.npy")

icyc0 = 50
nsample = 1000
dJdx0_dict={'asa':[]}
dx0opt_dict={'asa':[]}
res_dict={'asa':[]}
rmsdJ_dict = {}
rmsdx_dict = {}
corrdJ_dict = {}
solverlist=['minnorm','diag','pcr','ridge','pls']
for solver in solverlist:
    dJdx0_dict[solver] = []
    dx0opt_dict[solver] = []
    res_dict[solver] = []
    rmsdJ_dict[solver] = []
    rmsdx_dict[solver] = []
    corrdJ_dict[solver] = []
markers=['*','o','v','s','P','X','p']
marker_style=dict(markerfacecolor='none')
cmap = plt.get_cmap('tab10')

cycles = []
ics = []
for i in range(nsample):
    icyc = icyc0 + i
    cycles.append(icyc)
    xa = xf00[icyc+ioffset].mean(axis=1)
    xf = xfv [icyc+ioffset].mean(axis=1)
    ic = np.argmax(np.abs(xa - xf)) # center of verification region
    hwidth = 1 # half-width of verification region
    ics.append(ic)
    args = (xa,ic,hwidth)

    # ASA
    asa = ASA(vt,cost,jac,model.step_adj,*args)
    # base trajectory
    nx = xa.size
    xb0 = xf00[icyc].mean(axis=1)
    xb = [xb0]
    for j in range(vt):
        xb0 = model(xb0)
        xb.append(xb0)
    # analysis
    dJdx0 = asa(xb)
    dx0opt = asa.calc_dxopt(xb,dJdx0)
    dJdx0_dict['asa'].append(dJdx0)
    dx0opt_dict['asa'].append(dx0opt)
    #asa.plot_hov()
    res_nl, res_tl = asa.check_djdx(xb,dJdx0,dx0opt,\
        model,model.step_t,plot=False)
    res_dict['asa'].append([res_nl,res_tl])

    # EnASA
    xe0 = xf00[icyc]
    X0 = xe0 - xe0.mean(axis=1)[:,None]
    xev = xfv[icyc+ioffset]
    Je = np.zeros(nens)
    for k in range(nens):
        Je[k] = cost(xev[:,k],*args)

    enasa = EnASA(vt,X0,Je)
    for solver in solverlist:
        dJedx0 = enasa(solver=solver)
        dxe0opt = asa.calc_dxopt(xb,dJedx0)
        dJdx0_dict[solver].append(dJedx0)
        dx0opt_dict[solver].append(dxe0opt)
        res_nl, res_tl = asa.check_djdx(xb,dJedx0,dxe0opt,\
            model,model.step_t,plot=False)
        res_dict[solver].append([res_nl,res_tl])
        rmsdJ_dict[solver].append(np.sqrt(np.mean((dJedx0-dJdx0)**2)))
        rmsdx_dict[solver].append(np.sqrt(np.mean((dxe0opt-dx0opt)**2)))
        corr=np.correlate(dJedx0,dJdx0)/np.linalg.norm(dJedx0,ord=2)/np.linalg.norm(dJdx0,ord=2)
        corrdJ_dict[solver].append(corr[0])
    if i<0:
        print(f"ic={ic}")
        fig, ax = plt.subplots()
        for j,key in enumerate(dJdx0_dict.keys()):
            if key=='asa':
                ax.plot(dJdx0_dict[key][i],label=key,lw=2.0)
            else:
                ax.plot(dJdx0_dict[key][i],ls='dashed',marker=markers[j],label=f'EnASA,{key}',**marker_style)
        ax.legend()
        ax.grid()
        ax.set_title('dJ/dx0')
        plt.show()

        fig, ax = plt.subplots()
        for j,key in enumerate(dx0opt_dict.keys()):
            if key=='asa':
                ax.plot(dx0opt_dict[key][i],label=key,lw=2.0)
            else:
                ax.plot(dx0opt_dict[key][i],ls='dashed',marker=markers[j],label=f'EnASA,{key}',**marker_style)
        ax.legend()
        ax.grid()
        ax.set_title('dxopt')
        plt.show()

figdir = Path("fig")
if not figdir.exists(): figdir.mkdir()
nrows=2
ncols=int(np.ceil(len(res_dict.keys())/2))
fig, axs = plt.subplots(ncols=ncols,nrows=nrows,figsize=[10,8],constrained_layout=True)
for i,key in enumerate(res_dict.keys()):
    res = np.array(res_dict[key])
    ax = axs.flatten()[i]
    ax.plot(res[:,0],res[:,1],marker=markers[i],lw=0.0, c=cmap(i),#ms=10,\
        **marker_style)
    ax.set_title(key)
#ymin, ymax = ax.get_ylim()
ymin = -1.1
ymax = 1.1
line = np.linspace(ymin,ymax,100)
for ax in axs.flatten():
    ax.plot(line,line,color='k',zorder=0)
    ax.grid()
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(ymin,ymax)
    ax.set_aspect(1.0)
axs[-1,1].set_xlabel(r'NLM: $\frac{J(M(\mathbf{x}_0+\delta\mathbf{x}_0^\mathrm{opt}))-J(\mathbf{x}_T)}{J(\mathbf{x}_T)}$')
axs[0,0].set_ylabel(r'TLM: $\frac{J(\mathbf{x}_T+\mathbf{M}\delta\mathbf{x}_0^\mathrm{opt})-J(\mathbf{x}_T)}{J(\mathbf{x}_T)}$')
fig.suptitle(r'$\delta J/J$'+f', FT{vt} {nens} member')
fig.savefig(figdir/f"res_vt{vt}nens{nens}.png",dpi=300)
plt.show()

fig, ax = plt.subplots(figsize=[10,8],constrained_layout=True)
for i,key in enumerate(rmsdJ_dict.keys()):
    marker_style['markerfacecolor']=cmap(i+1)
    marker_style['markeredgecolor']='k'
    x = np.array(rmsdJ_dict[key])
    y = np.array(rmsdx_dict[key])
    ax.plot(x,y,lw=0.0,marker='.',ms=5,c=cmap(i+1),zorder=0)
    xm = x.mean()
    ym = y.mean()
    ax.plot([xm],[ym],lw=0.0,marker=markers[i+1],ms=10,c=cmap(i+1),label=f'{key}=({xm:.2e},{ym:.2e})',**marker_style)
ax.set_xlabel('dJdx0')
ax.set_ylabel('dxopt')
ax.grid()
ax.legend()
fig.suptitle(f'RMSD against ASA, FT{vt} {nens} member')
fig.savefig(figdir/f"rms_vt{vt}nens{nens}.png",dpi=300)
plt.show()
plt.close()

fig, ax = plt.subplots(figsize=[8,8],constrained_layout=True)
for i,key in enumerate(corrdJ_dict.keys()):
    marker_style['markerfacecolor']=cmap(i+1)
    marker_style['markeredgecolor']='k'
    data = np.array(corrdJ_dict[key])
    bplot = ax.boxplot(data,positions=[i+1],patch_artist=True,whis=(0,100))
    for patch in bplot['boxes']:
        patch.set_facecolor(cmap(i+1))
        patch.set_alpha(0.3)
    ym = data.mean()
    ax.plot([i+1],[ym],lw=0.0,marker=markers[i+1],ms=10,c=cmap(i+1),**marker_style)
ax.set_xticks(np.arange(1,len(corrdJ_dict.keys())+1))
ax.set_xticklabels(corrdJ_dict.keys())
ax.set_ylabel('spatial correlation')
ax.grid(axis='y')
#ax.legend()
fig.suptitle(f'Spatial correlation of dJdx0 against ASA, FT{vt} {nens} member')
fig.savefig(figdir/f"corr_vt{vt}nens{nens}.png",dpi=300)
plt.show()
plt.close()

# save results to netcdf
savedir = Path('data')
if not savedir.exists(): savedir.mkdir(parents=True)
for key in res_dict.keys():
    res = np.array(res_dict[key])
    dJdx0 = np.array(dJdx0_dict[key])
    dx0opt = np.array(dx0opt_dict[key])
    ix = np.arange(dJdx0.shape[1])
    if key == 'asa':
        datadict = {
            "cycle":{"dims":("cycle"),"data":cycles},
            "x":{"dims":("x"),"data":ix},
            "ic":{
                "dims":("cycle"),"data":ics
            },
            "dJdx0":{
                "dims":("cycle","x"),
                "data":dJdx0
            },
            "dx0opt":{
                "dims":("cycle","x"),
                "data":dx0opt
            },
            "res_nl":{
                "dims":("cycle"),"data":res[:,0]
            },
            "res_tl":{
                "dims":("cycle"),"data":res[:,1]
            },
        }
    else:
        rmsdJ = rmsdJ_dict[key]
        rmsdx = rmsdx_dict[key]
        corrdJ = corrdJ_dict[key]
        datadict = {
            "cycle":{"dims":("cycle"),"data":cycles},
            "x":{"dims":("x"),"data":ix},
            "ic":{
                "dims":("cycle"),"data":ics
            },
            "dJdx0":{
                "dims":("cycle","x"),
                "data":dJdx0
            },
            "dx0opt":{
                "dims":("cycle","x"),
                "data":dx0opt
            },
            "res_nl":{
                "dims":("cycle"),"data":res[:,0]
            },
            "res_tl":{
                "dims":("cycle"),"data":res[:,1]
            },
            "rmsdJ":{
                "dims":("cycle"),"data":rmsdJ
            },
            "rmsdx":{
                "dims":("cycle"),"data":rmsdx
            },
            "corrdJ":{
                "dims":("cycle"),"data":corrdJ
            },
        }
    ds = xr.Dataset.from_dict(datadict)
    print(ds)
    ds.to_netcdf(savedir/f"{key}_vt{vt}nens{nens}.nc")