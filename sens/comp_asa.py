import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
from asa import ASA
from enasa import EnASA
import sys
sys.path.append('../model')
from lorenz import L96

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
    xd = np.roll(x,nxh-ic,axis=0)[i0:i1] - np.roll(xa,nxh-ic,axis=0)[i0:i1]
    return 0.5*np.dot(xd,xd)
def jac(x,*args):
    xa, ic, hwidth = args
    nxh = x.size // 2
    i0 = nxh - hwidth
    i1 = nxh + hwidth + 1
    dJdxtmp = np.zeros_like(x)
    dJdxtmp[i0:i1] = np.roll(x,nxh-ic,axis=0)[i0:i1] - np.roll(xa,nxh-ic,axis=0)[i0:i1]
    dJdx = np.roll(dJdxtmp,-nxh+ic,axis=0)
    return dJdx

# load data
modelname = 'l96'
pt = 'letkf'
nens = 20
if len(sys.argv)>2:
    nens=int(sys.argv[2])
if nens==200:
    datadir = Path(f'/Volumes/dandelion/pyesa/data/{modelname}/extfcst_letkf_m{nens}')
else:
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
#solverlist=['minnorm','diag','pcr','ridge','pls']
solverlist=['pcr','pls']
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
Jes = []
for i in range(nsample):
    icyc = icyc0 + i
    cycles.append(icyc)
    xa = xf00[icyc+ioffset].mean(axis=1)
    #xa[:] = 0.0
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
    Jes.append(Je)
    Jeest = {}

    for solver in solverlist:
        enasa = EnASA(vt,X0,Je,solver=solver)
        dJedx0 = enasa()
        dxe0opt = asa.calc_dxopt(xb,dJedx0)
        dJdx0_dict[solver].append(dJedx0)
        dx0opt_dict[solver].append(dxe0opt)
        res_nl, res_tl = asa.check_djdx(xb,dJedx0,dxe0opt,\
            model,model.step_t,plot=False)
        res_dict[solver].append([res_nl,res_tl])
        Jeest[solver] = enasa.estimate()
        rmsdJ_dict[solver].append(np.sqrt(np.mean((dJedx0-dJdx0)**2)))
        rmsdx_dict[solver].append(np.sqrt(np.mean((dxe0opt-dx0opt)**2)))
        corr=np.correlate(dJedx0,dJdx0)/np.linalg.norm(dJedx0,ord=2)/np.linalg.norm(dJdx0,ord=2)
        corrdJ_dict[solver].append(corr[0])
        if i==0: print(f"{solver} score={enasa.score()}")
    if i<0:
        print(f"ic={ic}")
        fig, axs = plt.subplots(nrows=3,sharex=True,figsize=[8,8],constrained_layout=True)
        axs[0].plot(xa,label='analysis')
        axs[0].plot(xf,label='forecast')
        axs[0].plot(xf-xa,ls='dotted',label='diff')
        for j,key in enumerate(dJdx0_dict.keys()):
            if key=='asa':
                axs[1].plot(dJdx0_dict[key][i],label=key,lw=2.0)
                axs[2].plot(dx0opt_dict[key][i],label=key,lw=2.0)
            else:
                axs[1].plot(dJdx0_dict[key][i],ls='dashed',marker=markers[j],label=f'EnASA,{key}',**marker_style)
                axs[2].plot(dx0opt_dict[key][i],ls='dashed',marker=markers[j],label=f'EnASA,{key}',**marker_style)
        for ax in axs:
            ax.vlines([ic],0,1,colors='r',transform=ax.get_xaxis_transform())
            ax.legend(loc='upper left',bbox_to_anchor=(1.01,1.0))
            ax.grid()
        axs[1].set_title('dJ/dx0')
        axs[2].set_title('dxopt')
        fig.suptitle(f'vt={vt}h, Nens={nens}')
        plt.show()

        fig, axs = plt.subplots(ncols=2,constrained_layout=True)
        for i,key in enumerate(Jeest.keys()):
            if key=='diag':
                axs[0].plot(Je,Jeest[key],lw=0.0,marker=markers[i],c=cmap(i+1),label=key,**marker_style)
            else:
                axs[1].plot(Je,Jeest[key],lw=0.0,marker=markers[i],c=cmap(i+1),label=key,**marker_style)
        for ax in axs:
            ymin, ymax = ax.get_ylim()
            line = np.linspace(ymin,ymax,100)
            ax.plot(line,line,color='k',zorder=0)
            ax.set_xlabel('observed')
            ax.set_ylabel('estimated')
            ax.set_title(f'Je, vt={vt}h, Nens={nens}')
            ax.legend()
            #ax.set_title(key)
            ax.grid()
            ax.set_aspect(1.0)
        plt.show()
if nsample < 1000: exit()

# save results to netcdf
savedir = Path('data')
if not savedir.exists(): savedir.mkdir(parents=True)
Jes = np.array(Jes)
member = np.arange(1,Jes.shape[1]+1)
ds = xr.Dataset.from_dict(
    {
        "cycle":{"dims":("cycle"),"data":cycles},
        "member":{"dims":("member"),"data":member},
        "Je":{"dims":("cycle","member"),"data":Jes}
    }
)
ds.to_netcdf(savedir/f"Je_vt{vt}nens{nens}.nc")
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