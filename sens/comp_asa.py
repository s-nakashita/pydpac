import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xarray as xr
from asa import ASA
from enasa import EnASA
import argparse
import sys
sys.path.append('../model')
from lorenz import L96

# forecast model
nx = 40
dt = 0.05 / 6 # 1 hour
F = 8.0
ldble = 2.3 * dt # Lyapunov exponent (1/hour)
model = L96(nx,dt,F)

# SA settings
parser = argparse.ArgumentParser()
parser.add_argument("-vt","--vt",type=int,default=24,\
    help="verification time (hours)")
parser.add_argument("-ne","--nens",type=int,default=8,\
    help="ensemble size")
parser.add_argument("-nc","--n_components",type=int,\
    help="(minnorm,pcr,pls) number of components to keep")
parser.add_argument("-m","--metric",type=str,default="",\
    help="forecast metric type")
argsin = parser.parse_args()
vt = argsin.vt # hours
ioffset = vt // 6
nens = argsin.nens
n_components = argsin.n_components
metric = argsin.metric

recomp_asa = False
nensbase = 8

def cost(x,*args):
    xa, ic, hwidth = args
    nxh = x.size // 2
    i0 = nxh - hwidth
    i1 = nxh + hwidth + 1
    xd = np.roll(x-xa,nxh-ic,axis=0)[i0:i1]
    return 0.5*np.dot(xd,xd)
def jac(x,*args):
    xa, ic, hwidth = args
    nxh = x.size // 2
    i0 = nxh - hwidth
    i1 = nxh + hwidth + 1
    dJdxtmp = np.zeros_like(x)
    dJdxtmp[i0:i1] = np.roll(x-xa,nxh-ic,axis=0)[i0:i1]
    dJdx = np.roll(dJdxtmp,-nxh+ic,axis=0)
    return dJdx

# load data
modelname = 'l96'
pt = 'letkf'
if nens==200:
    datadir = Path(f'/Volumes/dandelion/pyesa/data/{modelname}/extfcst_letkf_m{nens}')
else:
    datadir = Path(f'/Volumes/dandelion/pyesa/data/{modelname}/extfcst_m{nens}')
xf00 = np.load(datadir/f"{modelname}_xf00_linear_{pt}.npy")
xfall = np.load(datadir/f"{modelname}_ufext_linear_{pt}.npy")
xfv  = np.load(datadir/f"{modelname}_xf{vt:02d}_linear_{pt}.npy")

savedir = Path('data')
if not savedir.exists(): savedir.mkdir(parents=True)

icyc0 = 50
nsample = 1000
if not recomp_asa:
    dJdx0_dict={}
    dx0opt_dict={}
    res_dict={}
    # ASA data
    ds_asa = xr.open_dataset(savedir/f'asa{metric}_vt{vt}nens{nensbase}.nc')
    dJdx0_dict['asa'] = ds_asa.dJdx0.values
    dx0opt_dict['asa'] = ds_asa.dx0opt.values
    res_dict['asa'] = np.stack([ds_asa.res_nl.values,ds_asa.res_tl.values],axis=1)
else:
    dJdx0_dict={'asa':[]}
    dx0opt_dict={'asa':[]}
    res_dict={'asa':[]}
rmsdJ_dict = {}
rmsdx_dict = {}
corrdJ_dict = {}
solverlist=['minnorm','diag','ridge','pcr','pls']
#solverlist=['pcr','minnorm'] #,'pls']

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
if not recomp_asa:
    ics = ds_asa.ic.values
    ic_list = ics.tolist()
else:
    ic_list = []
Je_list = []
x0s_list = []
for i in range(nsample):
    icyc = icyc0 + i
    cycles.append(icyc)
    xa = xf00[icyc+ioffset].mean(axis=1)
    if metric=='_en': xa[:] = 0.0
    xf = xfv [icyc+ioffset].mean(axis=1)
    if not recomp_asa:
        ic = ic_list[i]
    else:
        ic = np.argmax(np.abs(xa - xf)) # center of verification region
        ic_list.append(ic)
    hwidth = 1 # half-width of verification region
    args = (xa,ic,hwidth)

    # ASA
    asa = ASA(vt,cost,jac,model.step_adj,*args)
    # base trajectory
    nx = xa.size
    xb0 = xf00[icyc].mean(axis=1)
    xb = [xb0]
    xb1 = xb0.copy()
    for j in range(vt):
        xb1 = model(xb1)
        xb.append(xb1)
    if recomp_asa:
        # analysis
        dJdx0 = asa(xb)
        dx0opt = asa.calc_dxopt(xb,dJdx0)
        if metric=='_en':
            scale = 0.1/np.linalg.norm(dx0opt,ord=2)/np.exp((vt-24)*ldble)
            dx0opt = dx0opt * scale
        dJdx0_dict['asa'].append(dJdx0)
        dx0opt_dict['asa'].append(dx0opt)
        #asa.plot_hov()
        res_nl, res_tl = asa.check_djdx(xb,dJdx0,dx0opt,\
            model,model.step_t,plot=False)
        res_dict['asa'].append([res_nl,res_tl])
    else:
        dJdx0 = dJdx0_dict['asa'][i,]
        dx0opt = dx0opt_dict['asa'][i,]
    
    # EnASA
    xe0 = xf00[icyc]
    X0 = xe0 - xe0.mean(axis=1)[:,None]
    x0s_list.append(X0.std(axis=1))
    #xev = xfv[icyc+ioffset]
    xev = xfall[icyc,ioffset,:,:]
    Je = np.zeros(nens)
    for k in range(nens):
        Je[k] = cost(xev[:,k],*args)
    Je_list.append(Je)
    Jem = Je.mean()
    Je = Je - Jem
    Jeest = {}

    for solver in solverlist:
        logfile = f"{solver}_vt{vt}ne{nens}{metric}"
        if n_components is not None:
            logfile = f"{solver}_vt{vt}ne{nens}nc{n_components}{metric}"
        enasa = EnASA(vt,X0,Je,solver=solver,logfile=logfile)
        dJedx0 = enasa(n_components=n_components,cthres=0.99)
        dxe0opt = asa.calc_dxopt(xb,dJedx0)
        if metric=='_en':
            scale = 0.1/np.linalg.norm(dxe0opt,ord=2)/np.exp((vt-24)*ldble)
            dxe0opt = dxe0opt * scale
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
        if i==0: 
            print(f"{solver} score={enasa.score()} err={enasa.err}")
            if solver=='minnorm': print(f"nrank={enasa.nrank}")
    if i<0:
        print(f"ic={ic}")
        figdir = Path(f"fig/vt{vt}ne{nens}{metric}/c{icyc}")
        if n_components is not None:
            figdir = Path(f"fig/vt{vt}ne{nens}nc{n_components}{metric}/c{icyc}")
        if not figdir.exists(): figdir.mkdir(parents=True)
        nxh = xa.size // 2
        fig, axs = plt.subplots(nrows=3,sharex=True,figsize=[8,8],constrained_layout=True)
        axs[0].plot(np.roll(xb0,nxh-ic,axis=0),label='FT00')
        axs[0].plot(np.roll(xa,nxh-ic,axis=0),label='analysis')
        axs[0].plot(np.roll(xf,nxh-ic,axis=0),label=f'FT{vt}')
        axs[0].plot(np.roll(xf-xa,nxh-ic,axis=0),ls='dotted',label='diff')
        for j,key in enumerate(dJdx0_dict.keys()):
            if key=='asa':
                axs[1].plot(np.roll(dJdx0_dict[key][i],nxh-ic),label=key,lw=2.0)
                axs[2].plot(np.roll(dx0opt_dict[key][i],nxh-ic),label=key,lw=2.0)
            else:
                axs[1].plot(np.roll(dJdx0_dict[key][i],nxh-ic),ls='dashed',marker=markers[j],label=f'EnASA,{key}',**marker_style)
                axs[2].plot(np.roll(dx0opt_dict[key][i],nxh-ic),ls='dashed',marker=markers[j],label=f'EnASA,{key}',**marker_style)
        for ax in axs:
            ax.vlines([nxh],0,1,colors='r',transform=ax.get_xaxis_transform())
            ax.legend(loc='upper left',bbox_to_anchor=(1.01,1.0))
            ax.grid()
        axs[1].set_title('dJ/dx0')
        axs[2].set_title('dxopt')
        fig.suptitle(f'vt={vt}h, Nens={nens}')
        fig.savefig(figdir/'x+dJdx0+dxopt.png')
        plt.close()
        fig, axs = plt.subplots(nrows=2,sharex=True,figsize=[8,6],constrained_layout=True)
        axs[0].plot(np.roll(xb0,nxh-ic,axis=0),c='k',lw=2.0)
        axs[1].plot(np.roll(xf,nxh-ic,axis=0),c='k',lw=2.0)
        axs[0].plot(np.roll(xe0,nxh-ic,axis=0),c='b',ls='dotted',lw=0.5)
        axs[1].plot(np.roll(xev,nxh-ic,axis=0),c='b',ls='dotted',lw=0.5)
        for ax in axs:
            ax.vlines([nxh],0,1,colors='r',transform=ax.get_xaxis_transform(),zorder=0)
            ax.grid()
        axs[0].set_title('FT00')
        axs[1].set_title(f'FT{vt}')
        fig.suptitle(f'vt={vt}h, Nens={nens}')
        fig.savefig(figdir/'xe.png')
        plt.close()
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
            ax.set_xlabel('observed (centering)')
            ax.set_ylabel('estimated (centering)')
            ax.set_title(f'Je, vt={vt}h, Nens={nens}')
            ax.legend()
            #ax.set_title(key)
            ax.grid()
            ax.set_aspect(1.0)
        fig.savefig(figdir/'Je.png')
        plt.close()
if nsample < 1000: exit()

# save results to netcdf
Jes = np.array(Je_list)
member = np.arange(1,Jes.shape[1]+1)
ds = xr.Dataset.from_dict(
    {
        "cycle":{"dims":("cycle"),"data":cycles},
        "member":{"dims":("member"),"data":member},
        "Je":{"dims":("cycle","member"),"data":Jes}
    }
)
ds.to_netcdf(savedir/f"Je_vt{vt}nens{nens}.nc")
x0s = np.array(x0s_list)
member = np.arange(1,x0s.shape[1]+1)
ds = xr.Dataset.from_dict(
    {
        "cycle":{"dims":("cycle"),"data":cycles},
        "member":{"dims":("member"),"data":member},
        "x0s":{"dims":("cycle","member"),"data":x0s}
    }
)
ds.to_netcdf(savedir/f"x0s_nens{nens}.nc")
if recomp_asa:
    ics = np.array(ic_list)
for key in res_dict.keys():
    res = np.array(res_dict[key])
    dJdx0 = np.array(dJdx0_dict[key])
    dx0opt = np.array(dx0opt_dict[key])
    ix = np.arange(dJdx0.shape[1])
    if key == 'asa':
        if not recomp_asa: continue
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
    if (key == 'pls' or key == 'pcr' or key == 'minnorm') and n_components is not None:
        ds.to_netcdf(savedir/f"{key}nc{n_components}{metric}_vt{vt}nens{nens}.nc")
    else:
        ds.to_netcdf(savedir/f"{key}{metric}_vt{vt}nens{nens}.nc")