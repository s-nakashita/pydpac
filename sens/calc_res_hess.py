import numpy as np
from numpy.random import default_rng
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
dJdx0_dict={}
rescalcp_dict={}
rescalcm_dict={}
resestp_dict={}
resestm_dict={}
x0_dict={}

# ASA data
ds_asa = xr.open_dataset(savedir/f'asa{metric}_vt{vt}nens{nensbase}.nc')
dJdx0_dict['asa'] = ds_asa.dJdx0.values
rescalcp_dict['asa'] = []
rescalcm_dict['asa'] = []
resestp_dict['asa'] = []
resestm_dict['asa'] = []
x0_dict['asa'] = []
solverlist=['minnorm','diag','ridge','pcr','pls']
#solverlist=['pcr','minnorm'] #,'pls']
for key in solverlist:
    if (key == 'pls' or key == 'pcr' or key == 'minnorm') and n_components is not None:
        fname=f"{key}nc{n_components}{metric}_vt{vt}nens{nens}.nc"
    else:
        fname=f"{key}{metric}_vt{vt}nens{nens}.nc"
    ds_enasa = xr.open_dataset(savedir/fname)
    dJdx0_dict[key] = ds_enasa.dJdx0.values
    rescalcp_dict[key] = []
    rescalcm_dict[key] = []
    resestp_dict[key] = []
    resestm_dict[key] = []
    x0_dict[key] = []
cmap = plt.get_cmap('tab10')
colors = {'asa':cmap(0),'minnorm':cmap(1),'diag':cmap(2),'pcr':cmap(3),'ridge':cmap(4),'pls':cmap(5)}
markers = {'asa':'*','minnorm':'o','diag':'v','pcr':'s','ridge':'P','pls':'X'}
marker_style=dict(markerfacecolor='none')

cycles = []
A_list = []
ics = ds_asa.ic.values
ic_list = ics.tolist()
ncycle = 1000
nsample = 0
for i in range(ncycle):
    icyc = icyc0 + i
    cycles.append(icyc)
    xa = xf00[icyc+ioffset].mean(axis=1)
    if metric=='_en': xa[:] = 0.0
    xf = xfv [icyc+ioffset].mean(axis=1)
    ic = ic_list[i]
    hwidth = 1 # half-width of verification region
    args = (xa,ic,hwidth)
    Jb = cost(xf,*args)

    # ASA
    asa = ASA(vt,cost,jac,model.step_adj,*args)

    # initial state
    xe0 = xf00[icyc]
    xb0 = xe0.mean(axis=1)
    X0 = xe0 - xe0.mean(axis=1)[:,None]
    A = X0 @ X0.T / (nens-1)
    A_list.append(A)
    sig = np.sqrt(np.diag(A))
    cor = A / sig / sig
    for key in dJdx0_dict.keys():
        dJdx0 = dJdx0_dict[key][i,]
        # create initial perturbation
        lam = np.sqrt(np.dot(dJdx0,np.dot(A,dJdx0)))
        x0p = A @ dJdx0 / lam
        x0m = - x0p
        x0_dict[key].append(x0p)
        # calculate nonlinear forecast response
        xp = xb0 + x0p
        xm = xb0 + x0m
        for k in range(vt):
            xp = model(xp)
            xm = model(xm)
        Jp = cost(xp,*args)
        Jm = cost(xm,*args)
        rescalcp_dict[key].append((Jp-Jb)/Jb)
        rescalcm_dict[key].append((Jm-Jb)/Jb)
        # estimate forecast response by sensitivity
        resestp_dict[key].append(dJdx0@x0p/Jb)
        resestm_dict[key].append(dJdx0@x0m/Jb)
        if i<0:
            print(f"ic={ic} {key}")
            plt.plot(x0p,marker='^')
            plt.show()
            plt.close()
            plt.plot(xf)
            plt.plot(xp,ls='dashed')
            plt.plot(xm,ls='dotted')
            plt.show()
            plt.close()
    nsample += 1

if nsample < 10: exit()
# save results to netcdf
if vt==24:
    A = np.array(A_list)
    ix = np.arange(A.shape[1])
    ds = xr.Dataset.from_dict(
    {
        "cycle":{"dims":("cycle"),"data":cycles},
        "x1":{"dims":("x1"),"data":ix},
        "x2":{"dims":("x2"),"data":ix},
        "A":{"dims":("cycle","x1","x2"),"data":A},
    }
    )
    ds.to_netcdf(savedir/f"A_nens{nens}.nc")

for key in rescalcp_dict.keys():
    rescalcp = np.array(rescalcp_dict[key])
    rescalcm = np.array(rescalcm_dict[key])
    resestp = np.array(resestp_dict[key])
    resestm = np.array(resestm_dict[key])
    x0 = np.array(x0_dict[key])
    ix = np.arange(x0.shape[1])
    ds = xr.Dataset.from_dict(
    {
        "cycle":{"dims":("cycle"),"data":cycles},
        "x":{"dims":("x"),"data":ix},
        "x0":{"dims":("cycle","x"),"data":x0},
        "rescalcp":{"dims":("cycle"),"data":rescalcp},
        "rescalcm":{"dims":("cycle"),"data":rescalcm},
        "resestp":{"dims":("cycle"),"data":resestp},
        "resestm":{"dims":("cycle"),"data":resestm},
    }
    )
    print(ds)
    if (key == 'pls' or key == 'pcr' or key == 'minnorm') and n_components is not None:
        ds.to_netcdf(savedir/f"res_hess_{key}nc{n_components}{metric}_vt{vt}nens{nens}.nc")
    else:
        ds.to_netcdf(savedir/f"res_hess_{key}{metric}_vt{vt}nens{nens}.nc")