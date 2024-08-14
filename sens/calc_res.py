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
resuni_dict={} #univariate
resmul_dict={} #multivariate

# ASA data
ds_asa = xr.open_dataset(savedir/f'asa{metric}_vt{vt}nens{nensbase}.nc')
dJdx0_dict['asa'] = ds_asa.dJdx0.values
resuni_dict['asa'] = []
resmul_dict['asa'] = []
solverlist=['minnorm','diag','ridge','pcr','pls']
#solverlist=['pcr','minnorm'] #,'pls']
for key in solverlist:
    if (key == 'pls' or key == 'pcr' or key == 'minnorm') and n_components is not None:
        fname=f"{key}nc{n_components}{metric}_vt{vt}nens{nens}.nc"
    else:
        fname=f"{key}{metric}_vt{vt}nens{nens}.nc"
    ds_enasa = xr.open_dataset(savedir/fname)
    dJdx0_dict[key] = ds_enasa.dJdx0.values
    resuni_dict[key] = []
    resmul_dict[key] = []
cmap = plt.get_cmap('tab10')
colors = {'asa':cmap(0),'minnorm':cmap(1),'diag':cmap(2),'pcr':cmap(3),'ridge':cmap(4),'pls':cmap(5)}
markers = {'asa':'*','minnorm':'o','diag':'v','pcr':'s','ridge':'P','pls':'X'}
marker_style=dict(markerfacecolor='none')

cycles = []
ics = ds_asa.ic.values
ic_list = ics.tolist()
ix0_list = []
x0i_list = []
resuni_calc = []
resmul_calc = []
ncycle = 100
ngrid = 20
rng = default_rng(seed=509)
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
    sig = np.sqrt(np.diag(X0@X0.T)/(nens-1))
    cor = X0 @ X0.T / sig / sig / (nens-1)
    ix0s = rng.choice(nx,size=ngrid,replace=False)
    for j in range(ngrid):
        # create initial perturbation
        ix0 = ix0s[j]
        ix0_list.append(ix0)
        x0i = np.diag(sig) @ cor[:,ix0]
        x0i_list.append(x0i)
        x0u = np.zeros_like(x0i)
        x0u[ix0] = sig[ix0]
        # calculate nonlinear forecast response
        xi = xb0 + x0i
        xu = xb0 + x0u
        for k in range(vt):
            xi = model(xi)
            xu = model(xu)
        Ji = cost(xi,*args)
        Ju = cost(xu,*args)
        resmul_calc.append(Ji-Jb)
        resuni_calc.append(Ju-Jb)
        # estimate forecast response by sensitivity
        for key in dJdx0_dict.keys():
            dJdx0 = dJdx0_dict[key][i,]
            resmul_dict[key].append(dJdx0@x0i)
            resuni_dict[key].append(dJdx0@x0u)
        nsample += 1
        if i<0:
            print(f"ic={ic} ix0={ix0}")
            plt.plot(x0i,marker='^')
            plt.plot(x0u,lw=0,marker='x')
            plt.show()
            plt.close()
            plt.plot(xf)
            plt.plot(xi,ls='dashed')
            plt.plot(xu,ls='dotted')
            plt.show()
            plt.close()

if nsample < 10: exit()
# save results to netcdf
resuni_calc = np.array(resuni_calc)
resmul_calc = np.array(resmul_calc)
samples = np.arange(1,nsample+1)
ix0 = np.array(ix0_list)
x0i = np.array(x0i_list)
ix = np.arange(x0i.shape[1])
ds = xr.Dataset.from_dict(
    {
        "sample":{"dims":("sample"),"data":samples},
        "x":{"dims":("x"),"data":ix},
        "ix0":{"dims":("sample"),"data":ix0},
        "x0i":{"dims":("sample","x"),"data":x0i},
        "resuni":{"dims":("sample"),"data":resuni_calc},
        "resmul":{"dims":("sample"),"data":resmul_calc}
    }
)
ds.to_netcdf(savedir/f"res_calc{metric}_vt{vt}nens{nens}.nc")

for key in resmul_dict.keys():
    resmul = np.array(resmul_dict[key])
    resuni = np.array(resuni_dict[key])
    ds = xr.Dataset.from_dict(
    {
        "sample":{"dims":("sample"),"data":samples},
        "x":{"dims":("x"),"data":ix},
        "ix0":{"dims":("sample"),"data":ix0},
        "x0i":{"dims":("sample","x"),"data":x0i},
        "resuni":{"dims":("sample"),"data":resuni},
        "resmul":{"dims":("sample"),"data":resmul}
    }
    )
    print(ds)
    if (key == 'pls' or key == 'pcr' or key == 'minnorm') and n_components is not None:
        ds.to_netcdf(savedir/f"res_{key}nc{n_components}{metric}_vt{vt}nens{nens}.nc")
    else:
        ds.to_netcdf(savedir/f"res_{key}{metric}_vt{vt}nens{nens}.nc")