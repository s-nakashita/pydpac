import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pathlib import Path 
import sys
from scipy.stats import ttest_ind
alpha_95 = 0.05
alpha_99 = 0.01

plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 20

figdir_var = Path('work/baxter11.3')
figdir_envar = Path('work/baxter11_en.3')
#figdir_var = Path('/Volumes/FF520/nested_envar/data/baxter11.2')
#figdir_envar = Path('/Volumes/FF520/nested_envar/data/baxter11_en.2')

## Domain and step size definition
L = 1.0
T = 0.5
dx = 0.0625
dt = 0.05
dx_t = dx / 8.
dt_t = dt / 64.
dx_gm = dx
dt_gm = dt
dx_lam = dx / 4.
dt_lam = dt / 16.
lamstep = int(dt_gm / dt_lam)
obsstep = int(dt_lam / dt_t)
nx_t = int(L / dx_t)
ix_t = np.arange(1,nx_t+1)
x_t = np.linspace(dx_t,L,nx_t)
nx_gm = int(L / dx_gm)
ix_gm = np.arange(1,nx_gm+1)*int(nx_t/nx_gm)
x_gm = np.linspace(dx_gm,L,nx_gm)
Ls_lam = 0.5
nx_lam = int((L - Ls_lam) / dx_lam) + 1
x_lam = np.linspace(Ls_lam,L,nx_lam)
ix_lam = x_lam * nx_t / L
nsponge = 3
print(f"nx_t={nx_t} nx_gm={nx_gm} nx_lam={nx_lam}")
print(f"dx_t={dx_t} dx_gm={dx_gm} dx_lam={dx_lam}")
print(f"dt_t={dt_t} dt_gm={dt_gm} dt_lam={dt_lam}")
print(f"ix_t={ix_t}")
print(f"ix_gm={ix_gm}")
print(f"ix_lam={ix_lam}")
#
obsloc = ix_lam[1:-1:4]
xobsloc = x_lam[1:-1:4]
if len(sys.argv)>1:
    intobs = int(sys.argv[1])
    obsloc = ix_lam[1:-1:intobs]
    xobsloc = x_lam[1:-1:intobs]
nobs = obsloc.size
nmemlist = [40, 80, 120, 160, 320, 640, 960]
ntrial = 50

errbfile = f'errb_nobs{nobs}.csv'
errafile = f'erra_nobs{nobs}.csv'
erra_nestfile = f"erra_nest_nobs{nobs}.csv"
df_b = pd.read_csv(figdir_var / errbfile,index_col=0)
df_a = pd.read_csv(figdir_var / errafile,index_col=0)
df_a_nest = pd.read_csv(figdir_var / erra_nestfile,index_col=0)
err_b = df_b.values
rmse_b = err_b[:, 0]
errspecb = err_b[:, 1:]
err_a = df_a.values
rmse_a = err_a[:, 0]
errspeca = err_a[:, 1:]
erra_nest = df_a_nest.values
rmsea_nest = erra_nest[:, 0]
errspeca_nest = erra_nest[:, 1:]

espb  = np.zeros((4,50))
espbm = np.zeros(4)
espbs = np.zeros(4)
espb[0,] = errspecb[:,0]
espbm[0] = errspecb[:,0].mean()
espbs[0] = errspecb[:,0].std()
espb[1,] = np.mean(errspecb[:,1:7],axis=1)
espbm[1] = np.mean(errspecb[:,1:7],axis=1).mean()
espbs[1] = np.mean(errspecb[:,1:7],axis=1).std()
espb[2,] = np.mean(errspecb[:,7:17],axis=1)
espbm[2] = np.mean(errspecb[:,7:17],axis=1).mean()
espbs[2] = np.mean(errspecb[:,7:17],axis=1).std()
espb[3,] = errspecb[:,17]
espbm[3] = errspecb[:,17].mean()
espbs[3] = errspecb[:,17].std()
espa  = np.zeros((4,50))
espam = np.zeros(4)
espas = np.zeros(4)
espa[0,] = errspeca[:,0]
espam[0] = errspeca[:,0].mean()
espas[0] = errspeca[:,0].std()
espa[1,] = np.mean(errspeca[:,1:7],axis=1)
espam[1] = np.mean(errspeca[:,1:7],axis=1).mean()
espas[1] = np.mean(errspeca[:,1:7],axis=1).std()
espa[2,] = np.mean(errspeca[:,7:17],axis=1)
espam[2] = np.mean(errspeca[:,7:17],axis=1).mean()
espas[2] = np.mean(errspeca[:,7:17],axis=1).std()
espa[3,] = errspeca[:,17]
espam[3] = errspeca[:,17].mean()
espas[3] = errspeca[:,17].std()
espa_nest  = np.zeros((4,50))
espa_nestm = np.zeros(4)
espa_nests = np.zeros(4)
espa_nest[0,] = errspeca_nest[:,0]
espa_nestm[0] = errspeca_nest[:,0].mean()
espa_nests[0] = errspeca_nest[:,0].std()
espa_nest[1,] = np.mean(errspeca_nest[:,1:7],axis=1)
espa_nestm[1] = np.mean(errspeca_nest[:,1:7],axis=1).mean()
espa_nests[1] = np.mean(errspeca_nest[:,1:7],axis=1).std()
espa_nest[2,] = np.mean(errspeca_nest[:,7:17],axis=1)
espa_nestm[2] = np.mean(errspeca_nest[:,7:17],axis=1).mean()
espa_nests[2] = np.mean(errspeca_nest[:,7:17],axis=1).std()
espa_nest[3,] = errspeca_nest[:,17]
espa_nestm[3] = errspeca_nest[:,17].mean()
espa_nests[3] = errspeca_nest[:,17].std()
fig, ax = plt.subplots(figsize=[8,6],constrained_layout=True)
width=0.15
xaxis = np.arange(4)-width
ax2 = ax.twinx()
ax.bar(xaxis[:-1],espbm[:-1],yerr=espbs[:-1],width=width,label=f'Background={rmse_b.mean():.3f}')
ax2.bar(xaxis[-1],espbm[-1],yerr=espbs[-1],width=width) #,label=f'Background={rmse_b.mean():.3f}')
xaxis=xaxis+width
ax.bar(xaxis[:-1],espam[:-1],yerr=espas[:-1],width=width,label=f'3DVar={rmse_a.mean():.3f}')
ax2.bar(xaxis[-1],espam[-1],yerr=espas[-1],width=width) #,label=f'3DVar={rmse_a.mean():.3f}')
xaxis=xaxis+width
ax.bar(xaxis[:-1],espa_nestm[:-1],yerr=espa_nests[:-1],width=width,label=f'Nested 3DVar={rmsea_nest.mean():.3f}')
ax2.bar(xaxis[-1],espa_nestm[-1],yerr=espa_nests[-1],width=width) #,label=f'Nested 3DVar={rmsea_nest.mean():.3f}')
ax2.vlines([2.5],0,1,colors='gray',ls='dotted',transform=ax2.get_xaxis_transform())
ax.set_xticks(np.arange(4))
ax.set_xticklabels(['1',r'2$-$7',r'8$-$17','18'])
ax.legend(title='Domain RMSE',loc='upper center')
ax.set_xlabel('wavenumber')
ax.set_ylabel(r'Absolute error of k$\leq$17')
ax.set_ylim(0.0,0.125)
ax2.set_ylabel('Absolute error of k=18')
ax2.set_ylim(0.0,1.05)
fig.savefig(figdir_var/f'comp_errspec_nobs{nobs}.png',dpi=300)
fig.savefig(figdir_var/f'comp_errspec_nobs{nobs}.pdf')
plt.show()
plt.close()
#
# t-test
print(f"t-test for LAM - LAM_nest, nobs={nobs} 3DVar")
t_value, p_value = ttest_ind(rmse_a,rmsea_nest)
ratio = (rmse_a - rmsea_nest)/rmse_a * 100.0
print(f"RMSE {ratio.mean():.3f} {p_value:.3e} {p_value<alpha_95} {p_value<alpha_99}")
for k, label in enumerate(ax.get_xticklabels()):
    t_value, p_value = ttest_ind(espa[k,],espa_nest[k,])
    ratio = (espa[k,] - espa_nest[k,])/espa[k,] * 100.0
    print(f"{label} {ratio.mean():.3f} {p_value:.3e} {p_value<alpha_95} {p_value<alpha_99}")
print("")
#
for nmem in nmemlist:
    errbfile = f'errb_nobs{nobs}nmem{nmem}.csv'
    errafile = f'erra_nobs{nobs}nmem{nmem}.csv'
    erra_nestfile = f"erra_nest_nobs{nobs}nmem{nmem}.csv"
    erra_nestcfile = f"erra_nestc_nobs{nobs}nmem{nmem}.csv"
    df_b = pd.read_csv(figdir_envar / errbfile,index_col=0)
    df_a = pd.read_csv(figdir_envar / errafile,index_col=0)
    df_a_nest = pd.read_csv(figdir_envar / erra_nestfile,index_col=0)
    df_a_nestc = pd.read_csv(figdir_envar / erra_nestcfile,index_col=0)
    err_b = df_b.values
    rmse_b = err_b[:, 0]
    errspecb = err_b[:, 1:]
    err_a = df_a.values
    rmse_a = err_a[:, 0]
    errspeca = err_a[:, 1:]
    erra_nest = df_a_nest.values
    rmsea_nest = erra_nest[:, 0]
    errspeca_nest = erra_nest[:, 1:]
    erra_nestc = df_a_nestc.values
    rmsea_nestc = erra_nestc[:, 0]
    errspeca_nestc = erra_nestc[:, 1:]

    espb  = np.zeros((4,50))
    espbm = np.zeros(4)
    espbs = np.zeros(4)
    espb[0,] = errspecb[:,0]
    espbm[0] = errspecb[:,0].mean()
    espbs[0] = errspecb[:,0].std()
    espb[1,] = np.mean(errspecb[:,1:7],axis=1)
    espbm[1] = np.mean(errspecb[:,1:7],axis=1).mean()
    espbs[1] = np.mean(errspecb[:,1:7],axis=1).std()
    espb[2,] = np.mean(errspecb[:,7:17],axis=1)
    espbm[2] = np.mean(errspecb[:,7:17],axis=1).mean()
    espbs[2] = np.mean(errspecb[:,7:17],axis=1).std()
    espb[3,] = errspecb[:,17]
    espbm[3] = errspecb[:,17].mean()
    espbs[3] = errspecb[:,17].std()
    espa = np.zeros((4,50))
    espam = np.zeros(4)
    espas = np.zeros(4)
    espa[0,] = errspeca[:,0]
    espam[0] = errspeca[:,0].mean()
    espas[0] = errspeca[:,0].std()
    espa[1,] = np.mean(errspeca[:,1:7],axis=1)
    espam[1] = np.mean(errspeca[:,1:7],axis=1).mean()
    espas[1] = np.mean(errspeca[:,1:7],axis=1).std()
    espa[2,] = np.mean(errspeca[:,7:17],axis=1)
    espam[2] = np.mean(errspeca[:,7:17],axis=1).mean()
    espas[2] = np.mean(errspeca[:,7:17],axis=1).std()
    espa[3,] = errspeca[:,17]
    espam[3] = errspeca[:,17].mean()
    espas[3] = errspeca[:,17].std()
    espa_nest = np.zeros((4,50))
    espa_nestm = np.zeros(4)
    espa_nests = np.zeros(4)
    espa_nest[0,] = errspeca_nest[:,0]
    espa_nestm[0] = errspeca_nest[:,0].mean()
    espa_nests[0] = errspeca_nest[:,0].std()
    espa_nest[1,] = np.mean(errspeca_nest[:,1:7],axis=1)
    espa_nestm[1] = np.mean(errspeca_nest[:,1:7],axis=1).mean()
    espa_nests[1] = np.mean(errspeca_nest[:,1:7],axis=1).std()
    espa_nest[2,] = np.mean(errspeca_nest[:,7:17],axis=1)
    espa_nestm[2] = np.mean(errspeca_nest[:,7:17],axis=1).mean()
    espa_nests[2] = np.mean(errspeca_nest[:,7:17],axis=1).std()
    espa_nest[3,] = errspeca_nest[:,17]
    espa_nestm[3] = errspeca_nest[:,17].mean()
    espa_nests[3] = errspeca_nest[:,17].std()
    espa_nestc = np.zeros((4,50))
    espa_nestcm = np.zeros(4)
    espa_nestcs = np.zeros(4)
    espa_nestc[0,] = errspeca_nestc[:,0]
    espa_nestcm[0] = errspeca_nestc[:,0].mean()
    espa_nestcs[0] = errspeca_nestc[:,0].std()
    espa_nestc[1,] = np.mean(errspeca_nestc[:,1:7],axis=1)
    espa_nestcm[1] = np.mean(errspeca_nestc[:,1:7],axis=1).mean()
    espa_nestcs[1] = np.mean(errspeca_nestc[:,1:7],axis=1).std()
    espa_nestc[2,] = np.mean(errspeca_nestc[:,7:17],axis=1)
    espa_nestcm[2] = np.mean(errspeca_nestc[:,7:17],axis=1).mean()
    espa_nestcs[2] = np.mean(errspeca_nestc[:,7:17],axis=1).std()
    espa_nestc[3,] = errspeca_nestc[:,17]
    espa_nestcm[3] = errspeca_nestc[:,17].mean()
    espa_nestcs[3] = errspeca_nestc[:,17].std()
    fig, ax = plt.subplots(figsize=[9,6],constrained_layout=True)
    width=0.15
    ax2 = ax.twinx()
    #xaxis = np.arange(4)-width
    xaxis = np.arange(4)-1.5*width
    ax.bar(xaxis[:-1],espbm[:-1],yerr=espbs[:-1],width=width,label=f'Background={rmse_b.mean():.3f}')
    ax2.bar(xaxis[-1],espbm[-1],yerr=espbs[-1],width=width) #,label=f'Background={rmse_b.mean():.3f}')
    xaxis=xaxis+width
    ax.bar(xaxis[:-1],espam[:-1],yerr=espas[:-1],width=width,label=f'EnVar={rmse_a.mean():.3f}')
    ax2.bar(xaxis[-1],espam[-1],yerr=espas[-1],width=width) #,label=f'EnVar={rmse_a.mean():.3f}')
    xaxis=xaxis+width
    ax.bar(xaxis[:-1],espa_nestm[:-1],yerr=espa_nests[:-1],width=width,label=f'Nested EnVar={rmsea_nest.mean():.3f}')
    ax2.bar(xaxis[-1],espa_nestm[-1],yerr=espa_nests[-1],width=width) #,label=f'Nested EnVar={rmsea_nest.mean():.3f}')
    xaxis=xaxis+width
    ax.bar(xaxis[:-1],espa_nestcm[:-1],yerr=espa_nestcs[:-1],width=width,label=f'Nested EnVar_c={rmsea_nestc.mean():.3f}')
    ax2.bar(xaxis[-1],espa_nestcm[-1],yerr=espa_nestcs[-1],width=width) #,label=f'Nested EnVar={rmsea_nest.mean():.3f}')
    ax2.vlines([2.5],0,1,colors='gray',ls='dotted',transform=ax2.get_xaxis_transform())
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(['1',r'2$-$7',r'8$-$17','18'])
    ax.legend(title='Domain RMSE',loc='upper center')
    ax.set_xlabel('wavenumber')
    ax.set_ylabel(r'Absolute error of k$\leq$17')
    ax.set_ylim(0.0,0.125)
    ax2.set_ylabel('Absolute error of k=18')
    ax2.set_ylim(0.0,1.05)
    fig.savefig(figdir_envar/f'comp_errspec_nobs{nobs}nmem{nmem}.png',dpi=300)
    fig.savefig(figdir_envar/f'comp_errspec_nobs{nobs}nmem{nmem}.pdf')
    plt.show()
    plt.close()
    #
    # t-test
    print(f"t-test for LAM - LAM_nest, nobs={nobs} EnVar{nmem}")
    t_value, p_value = ttest_ind(rmse_a,rmsea_nest)
    ratio = (rmse_a - rmsea_nest)/rmse_a * 100.0
    print(f"RMSE {ratio.mean():.3f} {p_value:.3e} {p_value<alpha_95} {p_value<alpha_99}")
    for k, label in enumerate(ax.get_xticklabels()):
        t_value, p_value = ttest_ind(espa[k,],espa_nest[k,])
        ratio = (espa[k,] - espa_nest[k,])/espa[k,] * 100.0
        print(f"{label} {ratio.mean():.3f} {p_value:.3e} {p_value<alpha_95} {p_value<alpha_99}")
    print("")
