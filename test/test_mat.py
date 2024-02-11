import numpy as np 
import numpy.linalg as la
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 16
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import os
from pathlib import Path
import shutil
import sys
from test_ncm import ncm
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append('../')
from l05nest import nx_true, nx_gm, nk_gm, nx_lam, nsp, po, nk_lam, ni, b, c, dt_gm, F, intgm, ist_lam, nsp
sys.path.append('../model')
from lorenz3 import L05III
from lorenz_nest import L05nest
step_true = L05III(nx_true,nk_lam,ni,b,c,dt_gm,F)
step = L05nest(nx_true, nx_gm, nx_lam, nk_gm, nk_lam, ni, b, c, dt_gm, F, intgm, ist_lam, nsp)

figdir = '.'
functype = "tri" # gauss or gc5 or tri
figdir2 = f'{figdir}/{functype}'
if not os.path.isdir(figdir2):
    os.mkdir(figdir2)
#cmap = plt.get_cmap('tab20')
cmap = plt.get_cmap('viridis')

lblist = (np.linspace(0,0.1,21)).tolist()
intcol = cmap.N // (len(lblist) - 1) // 2
distmax = 2.0*np.pi
#distmax = nx_true / np.pi
##true GM (cyclic)
dist = np.eye(nx_true)
for i in range(nx_true):
    dist[i,:] = step_true.calc_dist(i)
    #dist[i,:] = 2.0 * np.pi * dist[i,:] / nx_true
fig, ax = plt.subplots(figsize=[4,4],constrained_layout=True)
mp=ax.matshow(dist)
fig.colorbar(mp,ax=ax,shrink=0.6,pad=0.01)
ax.set_title(f"distance")
fig.savefig(f"{figdir}/dist_true.png",dpi=300)
plt.close()
""" test
for lb in lblist[1:]:
    for f in ["gauss","gc5","tri"]:
        if f=="gauss":
            cmat = np.exp(-0.5*(dist/lb/distmax)**2)
        elif f=="gc5":
            z = dist / lb / distmax / np.sqrt(10.0/3.0)
            cmat = np.where(z<1.0, 1.0 - 5.0*(z**2)/3.0 + 0.625*(z**3) + 0.5*(z**4) - 0.25*(z**5), np.where(z<2.0, 4.0 - 5.0*z + 5.0*(z**2)/3.0 + 0.625*(z**3) - 0.5*(z**4) + (z**5)/12.0 - 2.0/z/3.0, 0.0))
        elif f=="tri":
            nj = np.sqrt(3.0/10.0) / lb
            print(f"lb={lb:.3f} nj={nj}")
            cmat = np.where(dist==0.0,1.0,np.sin(nj*dist/2.0)/np.tan(dist/2.0)/nj)
        plt.plot(cmat[:,nx_true//2],label=f)
    plt.title(f"lb={lb:.3f}")
    plt.legend()
    plt.savefig(f"{figdir}/funccomp_lb{lb:.3f}.png",dpi=300)
    plt.show()
    plt.close()
#exit()
"""
condlist = []
eiglist = []
for lb in lblist:
    if lb>0:
        if functype=="gauss":
            cmat = np.exp(-0.5*(dist/lb/distmax)**2)
        elif functype=="gc5":
            z = dist / lb / distmax / np.sqrt(10.0/3.0)
            cmat = np.where(z<1.0, 1.0 - 5.0*(z**2)/3.0 + 0.625*(z**3) + 0.5*(z**4) - 0.25*(z**5), np.where(z<2.0, 4.0 - 5.0*z + 5.0*(z**2)/3.0 + 0.625*(z**3) - 0.5*(z**4) + (z**5)/12.0 - 2.0/z/3.0, 0.0))
        elif functype=="tri":
            nj = np.sqrt(3.0/10.0) / lb
            print(f"lb={lb:.3f} nj={nj}")
            cmat = np.where(dist==0.0,1.0,np.sin(nj*dist/2.0)/np.tan(dist/2.0)/nj)
    else:
        cmat = np.eye(dist.shape[0])
    condlist.append(la.cond(cmat))
    fig, ax = plt.subplots(figsize=[4,4],constrained_layout=True)
    p = ax.matshow(cmat,vmin=-1.0,vmax=1.0,cmap='bwr')
    fig.colorbar(p,ax=ax,shrink=0.6,pad=0.01)
    ax.set_title(f"lb={lb:.3f}"+r"$\times 2\pi$")
    #ax.set_title(f"lb={lb:.1f}"+r"$\times N/\pi$")
    fig.savefig(f"{figdir2}/mat_true_lb{lb:.3f}.png",dpi=300)
    plt.close()
    eival, eivec = la.eigh(cmat)
    eiglist.append(eival[::-1])
fig, ax = plt.subplots(figsize=[10,8])
ax.plot(lblist,condlist)
ax.set_yscale("log")
ax.set_title(f"condition number, functype={functype}")
ax.set_xlabel(r"$L_b$") # for $\exp (-\frac{1}{2}(\frac{d}{2\pi\times L_b})^2)$")
#ax.set_xlabel(r"$L_b$ for $\exp (-\frac{1}{2}(\frac{d}{L_b\times N/\pi})^2)$")
fig.savefig(f"{figdir}/matcond{functype}_true.png",dpi=300)
#plt.show()
plt.close()
fig, ax = plt.subplots(figsize=[10,8])
for i,lb in enumerate(lblist):
    ev = eiglist[i]
    ax.plot(np.arange(1,ev.size+1),ev,lw=0.0,c=cmap(2*i*intcol),
    marker='x',label=f'lb={lb:.3f}')
    if lb>0 and functype=="gauss":
        knum = np.zeros(ev.size)
        for j in range(1,int(ev.size/2)):
            knum[2*j-1:2*j+1] = float(j)
        knum[-1] = ev.size / 2.0
        d = np.sqrt(2.0) / lb / distmax
        ev_anl = ev.size*np.exp(-knum*knum/d/d)/np.sum(np.exp(-knum*knum/d/d))
        #ax.plot(np.arange(1,ev.size+1),ev_anl,c=cmap(2*i*intcol+1))
        print(f"cond(calc)={condlist[i]} cond(anl)={np.sqrt(ev_anl[0]/ev_anl[-1])}")
#ax.legend(ncol=3)
fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(lblist[0],lblist[-1]), cmap=cmap),ax=ax,label="lb")
ax.set_yscale("log")
ax.set_xlabel("mode")
ax.set_title(f"eigenvalues, functype={functype}")
fig.savefig(f"{figdir}/mateig{functype}_true.png",dpi=300)
#plt.show()
plt.close()
#exit()
##GM (cyclic)
dist = np.eye(nx_gm)
for i in range(nx_gm):
    dist[i,:] = step.calc_dist_gm(i)
    #dist[i,:] = 2.0 * np.pi * dist[i,:] / nx_true
fig, ax = plt.subplots(figsize=[4,4],constrained_layout=True)
mp=ax.matshow(dist)
fig.colorbar(mp,ax=ax,shrink=0.6,pad=0.01)
ax.set_title(f"distance")
fig.savefig(f"{figdir}/dist_gm.png",dpi=300)
plt.close()
condlist = []
eiglist = []
for lb in lblist:
    if lb>0:
        if functype=="gauss":
            cmat = np.exp(-0.5*(dist/lb/distmax)**2)
        elif functype=="gc5":
            z = dist / lb / distmax / np.sqrt(10.0/3.0)
            cmat = np.where(z<1.0, 1.0 - 5.0*(z**2)/3.0 + 0.625*(z**3) + 0.5*(z**4) - 0.25*(z**5), np.where(z<2.0, 4.0 - 5.0*z + 5.0*(z**2)/3.0 + 0.625*(z**3) - 0.5*(z**4) + (z**5)/12.0 - 2.0/z/3.0, 0.0))
        elif functype=="tri":
            nj = np.sqrt(3.0/10.0) / lb
            print(f"lb={lb:.3f} nj={nj}")
            cmat = np.where(dist==0.0,1.0,np.sin(nj*dist/2.0)/np.tan(dist/2.0)/nj)
    else:
        cmat = np.eye(dist.shape[0])
    condlist.append(la.cond(cmat))
    fig, ax = plt.subplots(figsize=[4,4],constrained_layout=True)
    p = ax.matshow(cmat,vmin=-1.0,vmax=1.0,cmap='bwr')
    fig.colorbar(p,ax=ax,shrink=0.6,pad=0.01)
    ax.set_title(f"lb={lb:.3f}"+r"$\times 2\pi$")
    #ax.set_title(f"lb={lb:.1f}"+r"$\times N/\pi$")
    fig.savefig(f"{figdir2}/mat_gm_lb{lb:.3f}.png",dpi=300)
    plt.close()
    eival, eivec = la.eigh(cmat)
    eiglist.append(eival[::-1])
fig, ax = plt.subplots(figsize=[10,8])
ax.plot(lblist,condlist)
ax.set_yscale("log")
ax.set_title(f"condition number, functype={functype}")
ax.set_xlabel(r"$L_b$") # for $\exp (-\frac{1}{2}(\frac{d}{2\pi\times L_b})^2)$")
#ax.set_xlabel(r"$L_b$ for $\exp (-\frac{1}{2}(\frac{d}{L_b\times N/\pi})^2)$")
fig.savefig(f"{figdir}/matcond{functype}_gm.png",dpi=300)
#plt.show()
plt.close()
fig, ax = plt.subplots(figsize=[10,8])
for i,lb in enumerate(lblist):
    ev = eiglist[i]
    ax.plot(np.arange(1,ev.size+1),ev,lw=0.0,c=cmap(2*i*intcol),marker='x',label=f'lb={lb:.3f}')
    if lb>0 and functype=="gauss":
        knum = np.zeros(ev.size)
        for j in range(1,int(ev.size/2)):
            knum[2*j-1:2*j+1] = float(j)
        knum[-1] = ev.size / 2.0
        d = np.sqrt(2.0) / lb / distmax
        ev_anl = ev.size*np.exp(-knum*knum/d/d)/np.sum(np.exp(-knum*knum/d/d))
        #ax.plot(np.arange(1,ev.size+1),ev_anl,c=cmap(2*i*intcol+1))
        print(f"cond(calc)={condlist[i]} cond(anl)={np.sqrt(ev_anl[0]/ev_anl[-1])}")
#ax.legend(ncol=3)
fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(lblist[0],lblist[-1]), cmap=cmap),ax=ax,label="lb")
ax.set_yscale("log")
ax.set_xlabel("mode")
ax.set_title(f"eigenvalues, functype={functype}")
fig.savefig(f"{figdir}/mateig{functype}_gm.png",dpi=300)
#plt.show()
plt.close()
##LAM (noncyclic)
dist = np.eye(nx_lam)
for i in range(nx_lam):
    dist[i,:] = step.calc_dist_lam(i)
    #dist[i,:] = 2.0 * np.pi * dist[i,:] / nx_true
fig, ax = plt.subplots(figsize=[4,4],constrained_layout=True)
mp=ax.matshow(dist)
fig.colorbar(mp,ax=ax,shrink=0.6,pad=0.01)
ax.set_title(f"distance")
fig.savefig(f"{figdir}/dist_lam.png",dpi=300)
plt.close()
condlist = []
eiglist = []
for lb in lblist:
    if lb>0:
        if functype=="gauss":
            cmat = np.exp(-0.5*(dist/lb/distmax)**2)
        elif functype=="gc5":
            z = dist / lb / distmax / np.sqrt(10.0/3.0)
            cmat = np.where(z<1.0, 1.0 - 5.0*(z**2)/3.0 + 0.625*(z**3) + 0.5*(z**4) - 0.25*(z**5), np.where(z<2.0, 4.0 - 5.0*z + 5.0*(z**2)/3.0 + 0.625*(z**3) - 0.5*(z**4) + (z**5)/12.0 - 2.0/z/3.0, 0.0))
        elif functype=="tri":
            nj = np.sqrt(3.0/10.0) / lb
            print(f"lb={lb:.3f} nj={nj}")
            cmat = np.where(dist==0.0,1.0,np.sin(nj*dist/2.0)/np.tan(dist/2.0)/nj)
    else:
        cmat = np.eye(dist.shape[0])
    condlist.append(la.cond(cmat))
    fig, ax = plt.subplots(figsize=[4,4],constrained_layout=True)
    p = ax.matshow(cmat,vmin=-1.0,vmax=1.0,cmap='bwr')
    fig.colorbar(p,ax=ax,shrink=0.6,pad=0.01)
    ax.set_title(f"lb={lb:.3f}"+r"$\times 2\pi$")
    #ax.set_title(f"lb={lb:.1f}"+r"$\times N/\pi$")
    fig.savefig(f"{figdir2}/mat_lam_lb{lb:.3f}.png",dpi=300)
    plt.close()
    eival, eivec = la.eigh(cmat)
    eiglist.append(eival[::-1])
fig, ax = plt.subplots(figsize=[10,8])
ax.plot(lblist,condlist)
ax.set_yscale("log")
ax.set_title(f"condition number, functype={functype}")
ax.set_xlabel(r"$L_b$")# for $\exp (-\frac{1}{2}(\frac{d}{2\pi\times L_b})^2)$")
#ax.set_xlabel(r"$L_b$ for $\exp (-\frac{1}{2}(\frac{d}{L_b\times N/\pi})^2)$")
fig.savefig(f"{figdir}/matcond{functype}_lam.png",dpi=300)
#plt.show()
plt.close()
fig, ax = plt.subplots(figsize=[10,8])
for i,lb in enumerate(lblist):
    ev = eiglist[i]
    ax.plot(np.arange(1,ev.size+1),ev,lw=0.0,c=cmap(2*i*intcol),marker='x',label=f'lb={lb:.3f}')
    if lb>0 and functype=="gauss":
        knum = np.zeros(ev.size)
        for j in range(1,int(ev.size/2)):
            knum[2*j-1:2*j+1] = float(j)
        knum[-1] = ev.size / 2.0
        d = np.sqrt(2.0) / lb / distmax
        ev_anl = ev.size*np.exp(-knum*knum/d/d)/np.sum(np.exp(-knum*knum/d/d))
        #ax.plot(np.arange(1,ev.size+1),ev_anl,c=cmap(2*i*intcol+1))
        print(f"cond(calc)={condlist[i]} cond(anl)={np.sqrt(ev_anl[0]/ev_anl[-1])}")
#ax.legend(ncol=3)
fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(lblist[0],lblist[-1]), cmap=cmap),ax=ax,label="lb")
ax.set_yscale("log")
ax.set_xlabel("mode")
ax.set_title(f"eigenvalues, functype={functype}")
fig.savefig(f"{figdir}/mateig{functype}_lam.png",dpi=300)
#plt.show()
