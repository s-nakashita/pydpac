import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.interpolate import interp1d
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FixedFormatter
plt.rcParams['font.size'] = 16
from nmc_tools import psd, wnum2wlen, wlen2wnum

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
model_error = False
if len(sys.argv)>4:
    model_error = (sys.argv[4]=='True')
perts = ["mlef", "envar", "etkf", "po", "srf", "letkf", "kf", "var",\
    "mlefcw","mlefy","mlefbe","mlefbm",\
    "4detkf", "4dpo", "4dsrf", "4dletkf", "4dvar", "4dmlef"]
if model == "z08":
    perts = ["mlef", "grad", "etkf-fh", "etkf-jh"]#, "po", "srf", "letkf"]
    linestyle = {"mlef":"solid", "grad":"dashed",
     "etkf-fh":"solid", "etkf-jh":"dashed"}
    linecolor = {"mlef":'tab:blue',"grad":'tab:orange',"etkf-fh":'tab:green',"etkf-jh":'tab:red'}
cmap = "coolwarm"
f = "truth.npy"
if not os.path.isfile(f):
    print("not exist {}".format(f))
    exit
xt = np.load(f)[:na,]
print(xt.shape)
nx_t = xt.shape[1]
t = np.arange(na)
xs_t = np.arange(nx_t)
dx_t = 2.0 * np.pi / nx_t
xs_t_rad = xs_t * dx_t
xt2mod = interp1d(xs_t,xt,axis=1)
xlim = 15.0
for pt in perts:
    f = "{}_xa_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xa = np.load(f)
    print(xa.shape)
    f = "{}_xsa_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    xsa = np.load(f)
    print(xsa.shape)
    nx = xa.shape[1]
    intmod = nx_t // nx
    xs = np.arange(0,nx_t,intmod)
    xs_rad = xs * dx_t
    xd = xa - xt2mod(xs)
    fig, axs = plt.subplots(nrows=2,figsize=[10,10],constrained_layout=True)
    axs[0].plot(xs,np.sqrt(np.mean(xd**2,axis=0)),label='err')
    if pt != "kf" and pt != "var" and pt != "4dvar":
        axs[0].plot(xs,xsa.mean(axis=0),label='sprd')
    axs[0].set_xlim(xs[0],xs[-1])
    axs[0].set_xlabel("grid index")
    axs[0].set_ylabel("error or spread")
    axs[0].set_title("state space")
    axs[0].legend()
    axs[0].grid()
    #esp = fft(xd,axis=1)
    #freq = fftfreq(nx,dx)[:nx//2] #* 2.0 * np.pi
    #print(esp.shape)
    #print(freq)
    #psd = 2.0*np.mean(np.abs(esp[:,:nx//2])**2,axis=0)*dx*dx/2.0/np.pi
    wnum, epsd = psd(xd,xs_rad,axis=1)
    axs[1].plot(wnum,epsd,label='err')
    if pt != "kf" and pt != "var" and pt != "4dvar":
        #esps = fft(xsa,axis=1)
        #psd = 2.0*np.mean(np.abs(esps[:,:nx//2])**2,axis=0)*dx*dx/2.0/np.pi
        wnum, epsd = psd(xsa,xs_rad,axis=1)
        axs[1].plot(wnum,epsd,label='sprd')
    #axs[1].set_xlim(wnum[0],wnum[-1])
    axs[1].set(xlabel=r'wave number [radian$^{-1}$]',title='variance power spectra')
    axs[1].set_xscale('log')
    #axs[1].xaxis.set_major_locator(FixedLocator([180./np.pi,120./np.pi,60./np.pi,30./np.pi,1.0/np.pi,0.5/np.pi]))
    #axs[1].xaxis.set_major_formatter(FixedFormatter([r'$\frac{180}{\pi}$',r'$\frac{120}{\pi}$',r'$\frac{60}{\pi}$',r'$\frac{30}{\pi}$',r'$\frac{1}{\pi}$',r'$\frac{1}{2\pi}$']))
    axs[1].xaxis.set_major_locator(FixedLocator([480,240,120,60,8,2,1]))
    axs[1].xaxis.set_major_formatter(FixedFormatter(['480','240','120','60','8','2','1']))
    #axs[1].set_xlim(0.5/np.pi,wnum[-1])
    secax = axs[1].secondary_xaxis('top',functions=(wnum2wlen,wlen2wnum))
    secax.set_xlabel('wave length [radian]')
    secax.xaxis.set_major_locator(FixedLocator([2.0*np.pi,np.pi,np.pi/4.,np.pi/30.,np.pi/60.,np.pi/120.,np.pi/240.]))
    secax.xaxis.set_major_formatter(FixedFormatter([r'$2\pi$',r'$\pi$',r'$\frac{\pi}{4}$',r'$\frac{\pi}{30}$',r'$\frac{\pi}{60}$',r'$\frac{\pi}{120}$',r'$\frac{\pi}{240}$']))
    axs[1].set_yscale("log")
    axs[1].set_ylabel("power spectral density")
    axs[1].set_title("spectral space")
    axs[1].legend()
    axs[1].grid()
#    xd2 = ifft(esp,axis=1)
#    axs[0].plot(xs,xd2[0,],label='reconstructed')
    fig.suptitle(f"{op} {pt}")
    fig.savefig("{}_errspectra_{}_{}.png".format(model,op,pt))
    plt.show()
