import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, ifft
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
plt.rcParams['font.size'] = 16

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
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
nx = xt.shape[1]
t = np.arange(na)
xs = np.arange(nx)
dx = 2.0 * np.pi / nx
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
    xd = xa - xt
    fig, axs = plt.subplots(nrows=2,figsize=[10,10],constrained_layout=True)
    axs[0].plot(xs,xd.mean(axis=0),label='err')
    if pt != "kf" and pt != "var" and pt != "4dvar":
        axs[0].plot(xs,xsa.mean(axis=0),label='sprd')
    axs[0].set_xlim(xs[0],xs[-1])
    axs[0].set_xlabel("grid index")
    axs[0].set_ylabel("error or spread")
    axs[0].set_title("state space")
    axs[0].legend()
    axs[0].grid()
    esp = fft(xd,axis=1)
    freq = fftfreq(nx,dx)[:nx//2] * 2.0 * np.pi
    print(esp.shape)
    print(freq)
    psd = 2.0*np.mean(np.abs(esp[:,:nx//2])**2,axis=0)*dx*dx/2.0/np.pi
    axs[1].plot(freq,psd,label='err')
    if pt != "kf" and pt != "var" and pt != "4dvar":
        esps = fft(xsa,axis=1)
        psd = 2.0*np.mean(np.abs(esps[:,:nx//2])**2,axis=0)*dx*dx/2.0/np.pi
        axs[1].plot(freq,psd,label='sprd')
    axs[1].set_xlim(freq[0],freq[-1])
    axs[1].set_yscale("log")
    axs[1].set_xlabel("wave number")
    axs[1].set_ylabel("power spectral density")
    axs[1].set_title("spectral space")
    axs[1].legend()
    axs[1].grid()
#    xd2 = ifft(esp,axis=1)
#    axs[0].plot(xs,xd2[0,],label='reconstructed')
    fig.suptitle(f"{op} {pt}")
    fig.savefig("{}_errspectra_{}_{}.png".format(model,op,pt))
    plt.show()
