import sys
import os
import matplotlib.pyplot as plt
import numpy as np

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
perts = ["mlef", "grad", "etkf", "po", "srf", "letkf", "kf"]
if model == "z08":
    perts = ["mlef", "grad", "etkf-fh", "etkf-jh"]#, "po", "srf", "letkf"]

def ibin_sum(bin_increment, bin_range, innv):
    i1max = int(bin_range/bin_increment)
    full_range = 2*i1max + 1
    bin = np.linspace(-bin_range, bin_range, full_range)
    sbin_c = np.exp(-bin*bin/2.0)/np.sqrt(2.0*np.pi)
    bin_left_max = bin[0] - bin_increment / 2.
    bin_right_max = bin[-1] + bin_increment / 2.
    ibin = np.zeros(bin.size)
    for i in range(innv.size):
        for k in range(bin.size):
            bin_left = bin[k] - bin_increment / 2.
            bin_right = bin[k] + bin_increment / 2.
            if innv[i] >= bin_left and innv[i] < bin_right:
                ibin[k] += 1
        if innv[i] < bin_left_max:
            ibin[0] += 1
        if innv[i] > bin_right_max:
            ibin[-1] += 1
    sumbin = np.sum(ibin)
    sbin = ibin / sumbin / bin_increment
    return bin, sbin_c, sbin

def k_stest(sbin, sbin_c, bin_increment, Nx):
    D = 0.0
    for i in range(sbin.size):
        p1 = np.sum(sbin[:i+1])*bin_increment
        p2 = np.sum(sbin_c[:i+1])*bin_increment
        D = max(D, abs(p1-p2))
    print("===============================")
    print("K-S test D-value = {}".format(D))
    print("alpha(10%)       = {}".format(1.22/np.sqrt(Nx)))
    print("alpha( 5%)       = {}".format(1.36/np.sqrt(Nx)))
    print("alpha( 1%)       = {}".format(1.63/np.sqrt(Nx)))
    print("alpha(0.5%)      = {}".format(1.73/np.sqrt(Nx)))
    print("===============================")

for pt in perts:
    print(pt)
    fig, ax = plt.subplots()
    f = "{}_innv_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    innv = np.load(f)
    print(innv.shape)
    bin_increment = 0.1
    bin_range = 5.0
    ibin, sbin_c, sbin = ibin_sum(bin_increment, bin_range, innv.reshape(innv.size))
    print(ibin.shape)
    print(sbin.shape)
    #r = innv[:,0]
    #ax.hist(r, bins=100, density=True)
    #x = np.linspace(-20.0,20.0,200)
    #y = np.exp(-0.5*x*x)/np.sqrt(2.0*np.pi)
    #ax.plot(x,y)
    ax.bar(ibin, sbin, width=bin_increment, label="analysis")
    ax.plot(ibin, sbin_c, color="tab:orange", label="gauss")
    ax.set_title("innovation statistics "+pt+" "+op)
    ax.legend()
    fig.savefig("{}_innv_{}_{}.png".format(model,op,pt))
    k_stest(sbin, sbin_c, bin_increment, innv.size)