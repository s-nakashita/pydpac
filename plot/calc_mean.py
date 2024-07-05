import sys
import os
import numpy as np

op = sys.argv[1]
model = sys.argv[2]
na = int(sys.argv[3])
nmax = int(sys.argv[4])
vname = sys.argv[5]
pt = sys.argv[6]
print(f"{op} {model} {na} {nmax} {vname} {pt}")
if vname == "innv" or vname == "ua":
    if model == "z08":
        emean = np.zeros((na, 81))
    if model == "l96":
        emean = np.zeros((na, 40))
    i = 0
    for count in range(1,nmax+1):
        f = "{}_{}_{}_{}.npy".format(vname, op, pt, count)
        if not os.path.isfile(f):
            print(f"not exist {f}")
            continue
        e = np.load(f)
        if vname == "ua":
            emean += e
        else:
            if i == 0:
                emean += e
            else:
                emean = np.vstack((emean, e))
        i += 1
    if vname == "ua":
        emean /= nmax
    if i>0:
        np.save("{}_{}_{}_{}.npy".format(model, vname, op, pt), emean)
#if vname == "e" or vname == "chi" or vname == "dfs":
elif vname == "sv":
    if model == "z08":
        imean = np.zeros(81)
        fmean = np.zeros(81)
    if model == "l96":
        imean = np.zeros(40)
        fmean = np.zeros(40)
    i = 0
    for count in range(1,nmax+1):
        f = "isv_{}_{}h_{}.npy".format(pt, op, count)
        if not os.path.isfile(f):
            print(f"not exist {f}")
            continue
        e = np.load(f)
        imean += e
        f = "fsv_{}_{}h_{}.npy".format(pt, op, count)
        if not os.path.isfile(f):
            print(f"not exist {f}")
            continue
        e = np.load(f)
        fmean += e
        i += 1
    if i>0:
        imean /= i
        fmean /= i
        np.save("initialSVA_{}_{}h.npy".format(pt, op), imean)
        np.save("finalSVA_{}_{}h.npy".format(pt, op), fmean)
elif vname[:6] == "xdmean" or vname[:6] == "xsmean":
    i = 0
    for count in range(1,nmax+1):
        f = "{}_{}_{}_{}.txt".format(vname, op, pt, count)
        if not os.path.isfile(f):
            print(f"not exist {f}")
            continue
        e = np.loadtxt(f)
        if i==0:
            emean = e.copy()
            estd = e**2
        else:
            emean += e
            estd += e**2
        i += 1
    emean /= i
    estd /= i
    estd = np.sqrt(estd - emean**2)
    data = np.hstack((np.array([i,i]).reshape(-1,1),np.vstack((emean,estd))))
    if i>0:
        np.savetxt("{}_{}_{}_{}.txt".format(model, vname, op, pt), data)
else:
    if model == "z08" and vname == "e":
        emean = np.zeros(na+1)
    else:
        emean = np.zeros(na)
    estd = np.zeros_like(emean)
    i = 0
    for count in range(1,nmax+1):
        f = "{}_{}_{}_{}.txt".format(vname, op, pt, count)
        if not os.path.isfile(f):
            print(f"not exist {f}")
            continue
        e = np.loadtxt(f)
        emean += e
        estd += e**2
        i += 1
    emean /= i
    estd /= i
    estd = np.sqrt(estd - emean**2)
    data = np.hstack((np.array([i,i]).reshape(-1,1),np.vstack((emean,estd))))
    if i>0:
        np.savetxt("{}_{}_{}_{}.txt".format(model, vname, op, pt), data)
#np.savetxt("{}_{}_{}_{}_mean.txt".format(model, vname, op, pt), emean)
