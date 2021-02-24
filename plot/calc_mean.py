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
if vname == "e" or vname == "chi" or vname == "dof":
    if model == "z08" and vname == "e":
        emean = np.zeros(na+1)
    else:
        emean = np.zeros(na)
    i = 0
    for count in range(1,nmax+1):
        f = "{}_{}_{}_{}.txt".format(vname, op, pt, count)
        if not os.path.isfile(f):
            print(f"not exist {f}")
            continue
        e = np.loadtxt(f)
        emean += e
        i += 1
    emean /= nmax
    if i>0:
        np.savetxt("{}_{}_{}_{}.txt".format(model, vname, op, pt), emean)
#np.savetxt("{}_{}_{}_{}_mean.txt".format(model, vname, op, pt), emean)
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