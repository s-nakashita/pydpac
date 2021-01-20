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
for pt in perts:
    fig, ax = plt.subplots()
    f = "{}_innv_{}_{}.npy".format(model, op, pt)
    if not os.path.isfile(f):
        print("not exist {}".format(f))
        continue
    innv = np.load(f)
    print(innv.shape)
    r = innv.reshape(innv.size)
    #r = innv[:,0]
    ax.hist(r, bins=100, density=True)
    x = np.linspace(-20.0,20.0,200)
    y = np.exp(-0.5*x*x)/np.sqrt(2.0*np.pi)
    ax.plot(x,y)
    ax.set_title("innovation statistics "+pt+" "+op)
    fig.savefig("{}_innv_{}_{}.png".format(model,op,pt))