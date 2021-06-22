import sys
import os
import numpy as np
import matplotlib.pyplot as plt

pt = sys.argv[1]
ntype = sys.argv[2]
#x = np.arange(10)*6
x = np.arange(55)
if ntype == "inner":
    title = "SV inner product norm"
elif ntype == "aec":
    title = "SV analysis error norm"
fig, ax = plt.subplots()
figt, axt = plt.subplots()
i = 0
for hh in [6, 12, 24, 48]:
    es = np.loadtxt("sv{}_{}.txt".format(hh, pt))
    ax.plot(x, es, label="{}h".format(hh))
    est = np.loadtxt("sv{}_{}_tlm.txt".format(hh, pt))
    axt.plot(x, est, label="{}h".format(hh))
er = np.loadtxt("random_{}.txt".format(pt))
ax.plot(x, er, label="random")
ert = np.loadtxt("random_{}_tlm.txt".format(pt))
axt.plot(x, ert, label="random")
ax.set(xlabel="hour", ylabel="RMSE",
        title=title)
ax.set_xticks(x[::6])
ax.legend()
fig.savefig("rmse_{}.png".format(pt))
axt.set(xlabel="hour", ylabel="RMSE",
        title=title+" TLM")
axt.set_xticks(x[::6])
axt.legend()
figt.savefig("rmse_{}_tlm.png".format(pt))
plt.close()

#fig, ax = plt.subplots()
#i = 0
#for hh in [6, 12, 24, 48]:
#    es = np.loadtxt("sv{}_{}.txt".format(hh, pt))
#    ax.plot(x[:25], es[:25], label="{}h".format(hh))
#er = np.loadtxt("random_{}.txt".format(pt))
#ax.plot(x[:25], er[:25], label="random")
#ax.set(xlabel="hour", ylabel="RMSE",
#        title=title)
#ax.set_xticks(x[:25:6])
#ax.legend()
#fig.savefig("rmse_{}_zoom.png".format(pt))