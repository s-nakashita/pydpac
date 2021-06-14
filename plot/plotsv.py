import sys
import os
import numpy as np
import matplotlib.pyplot as plt

pt = sys.argv[1]
x = np.arange(10)*6
title = "SV inner product norm"
fig, ax = plt.subplots()
i = 0
for hh in [6, 12, 24, 48]:
    es = np.loadtxt("sv{}_{}.txt".format(hh, pt))
    ax.plot(x, es, label="{}h".format(hh))
er = np.loadtxt("random_{}.txt".format(pt))
ax.plot(x, er, label="random")
ax.set(xlabel="hour", ylabel="RMSE",
        title=title)
ax.set_xticks(x)
ax.legend()
fig.savefig("rmse_{}.png".format(pt))
plt.close()

fig, ax = plt.subplots()
i = 0
for hh in [6, 12, 24, 48]:
    es = np.loadtxt("sv{}_{}.txt".format(hh, pt))
    ax.plot(x[:4], es[:4], label="{}h".format(hh))
er = np.loadtxt("random_{}.txt".format(pt))
ax.plot(x[:4], er[:4], label="random")
ax.set(xlabel="hour", ylabel="RMSE",
        title=title)
ax.set_xticks(x[:4])
ax.legend()
fig.savefig("rmse_{}_zoom.png".format(pt))