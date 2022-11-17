import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import re
plt.rcParams['font.size'] = 18
plt.rcParams['legend.fontsize'] = 18
### default
op = 'linear'
model = 'l96'
na = 100
sigma = {"linear": 1.0, "quadratic": 8.0e-1}
### input
if len(sys.argv)>1: op = sys.argv[1]
if len(sys.argv)>2: model = sys.argv[2]
if len(sys.argv)>3: na = int(sys.argv[3])

## without localization (ensemble=40)
methods_nloc = ["var","etkf","mlef","mlef(incr)","4dvar","4detkf","4dmlef","4dmlef(incr)"]
## with localization (ensemble=20)
methods_loc = ["letkf","mlefbe","mlefcw","mlefy",\
    "mlefbe(incr)","mlefcw(incr)","mlefy(incr)",\
    "4dletkf","4dmlefbe","4dmlefcw","4dmlefy",\
    "4dmlefbe(incr)","4dmlefcw(incr)","4dmlefy(incr)"]

## timer
coltype = {'names':('op','pt','iter','time'),'formats':('U16','U16','i2','f')}

fig, ax = plt.subplots(figsize=(12,8))
fig2, ax2 = plt.subplots(figsize=(12,8))
width = 0.25
emean_nloc = []
time_nloc = []
colors = []
methods_nloc3d = []
methods_nloc4d = []
for method in methods_nloc:
    m = re.match(r'([0-9]*d*)([a-z]+)(\(incr\))', method)
    print(method)
    if m is not None:
        print(m)
        pt = m.group(1)+m.group(2)+"_incr"
    else:
        pt = method
    m = re.match(r'[0-9][a-z]+',method)
    if m is not None:
        methods_nloc4d.append(method)
        colors.append('tab:orange')
    else:
        methods_nloc3d.append(method)
        colors.append('tab:blue')
    filename = f"{model}_e_{op}_{pt}.txt"
    timername = "timer"
    print(filename)
    e = np.loadtxt(filename)
    print(e.shape)
    print(e[:,int(na/3):].mean(axis=1))
    emean_nloc.append(e[:,int(na/3):].mean(axis=1))
    _,mes,iters,timer = np.loadtxt(timername,dtype=coltype,unpack=True)
    time = 0.0
    tim_s = 0.0
    it = 0
    for me, tim in zip(mes,timer):
        if me == pt:
            time += tim
            tim_s += tim**2
            it += 1
    time /= it
    tim_s = np.sqrt(tim_s/it - time**2)
    print(f'{time:.4f} sec (std={tim_s:.4f})')
    time_nloc.append(np.array([time,tim_s]))

emean_loc = []
time_loc = []
methods_loc3d = []
methods_loc4d = []
for method in methods_loc:
    m = re.match(r'([0-9]*d*)([a-z]+)(\(incr\))', method)
    print(method)
    if m is not None:
        print(m)
        pt = m.group(1)+m.group(2)+"_incr"
    else:
        pt = method
    m = re.match(r'[0-9][a-z]+',method)
    if m is not None:
        methods_loc4d.append(method)
        colors.append('tab:red')
    else:
        methods_loc3d.append(method)
        colors.append('tab:green')
    filename = f"{model}_e_{op}_{pt}.txt"
    timername = "timer"
    print(filename)
    e = np.loadtxt(filename)
    print(e.shape)
    print(e[:,int(na/3):].mean(axis=1))
    emean_loc.append(e[:,int(na/3):].mean(axis=1))
    _,mes,iters,timer = np.loadtxt(timername,dtype=coltype,unpack=True)
    time = 0.0
    tim_s = 0.0
    it = 0
    for me, tim in zip(mes,timer):
        if me == pt:
            time += tim
            tim_s += tim**2
            it += 1
    time /= it
    tim_s = np.sqrt(tim_s/it - time**2)
    print(f'{time:.4f} sec (std={tim_s:.4f})')
    time_loc.append(np.array([time,tim_s]))
methods_all = methods_nloc + methods_loc
emean = np.array(emean_nloc+emean_loc)
time = np.array(time_nloc+time_loc)
print(emean.shape)
print(time.shape)
ns=0
ne=len(methods_nloc3d)
ax.barh(methods_nloc3d,emean[ns:ne,0],\
    xerr=emean[ns:ne,1],\
    #width=width,
    color=colors[ns:ne],\
    label='w/o loc (40 mem)')
ax2.barh(methods_nloc3d,time[ns:ne,0],\
    xerr=time[ns:ne,1],\
    #width=width,
    color=colors[ns:ne],\
    label='w/o loc (40 mem)')
ns=ne
ne+=len(methods_nloc4d)
ax.barh(methods_nloc4d,emean[ns:ne,0],\
    xerr=emean[ns:ne,1],\
    #width=width,
    color=colors[ns:ne],\
    label='w/o loc (40 mem), 4d')
ax2.barh(methods_nloc4d,time[ns:ne,0],\
    xerr=time[ns:ne,1],\
    #width=width,
    color=colors[ns:ne],\
    label='w/o loc (40 mem), 4d')
ns=ne
ne+=len(methods_loc3d)
ax.barh(methods_loc3d,emean[ns:ne,0],\
    xerr=emean[ns:ne,1],\
    #width=width,
    color=colors[ns:ne],\
    label='loc (20 mem)')
ax2.barh(methods_loc3d,time[ns:ne,0],\
    xerr=time[ns:ne,1],\
    #width=width,
    color=colors[ns:ne],\
    label='loc (20 mem)')
ns=ne
ne+=len(methods_loc4d)
ax.barh(methods_loc4d,emean[ns:ne,0],\
    xerr=emean[ns:ne,1],\
    #width=width,
    color=colors[ns:ne],\
    label='loc (20 mem), 4d')
ax2.barh(methods_loc4d,time[ns:ne,0],\
    xerr=time[ns:ne,1],\
    #width=width,
    color=colors[ns:ne],\
    label='loc (20 mem), 4d')
ax.vlines([sigma[op]],0,1,transform=ax.get_xaxis_transform(),\
    linestyle='dotted',colors='k')
ax.legend(loc='upper right')
ax2.legend(loc='lower right')
#plt.setp(ax.get_xticklabels(),rotation=-45,ha="left")
#plt.setp(ax2.get_xticklabels(),rotation=-45,ha="left")
ax.set_title(model.upper()+' '+op+f' {na} cycle averaged RMSE')
ax2.set_title(model.upper()+' '+op+f' {na} cycle total time (sec)')
fig.tight_layout()
fig2.tight_layout()
fig.savefig("{}_emean_{}.png".format(model, op))
fig2.savefig("{}_timer_{}.png".format(model, op))
plt.show()