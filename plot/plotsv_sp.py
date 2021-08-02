import sys 
import os
import numpy as np
import matplotlib.pyplot as plt

pt = sys.argv[1]
v_time = int(sys.argv[2])
count = int(sys.argv[3])

#v0 = np.load("initialSVA_{}_{}h.npy".format(pt,6*v_time))
for i in range(count+1):
    f = "isv_{}_{}h_{}.npy".format(pt,6*v_time,i)
    if os.path.isfile(f):
        break
v0 = np.load(f)
nx = v0.size
freq = np.arange(nx//2)
f0 = np.fft.fft(v0)
amp = np.abs(f0)
icount = 1
istart = i + 1
for i in range(istart, count+1):
    f = "isv_{}_{}h_{}.npy".format(pt,6*v_time,i)
    if not os.path.isfile(f):
        print(f"not exist {f}")
        continue
    v0 = np.load(f)
    f0 = np.fft.fft(v0)
    amp += np.abs(f0)
    icount += 1
fig, ax = plt.subplots()
ax.plot(freq, amp[:nx//2], linestyle="dashed", label="initial")

#v = np.load("finalSVA_{}_{}h.npy".format(pt,6*v_time))
for i in range(count+1):
    f = "fsv_{}_{}h_{}.npy".format(pt,6*v_time,i)
    if os.path.isfile(f):
        break
v = np.load(f)
f = np.fft.fft(v)
amp = np.abs(f)
icount = 1
istart = i + 1
for i in range(istart, count+1):
    f = "fsv_{}_{}h_{}.npy".format(pt,6*v_time,i)
    if not os.path.isfile(f):
        print(f"not exist {f}")
        continue
    v = np.load(f)
    fv = np.fft.fft(v)
    amp += np.abs(fv)
    icount += 1
ax.plot(freq, amp[:nx//2], label="final")

ax.legend()
ax.set_xlabel("wave number")
ax.set_ylabel("amplitude")
ax.set_xticks(freq)
ax.set_title("{}h".format(6*v_time))
fig.savefig("spectra_{}_{}h.png".format(pt, 6*v_time))