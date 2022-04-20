from math import tau
import numpy as np
from numpy.fft import rfft, irfft
import sys

def derivative(g,klist,period=tau,trunc=None):
    dg = []
    count = 0
    m  = np.zeros(g.size//2+1, dtype=np.complex128)
    mm = np.zeros_like(m)
    m.imag = np.arange(g.size//2+1)
    w = rfft(g)
    if trunc != None and trunc < w.size:
        w[trunc+1:] = 0.0
    for j, k in zip(range(len(klist)),klist):
        dg.append(np.zeros_like(g))
        if k==0:
            dg[j] = irfft(w) if trunc != None else g
        else:
            i = max(1,k)
            mm[:i] = 0.0
            mm[i:] = m[i:] ** k
            dg[j] = (tau / period) ** k * irfft(mm * w)
    return dg

if __name__ == "__main__":
    n = 16
    x = np.cos(2 * np.pi / n * np.arange(n)) + np.cos(3 * 2 * np.pi / n * np.arange(n))
    print(f"x={x}")
    klist = [-1, 0, 1, 2, 3]
    dxlist = derivative(x,klist)
    for k, dx in zip(klist, dxlist):
        print(f"{k}th derivative={dx}")
    dxlist = derivative(x, [1], trunc=2)
    print(f"truncated first derivative={dxlist[0]}")