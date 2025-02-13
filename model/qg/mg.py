import numpy as np
import matplotlib.pyplot as plt
try:
    from .fd import laplacian, l2norm
except ImportError:
    from fd import laplacian, l2norm
import sys
import time

def jacobi_step(pin, q, d, f, niter=100, tol=1e-5):
    dd = d * d
    p = pin.copy()
    r = np.zeros_like(pin)
    res = 0
    k = 0
    for i in range(niter):
        k += 1
        r = (laplacian(p) / dd - f * p - q)[1:-1, 1:-1]
        p[1:-1, 1:-1] += dd * r / (4 + dd * f)
        res = l2norm(r, d)
        if res < tol: break
    return p, res, k

def restrict(pin):
    p = pin[::2, ::2].copy()
    return p

def prolong(pin, debug=False):
    imax, jmax = pin.shape
    imax, jmax = 2 * imax - 1, 2 * jmax - 1
    p = np.zeros([imax, jmax])
# copy values at matching elements
    p[::2, ::2] = pin[:, :]
    if debug: plt.matshow(p); plt.show()
# one-dimensional interpolation
    p[::2, 1:-1:2] = 0.5 * (p[::2, 2::2] + p[::2, :-2:2])
    if debug: plt.matshow(p); plt.show()
    p[1:-1:2, ::2] = 0.5 * (p[2::2, ::2] + p[:-2:2, ::2])
    if debug: plt.matshow(p); plt.show()
    p[1:-1:2, 1:-1:2] = 0.5 * (p[1:-1:2, 2::2] + p[1:-1:2, :-2:2])
    if debug: plt.matshow(p); plt.show()
    return p

def v_cycle(p0, q, d, f, itermax=(10, 10, 10), tol=1.0e-5, nlev=None, debug=False, calc_time=False):
    if nlev == None: nlev = int(np.log2(p0.shape[0] - 1))
    if calc_time: start = time.perf_counter()
    p, res, niter = jacobi_step(p0, q, d, f, itermax[0], tol)
#    if debug: print(f"{p.shape} {p.min()} {p.max()}")
#    if debug: plt.matshow(p); plt.show()
    if debug: print(f"pre: res={res:5.2e}, niter={niter}")
    if calc_time:
        end = time.perf_counter()
        print(f"pre: {(end - start)*1e3:.3f}ms")
    if calc_time: start = time.perf_counter()
    qlist = [q]
    h = d
    for i in range(1, nlev):
        p = restrict(p)
        qlist.append(restrict(qlist[i-1]))
        h *= 2
#        if debug: print(f"down {i} {p.shape} {qlist[i].shape} h={h}")
        p, res, niter = jacobi_step(p, qlist[i], h, f, itermax[1], tol)
#        if debug: plt.matshow(p); plt.show()
#        if debug: print(f"{p.shape} {p.min()} {p.max()}")
        if debug: print(f"restrict {i}: res={res:5.2e}, niter={niter}")
    if calc_time:
        end = time.perf_counter()
        print(f"restrict: {(end - start)*1e3:.3f}ms")
    if calc_time: start = time.perf_counter()
    for i in range(nlev-2, -1, -1):
        p = prolong(p)
        h /= 2
#        if debug: print(f"up {i} {p.shape} {qlist[i].shape} h={h}")
        p, res, niter = jacobi_step(p, qlist[i], h, f, itermax[2], tol)
#        if debug: plt.matshow(p); plt.show()
#        if debug: print(f"{p.shape} {p.min()} {p.max()}")
        if debug: print(f"prolong {i}: res={res:5.2e}, niter={niter}")
    if calc_time:
        end = time.perf_counter()
        print(f"prolong: {(end - start)*1e3:.3f}ms")
    return p, res

def s_cycle(q, d, f, itermax=100, tol=1.0e-5, debug=False):
    nlev = int(np.log2(q.shape[0] - 1))
    p = np.zeros([3, 3])
    if debug: plt.matshow(p); plt.show()
    h = d
    qlist = [q]
    for i in range(1, nlev):
        qlist.append(restrict(qlist[i-1]))
        h *= 2
    for i in range(nlev-2, -1, -1):
        p = prolong(p)
        h /= 2
        p, res, niter = jacobi_step(p, qlist[i], h, f, itermax, tol)
        if debug: plt.matshow(p); plt.show()
        if debug: print(f"{p.shape} {p.min()} {p.max()}")
        if debug: print(f"prolong {i}: res={res:5.2e}, niter={niter}")
    return p, res

def prolong_test():
    n = 5
    p = np.zeros([n, n])
    x = np.linspace(-1, 1, n)
    y = x
    for i in range(1, n-1):
        for j in range(1, n-1):
            p[i, j] = np.exp(-(x[i]**2 + y[j]**2))
    p1 = prolong(p, debug=True)

def mg_test(plot=True, itermax=(1, 1, 20000), tol=1e-5, ncycle=1,
        cycle="v", nlev=None, debug=False):
    n = 129 
    d = 1.0 / (n - 1)
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y, indexing="ij")

    ptrue = (X**2 - X**4) * (Y**4 - Y**2)
    print(f"ptrue: min={ptrue.min()} max={ptrue.max()}")
    q = -2 * ((1 - 6 * X*X) * Y*Y * (1 - Y*Y) + \
              (1 - 6 * Y*Y) * X*X * (1 - X*X))
    print(f"q: min={q.min()} max={q.max()}")

    p0 = np.zeros_like(q)
    for i in range(ncycle):
        if cycle == "v":
            p, res = v_cycle(p0, q, d, 0.0, itermax, tol, nlev, debug, calc_time=debug)
        else:
            p, res = s_cycle(q, d, 0.0, itermax[2], tol, debug)
        err = l2norm(p - ptrue, d)
        p0 = p.copy()
        print(f"cycle={i} res={res:5.2e} err={err}")

    if plot:
        plt.rcParams["font.size"] = 12
        fig, axs = plt.subplots(1, 3, figsize=[14, 4])
        z = [ptrue, p, ptrue-p]
        title = [r"$(x^2-x^4)(y^4-y^2)$",
                 f"res={res:.2e} l2={err:.2e}",
                 r"mgrid$-$true"]
        for j in range(len(z)):
            ax = axs[j]
            if j < 2:
                c = ax.contourf(x, y, z[j], levels=np.linspace(-0.07, 0.0, 8))
            else:
                c = ax.contourf(x, y, z[j], cmap="coolwarm", vmin=-5e-6, vmax=5e-6)
            ax.set_title(title[j])
            ax.set_aspect("equal")
            fig.colorbar(c, ax=ax)
        fig.suptitle(f"Multigrid Jacobi itermax={itermax}")
        fig.tight_layout()
        fig.savefig("multigrid_jacobi.png", bbox_inches="tight", dpi=300)
        plt.show()

if __name__ == "__main__":
#    prolong_test()
    mg_test(plot=False, itermax=(1,1,10000), tol=1.0e-5, cycle="v", nlev=None, debug=True)
