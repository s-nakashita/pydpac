import numpy as np 
import scipy.linalg as la
rng = np.random.default_rng()

def ncm(amat,wgt=None,tol=1.0e-6,maxiter=20,debug=False):
    if wgt is None:
        wgt = np.eye(amat.shape[0])
    dsk = np.zeros_like(amat)
    yk = amat.copy()
    xk = np.eye(amat.shape[0])
    if debug: print(xk.shape,yk.shape,dsk.shape)
    dist = 1.0
    niter = 0
    while(dist>tol):
        xkm1 = xk.copy()
        ykm1 = yk.copy()
        rk=yk-dsk
        xk = proj_s(rk,wgt,debug=debug)
        dsk = xk - rk
        yk = proj_u(xk,wgt,debug=debug)
        if debug: print(xk.shape,yk.shape,dsk.shape)
        dist1 = la.norm(xk-xkm1,ord=np.inf)/la.norm(xk,ord=np.inf)
        dist2 = la.norm(yk-ykm1,ord=np.inf)/la.norm(yk,ord=np.inf)
        dist3 = la.norm(yk-xk,ord=np.inf)/la.norm(yk,ord=np.inf)
        dist = max(dist1,dist2,dist3)
        niter += 1
        print(f"iteration:{niter} dist={dist:.3e}")
        if niter>maxiter: break
    return xk

def proj_s(amat,wgt,debug=False):
    tmp = np.dot(np.dot(wgt,amat),wgt)
    eival, eivec = la.eigh(tmp)
    eival = eival[::-1]
    eivec = eivec[:,::-1]
    eival[eival<1.0e-16] = 0.0
    if debug:
        print(eival)
        print(eivec)
    tmp = np.dot(eivec,np.diag(eival))
    tmpp = np.dot(eivec,tmp.T)
    wgtinv = la.inv(wgt)
    return np.dot(np.dot(wgtinv,tmpp),wgtinv)

def proj_u(amat,wgt,debug=False):
    wgtinv = la.inv(wgt)
    lhmat = wgtinv*wgtinv
    rhvec = np.diag(amat) - 1.0
    theta = la.solve(lhmat,rhvec)
    return amat - np.dot(np.dot(wgtinv,np.diag(theta)),wgtinv)

if __name__ == "__main__":
    amat = np.array([[2,-1,0,0],[-1,2,-1,0],[0,-1,2,-1],[0,0,-1,2]])
    print(amat)
    print(amat.shape)
    xmat = ncm(amat,debug=True)
    print(xmat)
    print(la.norm(amat-xmat,ord='fro'))