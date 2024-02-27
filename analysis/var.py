import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
from .obs import Obs
from .minimize import Minimize
from .corrfunc import Corrfunc

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')
zetak = []
alphak = []

class Var():
    def __init__(self, obs, nx, pt="var", \
        ix=None, ioffset=0, sigb=1.0, lb=-1.0, \
        functype="gauss", a=0.5, bmat=None, bsqrt=None, \
        calc_dist1=None, cyclic=True, model="model"):
        self.pt = pt # DA type 
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.nx = nx # state size
        # climatological background error covariance
        self.sigb = sigb # error variance
        self.lb = lb # error correlation length (< 0.0 : diagonal)
        self.functype = functype # correlation function type (gauss or gc5 or tri)
        self.a = a # gc5 shape parameter
        self.corrfunc = Corrfunc(self.lb,a=self.a)
        self.bmat = bmat # prescribed background error covariance
        self.bsqrt = bsqrt # prescribed preconditioning matrix
        self.cyclic = cyclic # boundary treatment
        if calc_dist1 is None:
            self.calc_dist1 = self._calc_dist1
        else:
            self.calc_dist1 = calc_dist1 # distance function
        if ix is None:
            self.ix = np.arange(self.nx)
        else:
            self.ix = ix # grid
        self.ioffset = ioffset
        self.model = model
        self.verbose = True
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig}")
        logger.info(f"sigb={self.sigb} lb={self.lb} functype={self.functype}")
        logger.info(f"bmat in={self.bmat is not None}")

    def _calc_dist1(self, i, j):
        if self.cyclic:
            dist = np.abs(self.nx/np.pi*np.sin(np.pi*(i-j)/self.nx))
        else:
            dist = np.abs(self.nx/np.pi/2*np.sin(np.pi*(i-j)/self.nx/2))
        return dist

    def calc_pf(self, xf, **kwargs):
        cycle = kwargs['cycle']
        if cycle == 0:
            dist=None
            if self.bmat is None:
                if self.lb < 0:
                    self.bmat = self.sigb**2*np.eye(self.nx)
                else:
                    dist = np.eye(self.nx)
                    self.bmat = np.eye(self.nx)
                    for i in range(self.nx):
                        for j in range(self.nx):
                            dist[i,j] = self.calc_dist1(i+self.ioffset,self.ix[j])
                        if self.functype == "gc5":
                            if self.cyclic:
                                ctmp = self.corrfunc(np.roll(dist[i,],-i)[:self.nx//2+1],ftype=self.functype)
                                ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:-1])])
                                self.bmat[i,] = np.roll(ctmp2,i)
                            else:
                                if i < self.nx-i:
                                    ctmp = self.corrfunc(np.roll(dist[i,],-i)[:self.nx-i],ftype=self.functype)
                                    ctmp2 = np.hstack([ctmp,np.flip(ctmp[1:-1])])
                                    self.bmat[i,] = np.roll(ctmp2,i)[:self.nx]
                                else:
                                    ctmp = self.corrfunc(np.flip(dist[i,:i+1]),ftype=self.functype)
                                    ctmp2 = np.hstack([np.flip(ctmp),ctmp[1:-1]])
                                    self.bmat[i,] = ctmp2[:self.nx]
                        else:
                            self.bmat[i,] = self.corrfunc(dist[i,],ftype=self.functype)
                    #if self.functype=="gauss":
                    #    self.bmat[i,] = np.exp(-0.5*(dist/self.lb)**2)
                    #elif self.functype=="gc5":
                    #    z = dist / self.lb / np.sqrt(10.0/3.0)
                    #    self.bmat = np.where(z<1.0, 1.0 - 5.0*(z**2)/3.0 + 0.625*(z**3) + 0.5*(z**4) - 0.25*(z**5), np.where(z<2.0, 4.0 - 5.0*z + 5.0*(z**2)/3.0 + 0.625*(z**3) - 0.5*(z**4) + (z**5)/12.0 - 2.0/z/3.0, 0.0))
                    #elif self.functype=="tri":
                    #    nj = np.sqrt(3.0/10.0) / self.lb * 2.0 * np.pi
                    #    logger.debug(f"lb={self.lb:.3f} nj={nj}")
                    #    self.bmat = np.where(dist==0.0,1.0,np.sin(nj*dist/2.0)/np.tan(dist/2.0)/nj)
                    self.bmat = np.diag(np.full(self.nx,self.sigb)) @ self.bmat @ np.diag(np.full(self.nx,self.sigb))
            else:
                # use only the correlation structure
                diag = np.diag(self.bmat)
                dsqrtinv = np.diag(1.0/np.sqrt(diag))
                cmat = dsqrtinv @ self.bmat @ dsqrtinv
                self.bmat = np.diag(np.full(cmat.shape[0],self.sigb)) @ cmat @ np.diag(np.full(cmat.shape[0],self.sigb))
            if self.verbose:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10,4),ncols=3,constrained_layout=True)
                xaxis = np.arange(self.nx+1)
                mappable = ax[0].pcolor(xaxis, xaxis, self.bmat, cmap='Blues')
                fig.colorbar(mappable, ax=ax[0],shrink=0.4,pad=0.01)
                ax[0].set_title(r"$\mathrm{cond}(\mathbf{B})=$"+f"{la.cond(self.bmat):.3e}")
                ax[0].invert_yaxis()
                ax[0].set_aspect("equal")
                binv = la.inv(self.bmat)
                mappable = ax[1].pcolor(xaxis, xaxis, binv, cmap='Blues')
                fig.colorbar(mappable, ax=ax[1],shrink=0.4,pad=0.01)
                ax[1].set_title(r"$\mathbf{B}^{-1}$")
                ax[1].invert_yaxis()
                ax[1].set_aspect("equal")
                if dist is None:
                    ax[2].remove()
                else:
                    mappable = ax[2].pcolor(xaxis, xaxis, dist, cmap='viridis')
                    fig.colorbar(mappable, ax=ax[2],shrink=0.4,pad=0.01)
                    ax[2].set_title(r"$d$")
                    ax[2].invert_yaxis()
                    ax[2].set_aspect("equal")
                fig.savefig("Bv{:.1f}l{:.3f}_{}.png".format(self.sigb,self.lb,self.model))
                plt.close()
        return self.bmat

    def callback(self, xk, alpha=None):
        global zetak, alphak
        logger.debug("xk={}".format(xk))
        zetak.append(xk)
        if alpha is not None:
            alphak.append(alpha)

    def prec(self,w,first=False):
        global bsqrt
        if first:
            if self.bsqrt is not None:
                bsqrt = self.bsqrt
            else:
                eival, eivec = la.eigh(self.bmat)
                eival[eival<1.0e-16] = 0.0
                npos = np.sum(eival>=1.0e-16)
                logger.info(f"#positive eigenvalues in bmat={npos}")
                bsqrt = np.dot(eivec,np.diag(np.sqrt(eival)))
        return np.dot(bsqrt,w), bsqrt

    def calc_j(self, w, *args, return_each=False):
        JH, rinv, ob = args
        jb = 0.5 * np.dot(w,w)
        x, _ = self.prec(w)
        d = JH @ x - ob
        jo = 0.5 * d.T @ rinv @ d
        if return_each:
            return jb, jo
        else:
            return jb + jo

    def calc_grad_j(self, w, *args):
        JH, rinv, ob = args
        x, bsqrt = self.prec(w)
        d = JH @ x - ob
        return w + bsqrt.T @ JH.T @ rinv @ d

    def calc_hess(self, w, *args):
        JH, rinv, ob = args
        _, bsqrt = self.prec(w)
        return np.eye(w.size) + bsqrt.T @ JH.T @ rinv @ JH @ bsqrt

    def __call__(self, xf, pf, y, yloc, method="CG", cgtype=1,
        gtol=1e-6, maxiter=None,\
        disp=False, save_hist=False, save_dh=False, icycle=0,
        evalout=False):
        global zetak, alphak, bsqrt
        zetak = []
        alphak = []
        _, rsqrtinv, rinv = self.obs.set_r(yloc)
        JH = self.obs.dh_operator(yloc, xf)
        ob = y - self.obs.h_operator(yloc,xf)
        nobs = ob.size
        if save_dh:
            np.save("{}_d_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), ob)

        w0 = np.zeros_like(xf)
        x0, bsqrt = self.prec(w0,first=True)
        args_j = (JH, rinv, ob)
        iprint = np.zeros(2, dtype=np.int32)
        options = {'iprint':iprint, 'method':method, 'cgtype':cgtype,\
                'gtol':gtol, 'disp':disp, 'maxiter':maxiter}
        minimize = Minimize(w0.size, self.calc_j, jac=self.calc_grad_j, hess=self.calc_hess,
                            args=args_j, **options)
        logger.info(f"save_hist={save_hist} cycle={icycle}")
        if save_hist:
            w, flg = minimize(w0, callback=self.callback)
            jh = np.zeros((len(zetak),2))
            gh = np.zeros(len(zetak))
            for i in range(len(zetak)):
                #jh[i] = self.calc_j(np.array(zetak[i]), *args_j)
                # calculate jb and jo separately
                jb, jo = self.calc_j(np.array(zetak[i]), *args_j, return_each=True)
                jh[i,0] = jb
                jh[i,1] = jo
                g = self.calc_grad_j(np.array(zetak[i]), *args_j)
                gh[i] = np.sqrt(g.transpose() @ g)
            np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), jh)
            np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), gh)
            if len(alphak)>0: np.savetxt("{}_alpha_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), alphak)
        else:
            w, flg = minimize(w0)
        
        x, _ = self.prec(w)
        xa = xf + x
        innv = np.zeros_like(ob)
        fun = self.calc_j(w, *args_j)
        chi2 = fun / nobs

        pai = self.calc_hess(w, *args_j)
        lam, v = la.eigh(pai)
        dfs = xf.size - np.sum(1.0/lam)
        spa = bsqrt @ v @ np.diag(1.0/np.sqrt(lam)) @ v.transpose()
        pa = np.dot(spa,spa.T)
        #spf = la.cholesky(pf)

        if evalout:
            tmp = np.dot(np.dot(rsqrtinv,JH),spa)
            infl_mat = np.dot(tmp,tmp.T)
            eval, _ = la.eigh(infl_mat)
            logger.debug("eval={}".format(eval))
            return xa, pa, spa, innv, chi2, dfs, eval[::-1]
        else:
            return xa, pa, spa, innv, chi2, dfs
