import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
from .chi_test import Chi
from .minimize import Minimize

zetak = []
alphak = []
logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')
        
class Lmlef():
# MLEF with Observation space localization
# Reference: Yokota et al. (2016,SOLA)
    def __init__(self, nmem, obs, 
        nvars=1,ndims=1,
        iinf=None, infl_parm=1.0, 
        iloc=0, lsig=-1.0, calc_dist1=None, 
        ltlm=False, incremental=True, model="model"):
        # necessary parameters
        self.pt = "mlef"
        self.nmem = nmem # ensemble size
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        # optional parameters
        # for 2 or more variables
        self.nvars = nvars
        # for 2 or more dimensional data
        self.ndims = ndims
        # inflation
        self.iinf = iinf # iinf = None->No inflation
                         #      = -1  ->Pre-multiplicative inflation
                         #      = 0   ->Post-multiplicative inflation
                         #      = 1   ->Additive inflation
                         #      = 2   ->RTPP(Relaxation To Prior Perturbations)
                         #      = 3   ->RTPS(Relaxation To Prior Spread)
        self.infl_parm = infl_parm # inflation parameter
        # localization
        self.iloc = iloc # iloc = -1  ->CW
                         #      = 0   ->Y
        self.loctype = {-1:'CW',0:'Y'}
        self.pt += self.loctype[self.iloc].lower()
        self.lsig = lsig # localization parameter
        if calc_dist1 is None:
            def calc_dist1(i, j):
                return min(abs(j-i),self.ndim-abs(j-i))
        #else:
        self.calc_dist1 = calc_dist1 # distance calculation routine
        # tangent linear
        self.ltlm = ltlm # True->Use tangent linear approximation False->Not use
        # incremental form
        self.incremental = incremental
        self.model = model
        logger.info(f"R-localization type : {self.pt}")

    def calc_pf(self, xf, **kwargs):
        spf = xf[:, 1:] - xf[:, 0].reshape(-1,1)
        pf = spf @ spf.transpose()
        logger.debug(f"pf max{np.max(pf)} min{np.min(pf)}")
        return pf

    def precondition(self,zmat):
        #u, s, vt = la.svd(zmat)
        #v = vt.transpose()
        #is2r = 1 / (1 + s**2)
        rho = 1.0
        if self.iinf==-1:
            logger.debug("==pre multiplicative inflation==, alpha={}".format(self.infl_parm))
            rho = 1.0 / self.infl_parm
        c = zmat.transpose() @ zmat
        lam, v = la.eigh(c)
        D = np.diag(1.0/(np.sqrt(lam + np.full(lam.size,rho))))
        vt = v.transpose()
        tmat = v @ D @ vt
        heinv = tmat @ tmat.T
        #logger.debug("tmat={}".format(tmat))
        #logger.debug("heinv={}".format(heinv))
        logger.debug("eigen value ={}".format(lam))
        #print(f"rank(zmat)={lam[lam>1.0e-10].shape[0]}")
        return tmat, heinv

    def callback(self, xk, alpha=None):
        global zetak, alphak
        logger.debug("xk={}".format(xk))
        zetak.append(xk)
        if alpha is not None:
            alphak.append(alpha)

    def calc_j(self, v, *args):
        if self.iloc == -1: #CW
            # control variable v=zeta
            if not self.incremental:
                xc, pf, y, yloc, tmat, gmat, heinv, rinv = args
                x = xc + gmat @ v
                ob = y - self.obs.h_operator(yloc, x)
                j = 0.5 * (v.transpose() @ heinv @ v + ob.transpose() @ rinv @ ob)
            else:
                # incremental form
                d, tmat, zmat, heinv = args
                nmem = v.size
                w = tmat @ v
                j = 0.5 * (v.transpose() @ heinv @ v + (zmat@w - d).transpose() @ (zmat@w - d))
        else: #Y
            # control variable v=w
            d_, zmat = args # local normalized innovation = R^{-1/2}(y-Hx), and local Z
            j = 0.5 * np.dot(v,v) / self.infl_parm + 0.5 * np.dot(d_,d_)
        return j
    

    def calc_grad_j(self, v, *args):
        if self.iloc == -1: #CW
            # control variable v=zeta
            if not self.incremental:
                xc, pf, y, yloc, tmat, gmat, heinv, rinv = args
                x = xc + gmat @ v
                hx = self.obs.h_operator(yloc, x)
                ob = y - hx
                if self.ltlm:
                    dh = self.obs.dh_operator(yloc, x) @ pf
                else:
                    dh = self.obs.h_operator(yloc, x[:, None] + pf) - hx[:, None]
                grad = heinv @ v - tmat @ dh.transpose() @ rinv @ ob
            else:
                # incremental form
                d, tmat, zmat, heinv = args
                nmem = v.size
                w = tmat @ v
                grad = heinv @ v + tmat @ zmat.transpose() @ (zmat@w - d)
        else: #Y
            # control variable v=w
            d_, zmat = args
            grad = v / self.infl_parm - zmat.transpose() @ d_
        return grad

    def calc_hess(self, v, *args):
        if self.iloc == -1: #CW
            # control variable v=zeta
            if not self.incremental:
                xc, pf, y, yloc, tmat, gmat, heinv, rinv = args
                x = xc + gmat @ v
                if self.ltlm:
                    dh = self.obs.dh_operator(yloc, x) @ pf
                else:
                    dh = self.obs.h_operator(yloc, x[:, None] + pf) - self.obs.h_operator(yloc, x)[:, None]
                hess = tmat @ (np.eye(v.size) + dh.transpose() @ rinv @ dh) @ tmat
            else:
                # incremental form
                d, tmat, zmat, heinv = args
                hess = tmat @ (np.eye(v.size) + zmat.transpose() @ zmat) @ tmat
        else: #Y
            # control variable v=w
            d_, zmat = args
            hess = np.eye(v.size) / self.infl_parm + zmat.transpose() @ zmat
        return hess

    def cost_j(self, nx, nmem, xopt, icycle, *args):
        #xc, pf, y, yloc, tmat, gmat, heinv, rinv= args
        delta = np.linspace(-nx,nx,4*nx)
        jvalb = np.zeros((len(delta)+1,nmem))
        jvalb[0,:] = xopt
        for k in range(nmem):
            x0 = np.zeros(nmem)
            for i in range(len(delta)):
                x0[k] = delta[i]
                j = self.calc_j(x0, *args)
                jvalb[i+1,k] = j
        np.save("{}_cJ_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), jvalb)

    def dfs(self, zmat):
        u, s, vt = la.svd(zmat)
        ds = np.sum(s**2/(1.0+s**2))
        return ds

    def r_loc(self, sigma, obsloc, xloc):
        if sigma < 0.0:
            loc_scale = 1.0e5
        else:
            loc_scale = sigma
        nobs = obsloc.size
        far = np.arange(nobs)
        Rwf_loc = np.ones(nobs)

        # distance threshold
        dist0 = loc_scale * np.sqrt(10.0/3.0) * 2.0
        logger.debug(dist0)

        dist = np.zeros(nobs)
        for k in range(nobs):
            dist[k] = self.calc_dist1(xloc, obsloc[k])
        far = far[dist>dist0]
        logger.debug(far)
        Rwf_loc = np.exp(-0.5*(dist/loc_scale)**2)
        return far, Rwf_loc

    def __call__(self, xb, pb, y, yloc, r=None, rmat=None, rinv=None,
        method="CG", cgtype=1,
        gtol=1e-6, maxiter=None, restart=False, maxrest=20, update_ensemble=False,
        disp=False, save_hist=False, save_dh=False, save_w=False, icycle=0):
        global zetak, alphak
        zetak = []
        alphak = []
        if (r is None) or (rmat is None) or (rinv is None):
            logger.info("set R")
            r, rmat, rinv = self.obs.set_r(yloc)
        else:
            logger.info("use input R")
        xf = xb[:, 1:]
        xc = xb[:, 0]
        nmem = xf.shape[1]
        chi2_test = Chi(y.size, nmem, rmat)
        pf = xf - xc[:, None]
        #if self.linf:
        #    logger.info("==inflation==, alpha={}".format(self.infl_parm))
        #    pf *= self.infl_parm
        fpf = pf @ pf.T
        if self.iinf == 3:
            stdv_f = np.sqrt(np.diag(fpf))
        if save_dh:
            np.save("{}_pf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), fpf)
            np.save("{}_spf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), pf)
        
        xa = xc.copy()
        pa = np.zeros_like(pf)
        ob = y - self.obs.h_operator(yloc,xc)
        if self.ltlm:
            logger.debug("dhdx={}".format(self.obs.dhdx(xc)))
            dh = self.obs.dh_operator(yloc,xc) @ pf
        else:
            dh = self.obs.h_operator(yloc,xc[:, None]+pf) - self.obs.h_operator(yloc,xc)[:, None]
        if save_dh:
            np.save("{}_dh_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), dh)
            np.save("{}_d_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), ob)
        logger.info("save_dh={}".format(save_dh))
        logger.info("save_hist={}".format(save_hist))
        wlist = []
        Wlist = []
        if self.iloc == -1: #CW
            logger.info("==R-localization, {}== lsig={}".format(self.loctype[self.iloc],self.lsig))
            iprint = np.zeros(2, dtype=np.int32)
            options = {'gtol':gtol, 'disp':disp, 'maxiter':maxiter}
            for i in range(xc.size):
                far, Rwf_loc = self.r_loc(self.lsig, yloc, float(i))
                logger.debug(f"analysis grid={i} Number of assimilated obs.={y.size - len(far)}")
                dhi = np.delete(dh, far, axis=0)
                Rmat = np.diag(np.diag(rmat) * np.sqrt(Rwf_loc))
                Rmat = np.delete(Rmat, far, axis=0)
                Rmat = np.delete(Rmat, far, axis=1)
                zmat = Rmat @ dhi
                logger.debug("cond(zmat)={}".format(la.cond(zmat)))
                tmat, heinv = self.precondition(zmat)
                logger.debug("zmat.shape={}".format(zmat.shape))
                logger.debug("tmat.shape={}".format(tmat.shape))
                logger.debug("heinv.shape={}".format(heinv.shape))
                #gvec = pf[i,:] @ tmat
                gmat = pf @ tmat
                logger.debug("gmat.shape={}".format(gmat.shape))
                if not self.incremental:
                    yi = np.delete(y, far)
                    yiloc = np.delete(yloc, far)
                    Rinv = np.diag(np.diag(rinv) * Rwf_loc)
                    Rinv = np.delete(Rinv, far, axis=0)
                    Rinv = np.delete(Rinv, far, axis=1)
                else:
                    obi = np.delete(ob, far)
                    di = Rmat @ obi
                x0 = np.zeros(pf.shape[1])
                if not self.incremental:
                    args_j = (xc, pf, yi, yiloc, tmat, gmat, heinv, Rinv)
                else:
                    args_j = (di, tmat, zmat, heinv)
                minimize = Minimize(x0.size, self.calc_j, jac=self.calc_grad_j, hess=self.calc_hess,
                            args=args_j, iprint=iprint, method=method, cgtype=cgtype,
                            maxiter=maxiter, restart=restart, loglevel=1)
                if save_hist:
                    x, flg = minimize(x0, callback=self.callback)
                    jh = np.zeros(len(zetak))
                    gh = np.zeros(len(zetak))
                    for j in range(len(zetak)):
                        jh[j] = self.calc_j(np.array(zetak[j]), *args_j)
                        g = self.calc_grad_j(np.array(zetak[j]), *args_j)
                        gh[j] = np.sqrt(g.transpose() @ g)
                    np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), jh)
                    np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), gh)
                    np.savetxt("{}_alpha_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), alphak)
                    if self.model=="z08":
                        xmax = max(np.abs(np.min(x)),np.max(x))
                        logger.debug("resx max={}".format(xmax))
                        if xmax < 1000:
                            self.cost_j(1000, xf.shape[1], x, icycle, *args_j)
                        else:
                            xmax = int(np.ceil(xmax*0.001)*1000)
                            logger.debug("resx max={}".format(xmax))
                            self.cost_j(xmax, xf.shape[1], x, icycle, *args_j)
                    elif self.model=="l96":
                        self.cost_j(200, xf.shape[1], x, icycle, *args_j)
                else:
                    x, flg = minimize(x0)
                xa[i] = xc[i] + gmat[i] @ x
                wlist.append(tmat @ x)
                if self.ltlm:
                    dh = self.obs.dh_operator(yloc,xa) @ pf
                else:
                    dh = self.obs.h_operator(yloc, xa[:, None] + pf) - self.obs.h_operator(yloc, xa)[:, None]
                dhi = np.delete(dh, far, axis=0)
                zmat = Rmat @ dhi
                tmat, heinv = self.precondition(zmat)
                pa[i,:] = pf[i,:] @ tmat 
                Wlist.append(tmat)
        else: #Y
            logger.info("==R-localization, {}== lsig={}".format(self.loctype[self.iloc],self.lsig))
            if maxiter is None:
                maxiter = 5
            iter=0
            w = np.zeros((xa.size,nmem))
            while iter < maxiter:
                gnorm = 0.0
                for i in range(xa.size):
                    far, Rwf_loc = self.r_loc(self.lsig, yloc, float(i))
                    logger.debug(f"Number of assimilated obs.={y.size - len(far)}")
                    dhi = np.delete(dh, far, axis=0)
                    Rmat = np.diag(np.diag(rmat) * np.sqrt(Rwf_loc))
                    Rmat = np.delete(Rmat, far, axis=0)
                    Rmat = np.delete(Rmat, far, axis=1)
                    zmat = Rmat @ dhi
                    d_ = Rmat @ np.delete(ob,far)
                    args_j = (d_,zmat)
                    wk = w[i,]
                    grad = self.calc_grad_j(wk,*args_j)
                    gnormk = np.sqrt(np.dot(grad,grad))
                    logger.info(f"local gradient norm={gnormk:.4e}")
                    gnorm += gnormk
                    if gnormk > gtol:
                        hess = self.calc_hess(wk,*args_j)
                        desc = la.solve(hess, -grad)
                        wk = wk + desc
                        w[i,] = wk
                if gnorm < gtol*xa.size:
                    logger.info(f"Optimization finished successfully at {iter} iterations")
                    break
                xa = xc + np.sum(pf*w,axis=1)
                ## update innovation
                ob = y - self.obs.h_operator(yloc,xa)
                if self.ltlm:
                    logger.debug("dhdx={}".format(self.obs.dhdx(xa)))
                    dh = self.obs.dh_operator(yloc,xa) @ pf
                else:
                    dh = self.obs.h_operator(yloc,xa[:, None]+pf) - self.obs.h_operator(yloc,xa)[:, None]
                iter += 1
            ## update ensemble
            for i in range(xa.size):
                far, Rwf_loc = self.r_loc(self.lsig, yloc, float(i))
                dhi = np.delete(dh, far, axis=0)
                Rmat = np.diag(np.diag(rmat) * np.sqrt(Rwf_loc))
                Rmat = np.delete(Rmat, far, axis=0)
                Rmat = np.delete(Rmat, far, axis=1)
                zmat = Rmat @ dhi
                tmat, heinv = self.precondition(zmat)
                pa[i,:] = pf[i,:] @ tmat
                wlist.append(w[i,])
                Wlist.append(tmat)
        if save_w:
            logger.debug(f"wlist={np.array(wlist).shape}")
            logger.debug(f"Wlist={np.array(Wlist).shape}")
            np.save("wa_{}_{}_cycle{}.npy".format(self.op, self.pt, icycle), np.array(wlist))
            np.save("Wmat_{}_{}_cycle{}.npy".format(self.op, self.pt, icycle), np.array(Wlist))
        # statistical evaluation
        if self.ltlm:
            dh = self.obs.dh_operator(yloc,xa) @ pf
        else:
            dh = self.obs.h_operator(yloc, xa[:, None] + pf) - self.obs.h_operator(yloc, xa)[:, None]
        zmat = rmat @ dh
        logger.debug("cond(zmat)={}".format(la.cond(zmat)))
        tmat, heinv = self.precondition(zmat)
        d = y - self.obs.h_operator(yloc, xa)
        logger.info("zmat shape={}".format(zmat.shape))
        logger.info("d shape={}".format(d.shape))
        innv, chi2 = chi2_test(zmat, d)
        ds = self.dfs(zmat)
        logger.info("dfs={}".format(ds))
        if save_dh:
            np.save("{}_dx_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), xa - xc)
        if self.iinf == 0:
            logger.info("==multiplicative inflation==, alpha={}".format(self.infl_parm))
            pa *= self.infl_parm
        if self.iinf == 1:
            logger.info("==additive inflation==, alpha={}".format(self.infl_parm))
            pa += np.random.randn(pa.shape[0], pa.shape[1])*self.infl_parm
        if self.iinf == 2:
            logger.info("==RTPP, alpha={}".format(self.infl_parm))
            pa = (1.0 - self.infl_parm)*pa + self.infl_parm * pf
        if self.iinf == 3:
            logger.info("==RTPS, alpha={}".format(self.infl_parm))
            fpa = pa @ pa.T
            stdv_a = np.sqrt(np.diag(fpa))
            beta = np.sqrt(((1.0 - self.infl_parm)*stdv_a + self.infl_parm*stdv_f)/stdv_a)
            logger.info(f"beta={beta}")
            pa = pa * beta[:, None]
        u = np.zeros_like(xb)
        u[:, 0] = xa
        u[:, 1:] = xa[:, None] + pa
        fpa = pa @ pa.T
        if save_dh:
            np.save("{}_pa_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), fpa)
            np.save("{}_ua_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), u)
        return u, fpa, pa, innv, chi2, ds
