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

    def __init__(self, pt, nmem, obs, 
        linf=False, infl_parm=1.0, 
        lsig=-1.0, calc_dist=None, calc_dist1=None, 
        ltlm=False, incremental=True, model="model"):
        # necessary parameters
        self.pt = pt # DA type (MLEF or GRAD)
        self.nmem = nmem # ensemble size
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        # optional parameters
        # inflation
        self.linf = linf # True->Apply inflation False->Not apply
        self.infl_parm = infl_parm # inflation parameter
        # localization
        self.lsig = lsig # localization parameter
        if calc_dist is None:
            def calc_dist(i):
                dist = np.zeros(self.ndim)
                for j in range(self.ndim):
                    dist[j] = min(abs(j-i),self.ndim-abs(j-i))
                return dist
        #else:
        self.calc_dist = calc_dist # distance calculation routine
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
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig} infl_parm={self.infl_parm} lsig={self.lsig}")
        logger.info(f"linf={self.linf} ltlm={self.ltlm} incremental={self.incremental}")

    def calc_pf(self, xf, pa, cycle):
        spf = xf[:, 1:] - xf[:, 0].reshape(-1,1)
        pf = spf @ spf.transpose()
        logger.debug(f"pf max{np.max(pf)} min{np.min(pf)}")
        return pf

    def precondition(self,zmat):
        #u, s, vt = la.svd(zmat)
        #v = vt.transpose()
        #is2r = 1 / (1 + s**2)
        c = zmat.transpose() @ zmat
        lam, v = la.eigh(c)
        D = np.diag(1.0/(np.sqrt(lam + np.ones(lam.size))))
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

    def calc_j(self, zeta, *args):
        xc, pf, y, yloc, tmat, u, heinv, rmat = args
        x = xc + u
        ob = y - self.obs.h_operator(yloc, x)
        ob = rmat @ ob
        j = 0.5 * (zeta.transpose() @ heinv @ zeta + ob.transpose() @ ob)
        return j
    

    def calc_grad_j(self, zeta, *args):
        xc, pf, y, yloc, tmat, u, heinv, rmat = args
        x = xc + u
        hx = self.obs.h_operator(yloc, x)
        ob = y - hx
        if self.ltlm:
            dh = self.obs.dh_operator(yloc,x) @ pf
        else:
            xe = x[:, None] + pf
            dh = self.obs.h_operator(yloc,xe) - self.obs.h_operator(yloc,x)[:, None]
        zmat = rmat @ dh
        grad = heinv @ zeta - tmat @ zmat.transpose() @ rmat @ ob
        return grad

    def calc_hess(self, zeta, *args):
        xc, pf, y, yloc, tmat, u, heinv, rmat = args
        x = xc + u
        if self.ltlm:
            dh = self.obs.dh_operator(yloc,x) @ pf
        else:
            xe = x[:, None] + pf
            dh = self.obs.h_operator(yloc,xe) - self.obs.h_operator(yloc,x)[:, None]
        zmat = rmat @ dh
        hess = tmat @ (np.eye(zeta.size) + zmat.transpose() @ zmat) @ tmat
        return hess

    # quadratic interpolation
    def calc_step(self, alpha_t, zeta, dk, *args):
        xc, pf, y, yloc, tmat, u, heinv, rmat = args
        zeta_t = zeta + alpha_t * dk
        x = xc + u
        hx = self.obs.h_operator(yloc, x)
        ob = rmat @ (y - hx)
        if self.ltlm:
            dh = self.obs.dh_operator(yloc, x) @ pf
        else:
            dh = self.obs.h_operator(yloc, x[:,None] + pf) - hx[:,None]
        zmat = rmat @ dh
        cost_t = (zeta_t - zeta).transpose() @ tmat @ zmat.transpose() @ ob \
            - (zeta_t - zeta).transpose() @ heinv @ zeta
        cost_b = (zeta_t - zeta).transpose() @ heinv @ (zeta_t - zeta) \
            + (zeta_t - zeta).transpose() @ tmat @ zmat.transpose() @ zmat @ tmat @ (zeta_t - zeta)
        return alpha_t * cost_t / cost_b

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

    def dof(self, zmat):
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
        #if self.model == "l96":
        #    dist0 = 6.5
        #else:
        dist0 = loc_scale * np.sqrt(10.0/3.0) * 2.0
        logger.debug(dist0)

        dist = np.zeros(nobs)
        #for k in range(nobs):
        #    dist[k] = min(abs(obsloc[k] - xloc), nx-abs(obsloc[k] - xloc))
        for k in range(nobs):
            dist[k] = self.calc_dist1(xloc, obsloc[k])
        far = far[dist>dist0]
        logger.debug(far)
        Rwf_loc = np.exp(-0.5*(dist/loc_scale)**2)
        return far, Rwf_loc

    def __call__(self, xb, pb, y, yloc, r=None, rmat=None, rinv=None,
        method="CGF", cgtype=1,
        gtol=1e-6, maxiter=5, restart=False, maxrest=5, update_ensemble=False,
        disp=False, save_hist=False, save_dh=False, icycle=0):
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
        logger.info("==R-localization==, lsig={}".format(self.lsig))
        iprint = np.zeros(2, dtype=np.int32)
        options = {'gtol':gtol, 'disp':disp, 'maxiter':maxiter}
        ylist = []
        yloclist = []
        tmatlist = []
        heinvlist = []
        Rmatlist = []
        # PREPARE : calculate each location values
        for i in range(xc.size):
            far, Rwf_loc = self.r_loc(self.lsig, yloc, float(i))
            logger.info(f"Number of assimilated obs.={y.size - len(far)}")
            Rmat = np.diag(np.diag(rmat) * np.sqrt(Rwf_loc))
            Rmat = np.delete(Rmat, far, axis=0)
            Rmat = np.delete(Rmat, far, axis=1)
            yi = np.delete(y, far)
            yiloc = np.delete(yloc, far)
            ylist.append(yi)
            yloclist.append(yiloc)
            dhi = np.delete(dh, far, axis=0)
            zmat = Rmat @ dhi
            logger.debug("cond(zmat)={}".format(la.cond(zmat)))
            tmat, heinv = self.precondition(zmat)
            logger.debug("zmat.shape={}".format(zmat.shape))
            logger.debug("tmat.shape={}".format(tmat.shape))
            logger.debug("heinv.shape={}".format(heinv.shape))
            tmatlist.append(tmat)
            heinvlist.append(heinv)
            #Rinv = np.diag(np.diag(rinv) * Rwf_loc)
            #Rinv = np.delete(Rinv, far, axis=0)
            #Rinv = np.delete(Rinv, far, axis=1)
            Rmatlist.append(Rmat)
        logger.debug(f"tmat={len(tmatlist)}")
        # MAIN : start minimization (steepest descent)
        niter = 0
        zetalist = [np.zeros(pf.shape[1])]*(xc.size)
        logger.debug(f"zeta={len(zetalist)}")
        alphab = np.ones(xc.size)
        while( niter < maxiter ):
            niter += 1
            # convergence test (>0:not convergence,=0:convergence)
            nconv = 0
            gnorm = 0.0
            # global u = Pf @ w
            u = np.zeros_like(xc)
            for i in range(xc.size):
                tmat = tmatlist[i]
                zeta = zetalist[i]
                u[i] = pf[i,:] @ tmat @ zeta
            for i in range(xc.size):
                zeta = zetalist[i]
                yi = ylist[i]
                yiloc = yloclist[i]
                tmat = tmatlist[i]
                heinv = heinvlist[i]
                Rmat = Rmatlist[i]
                args_j = (xc, pf, yi, yiloc, tmat, u, heinv, Rmat)
                grad = self.calc_grad_j(zeta, *args_j)
                logger.debug(f"{i}th gnorm : {np.sqrt(np.dot(grad, grad))}")
                gnorm += np.sqrt(np.dot(grad, grad))
                #if gnorm > gtol:
                nconv += 1
                alpha = self.calc_step(alphab[i], zeta, -grad, *args_j)
                zetalist[i] = zeta - alpha*grad
                alphab[i] = alpha
            logger.debug(f"step length {alphab}")
            gnorm = gnorm / nconv
            if gnorm < gtol:
                logger.info(f"convergence at {niter} iterations")
                break
        if niter >= maxiter:
            logger.info(f"not convergence, finel gnorm={gnorm}")
        # POSTPROCESS : update analysis and ensemble
        for i in range(xc.size):
            xa[i] = xc[i] + pf[i,:] @ tmatlist[i] @ zetalist[i]
        for i in range(xc.size):
            yiloc = yloclist[i]
            Rmat = Rmatlist[i]
            if self.ltlm:
                dhtmp = self.obs.dh_operator(yiloc,xa) @ pf
            else:
                xetmp = xa[:,None] + pf
                dhtmp = self.obs.h_operator(yiloc,xetmp) - self.obs.h_operator(yiloc,xa)[:, None]
            zmat = Rmat @ dhtmp
            logger.debug("cond(zmat)={}".format(la.cond(zmat)))
            tmat, heinv = self.precondition(zmat)
            pa[i,:] = pf[i,:] @ tmat 
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
        ds = self.dof(zmat)
        logger.info("dof={}".format(ds))
        if save_dh:
            np.save("{}_dx_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), xa - xc)
        if self.linf:
            logger.info("==inflation==, alpha={}".format(self.infl_parm))
            pa *= self.infl_parm

        u = np.zeros_like(xb)
        u[:, 0] = xa
        u[:, 1:] = xa[:, None] + pa
        fpa = pa @ pa.T
        if save_dh:
            np.save("{}_pa_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), fpa)
            np.save("{}_ua_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), u)
        return u, fpa, pa, innv, chi2, ds
