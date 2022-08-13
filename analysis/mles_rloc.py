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
        
class Mles_rloc():
# 4-dimensional MLEF with R-localization
# Reference(En4DVar) Liu et al. 2008: "An ensemble-based four-dimensional variational data assimilation scheme. Part I: Technical formulation and preliminary test," Mon. Wea. Rev., 136, 3363-3373.
    def __init__(self, pt, nmem, obs, step, nt, window_l, 
                nvars=1,ndims=1,
                linf=False, infl_parm=1.0, 
                lsig=-1.0, calc_dist=None, calc_dist1=None,
                ltlm=False, model="model"):
        # necessary parameters
        self.pt = pt # DA type (prefix 4d + MLEF)
        self.nmem = nmem # ensemble size
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.step = step # forward model
        self.nt = nt     # assimilation interval
        self.window_l = window_l # assimilation window length
        # optional parameters
        # for 2 or more variables
        self.nvars = nvars
        # for 2 or more dimensional data
        self.ndims = ndims
        # inflation
        self.linf = linf # True->Apply inflation False->Not apply
        self.infl_parm = infl_parm # inflation parameter
        # localization
        self.lsig = lsig # localization parameter
        if calc_dist is None:
            def calc_dist(self, i):
                dist = np.zeros(self.ndim)
                for j in range(self.ndim):
                    dist[j] = min(abs(j-i),self.ndim-abs(j-i))
                return dist
        else:
            self.calc_dist = calc_dist # distance calculation routine
        if calc_dist1 is None:
            def calc_dist1(self, i, j):
                return min(abs(j-i),self.ndim-abs(j-i))
        else:
            self.calc_dist1 = calc_dist1 # distance calculation routine
        # tangent linear
        self.ltlm = ltlm # True->Use tangent linear approximation False->Not use
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig} infl_parm={self.infl_parm} lsig={self.lsig}")
        logger.info(f"linf={self.linf} ltlm={self.ltlm}")
        logger.info(f"nt={self.nt} window_l={self.window_l}")

    def calc_pf(self, xf, pa, cycle):
        spf = xf[:, 1:] - xf[:, 0].reshape(-1,1)
        pf = spf @ spf.transpose()
        logger.debug(f"pf max{np.max(pf)} min{np.min(pf)}")
        return pf

    def precondition(self,zmat):
        #u, s, vt = la.svd(zmat)
        #v = vt.transpose()
        #is2r = 1 / (1 + s**2)
        c = np.zeros((zmat[0].shape[1], zmat[0].shape[1]))
        for l in range(len(zmat)):
            c = c + zmat[l].transpose() @ zmat[l]
        lam, v = la.eigh(c)
        D = np.diag(1.0/(np.sqrt(lam + np.ones(lam.size))))
        vt = v.transpose()
        tmat = v @ D @ vt
        heinv = tmat @ tmat.T
        logger.debug("tmat={}".format(tmat))
        logger.debug("heinv={}".format(heinv))
        logger.info("eigen value ={}".format(lam))
        return tmat, heinv

    def callback(self, xk, alpha=None):
        global zetak, alphak
        logger.debug("xk={}".format(xk))
        zetak.append(xk)
        if alpha is not None:
            alphak.append(alpha)

    def calc_j(self, zeta, *args):
        #xc, pf, y, yloc, tmat, gmat, heinv, rinv = args
        d, tmat, zmat, heinv = args
        nmem = zeta.size
        #x = xc + gmat @ zeta
        w = tmat @ zeta
        #j = 0.5 * (zeta.transpose() @ heinv @ zeta)
        j = 0.5 * (w.transpose() @ w)
        for l in range(len(d)):
            #ob = y[l] - self.obs.h_operator(yloc[l], x)
            #j += 0.5 * (ob.transpose() @ rinv @ ob)
            ob = zmat[l] @ w - d[l]
            j += 0.5 * (ob.transpose() @ ob)
            #for k in range(self.nt):
            #    x = self.step(x)
        return j
    

    def calc_grad_j(self, zeta, *args):
        #xc, pf, y, yloc, tmat, gmat, heinv, rinv = args
        d, tmat, zmat, heinv = args
        nmem = zeta.size
        #x = xc + gmat @ zeta
        w = tmat @ zeta
        #xl = x[:, None] + pf
        g = heinv @ zeta
        for l in range(len(d)): 
            #hx = self.obs.h_operator(yloc[l], x)
            #ob = y[l] - hx
            #if self.ltlm:
            #    dh = self.obs.dh_operator(yloc[l], x) @ (xl - x[:, None])
            #else:
            #    dh = self.obs.h_operator(yloc[l], xl) - hx[:, None]
            #g = g - tmat @ dh.transpose() @ rinv @ ob
            ob = zmat[l] @ w - d[l]
            g = g + tmat @ zmat[l].transpose() @ ob
            #for k in range(self.nt):
            #    x = self.step(x)
            #    xl = self.step(xl) 
        return g

    def calc_hess(self, zeta, *args):
        #xc, pf, y, yloc, tmat, gmat, heinv, rinv = args
        d, tmat, zmat, heinv = args
        #x = xc + gmat @ zeta
        #xl = x[:, None] + pf
        hess = np.eye(zeta.size)
        for l in range(len(d)):
            #if self.ltlm:
            #    dh = self.obs.dh_operator(yloc[l], x) @ (xl - x[:, None])
            #else:
            #    dh = self.obs.h_operator(yloc, xl) - self.obs.h_operator(yloc, x)[:, None]
            #hess = hess + dh.transpose() @ rinv @ dh
            hess = hess + zmat[l].transpose() @ zmat[l]
            #for k in range(self.nt):
            #    x = self.step(x)
            #    xl = self.step(xl)
        hess = tmat @ hess @ tmat
        return hess

    def cost_j(self, nx, nmem, xopt, icycle, *args):
        #xc, pf, y, yloc, tmat, gmat, heinv, rinv= args
        d, tmat, zmat, heinv = args
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
        z = np.sum(np.array(zmat), axis=0)
        u, s, vt = la.svd(z)
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

    def __call__(self, xb, pb, y, yloc, method="LBFGS", cgtype=None,
        gtol=1e-6, maxiter=None,
        disp=False, save_hist=False, save_dh=False, icycle=0):
        global zetak, alphak
        zetak = []
        alphak = []
        logger.debug(f"obsloc={yloc.shape}")
        logger.debug(f"obssize={y.shape}")
        r, rmat, rinv = self.obs.set_r(yloc[0])
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
        
        xl = np.zeros((xf.shape[0], xf.shape[1]+1))
        xl[:, 0] = xc 
        xl[:, 1:] = xf
        xlc = xl[:, 0]
        pl = np.zeros_like(pf)
        pl = pf[:,:]
        dhmat = []  # observation perturbations
        d = []      # innovations
        spfmat = [] # square root of Pf
        for l in range(min(self.window_l, y.shape[0])):
            spfmat.append(pl)
            if self.ltlm:
                logger.debug("dhdx={}".format(self.obs.dh_operator(yloc[l],xlc)))
                dh = self.obs.dh_operator(yloc[l],xlc) @ pl
            else:
                dh = self.obs.h_operator(yloc[l],xl[:,1:]) - self.obs.h_operator(yloc[l],xlc)[:, None]
            dhmat.append(dh)
            ob = y[l] - self.obs.h_operator(yloc[l],xlc)
            d.append(ob)
            if save_dh:
                np.save("{}_dh_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), dh)
                np.save("{}_d_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), d[l]) 
            for k in range(self.nt):
                xl = self.step(xl)
            xlc = xl[:, 0]
            pl = xl[:, 1:] - xlc[:, None]
        logger.info("save_dh={}".format(save_dh))
        logger.info("save_hist={}".format(save_hist))
        logger.info("==R-localization==, lsig={}".format(self.lsig))
        iprint = np.zeros(2, dtype=np.int32)
        options = {'gtol':gtol, 'disp':disp, 'maxiter':maxiter}
        xa = xc.copy()
        pa = np.zeros_like(pf)
        for i in range(xc.size):
            Rloc = []
            zmat = []
            di = []
            for k in range(len(d)):
                far, Rwf_loc = self.r_loc(self.lsig, yloc[k], float(i))
                logger.info(f"Number of assimilated obs.={y[k].size - len(far)}")
                Rmat = np.diag(np.diag(rmat) * np.sqrt(Rwf_loc))
                Rmat = np.delete(Rmat, far, axis=0)
                Rmat = np.delete(Rmat, far, axis=1)
                obi = np.delete(d[k], far)
                dhi = np.delete(dhmat[k], far, axis=0)
                Rloc.append(Rmat)
                zmat.append(Rmat @ dhi)
                di.append(Rmat @ obi)
            logger.debug("cond(zmat)={}".format(la.cond(zmat[0])))
            tmat, heinv = self.precondition(zmat)
            logger.debug("zmat.shape={}".format(np.array(zmat).shape))
            logger.debug("tmat.shape={}".format(tmat.shape))
            logger.debug("heinv.shape={}".format(heinv.shape))
            gvec = pf[i,:] @ tmat
            logger.debug("gvec.shape={}".format(gvec.shape))
            x0 = np.zeros(pf.shape[1])
        #args_j = (xc, pf, y, yloc, tmat, gmat, heinv, rinv)
            args_j = (di, tmat, zmat, heinv)
            minimize = Minimize(x0.size, self.calc_j, jac=self.calc_grad_j, hess=self.calc_hess,
                            args=args_j, iprint=iprint, method=method, cgtype=cgtype,
                            maxiter=maxiter)
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
                        logger.info("resx max={}".format(xmax))
                        self.cost_j(xmax, xf.shape[1], x, icycle, *args_j)
                elif self.model=="l96":
                    self.cost_j(200, xf.shape[1], x, icycle, *args_j)
            else:
                x, flg = minimize(x0)
            xa[i] = xc[i] + gvec @ x
            xlc = xa[:]
            zmat = [] # observation perturbations
            for l in range(min(self.window_l, y.shape[0])):
                if self.ltlm:
                    logger.debug("dhdx={}".format(self.obs.dh_operator(yloc[l],xlc)))
                    dh = self.obs.dh_operator(yloc[l],xlc) @ spfmat[l]
                else:
                    dh = self.obs.h_operator(yloc[l],xlc[:,None]+spfmat[l]) - self.obs.h_operator(yloc[l],xlc)[:, None]
                dhi = np.delete(dh, far, axis=0)
                zmat.append(Rloc[l] @ dhi)
                for k in range(self.nt):
                    xlc = self.step(xlc)
            tmat, heinv = self.precondition(zmat)
            pa[i, :] = pf[i, :] @ tmat
        if save_dh:
            np.save("{}_dx_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), xa - xc)
        # statistical evaluation
        zmat = [] # observation perturbations
        d = [] # normalized innovation vectors
        xlc = xa
        for l in range(min(self.window_l, y.shape[0])):
            if self.ltlm:
                logger.debug("dhdx={}".format(self.obs.dh_operator(yloc[l],xlc)))
                dh = self.obs.dh_operator(yloc[l],xlc) @ spfmat[l]
            else:
                dh = self.obs.h_operator(yloc[l],xlc[:,None]+spfmat[l]) - self.obs.h_operator(yloc[l],xlc)[:, None]
            zmat.append(rmat @ dh)
            ob = y[l] - self.obs.h_operator(yloc[l], xlc)
            d.append(ob)
            for k in range(self.nt):
                xlc = self.step(xlc)
        logger.debug("cond(zmat)={}".format(la.cond(zmat[0])))
        tmat, heinv = self.precondition(zmat)
        logger.info("zmat shape={}".format(np.array(zmat).shape))
        logger.info("d shape={}".format(np.array(d).shape))
        #innv, chi2 = chi2_test(zmat, d)
        ds = self.dof(zmat)
        logger.info("dof={}".format(ds))
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
        #return u, fpa, pa, innv, chi2, ds
        return u, fpa, ds
