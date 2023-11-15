from logging import getLogger
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
import copy
from .chi_test import Chi
from .minimize import Minimize

zetak = []
alphak = []
fileConfig("./logging_config.ini")
logger = getLogger('anl')
        
class Mlef_rloc():

    def __init__(self, pt, nmem, obs, 
        nvars=1,ndims=1,
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
        zetak.append(copy.copy(xk))
        if alpha is not None:
            alphak.append(alpha)

    def calc_j(self, zeta, *args):
        if not self.incremental:
            xc, i, pf, y, yloc, tmat, gmat, heinv, rinv = args
            x = xc + gmat @ zeta
            ob = y - self.obs.h_operator(yloc, x)
            jb = 0.5 * zeta.transpose() @ heinv @ zeta
            jo = 0.5 * ob.transpose() @ rinv @ ob
            #j = 0.5 * (zeta.transpose() @ heinv @ zeta + ob.transpose() @ rinv @ ob)
        else:
            # incremental form
            d, tmat, zmat, heinv = args
            nmem = zeta.size
            w = tmat @ zeta
            jb = 0.5 * zeta.transpose() @ heinv @ zeta
            jo = 0.5 * (zmat@w - d).transpose() @ (zmat@w - d)
            #j = 0.5 * (zeta.transpose() @ heinv @ zeta + (zmat@w - d).transpose() @ (zmat@w - d))
        logger.info(f"jb:{jb:.6e} jo:{jo:.6e}")
        j = jb + jo
        return j
    

    def calc_grad_j(self, zeta, *args):
        if not self.incremental:
            xc, i, pf, y, yloc, tmat, gmat, heinv, rinv = args
            x = xc + gmat @ zeta
            hx = self.obs.h_operator(yloc, x)
            ob = y - hx
            if self.ltlm:
                dh = self.obs.dh_operator(yloc,x) @ pf
            else:
                xe = x[:,None] + pf
                dh = self.obs.h_operator(yloc,xe) - self.obs.h_operator(yloc,x)[:, None]
            grad = heinv @ zeta - tmat @ dh.transpose() @ rinv @ ob
        else:
            # incremental form
            d, tmat, zmat, heinv = args
            nmem = zeta.size
            w = tmat @ zeta
            grad = heinv @ zeta + tmat @ zmat.transpose() @ (zmat@w - d)
        return grad

    def calc_hess(self, zeta, *args):
        if not self.incremental:
            xc, i, pf, y, yloc, tmat, gmat, heinv, rinv = args
            x = xc + gmat @ zeta
            if self.ltlm:
                dh = self.obs.dh_operator(yloc,x) @ pf
            else:
                xe = x[:,None] + pf
                dh = self.obs.h_operator(yloc,xe) - self.obs.h_operator(yloc,x)[:, None]
            hess = tmat @ (np.eye(zeta.size) + dh.transpose() @ rinv @ dh) @ tmat
        else:
            # incremental form
            d, tmat, zmat, heinv = args
            hess = tmat @ (np.eye(zeta.size) + zmat.transpose() @ zmat) @ tmat
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
            logger.debug(f"{xloc} {obsloc[k]} {dist[k]}")
        far = far[dist>dist0]
        logger.debug(far)
        Rwf_loc = np.exp(-0.5*(dist/loc_scale)**2)
        return far, Rwf_loc

    def __call__(self, xb, pb, y, yloc, r=None, rmat=None, rinv=None,
        method="CGF", cgtype=1,
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
        if save_dh:
            np.save("{}_pf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), fpf)
            np.save("{}_spf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), pf)
        
        xa = xc.copy()
        pa = np.zeros_like(pf)
        ob = y - self.obs.h_operator(yloc,xc)
        if self.ltlm:
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
        wlist = []
        Wlist = []
        for i in range(xc.size):
            far, Rwf_loc = self.r_loc(self.lsig, yloc, float(i))
            logger.info(f"Number of assimilated obs.={y.size - len(far)}")
            Rmat = np.diag(np.diag(rmat) * np.sqrt(Rwf_loc))
            Rmat = np.delete(Rmat, far, axis=0)
            Rmat = np.delete(Rmat, far, axis=1)
            yi = np.delete(y, far)
            yiloc = np.delete(yloc, far)
            #if not self.incremental:
                #if self.ltlm:
                #    xtmp = np.zeros_like(xc)
                #    xtmp[i] = xc[i]
                #    logger.debug("xtmp={}".format(xtmp))
                #    dhtmp = self.obs.dh_operator(yiloc,xtmp) @ pf
                #else:
                #    xtmp = np.zeros_like(xc)
                #    xtmp[i] = xc[i]
                #    xetmp = np.zeros_like(pf)
                #    xetmp[i,:] = xc[i] + pf[i,:]
                #    dhtmp = self.obs.h_operator(yiloc,xetmp) - self.obs.h_operator(yiloc,xtmp)[:, None]
                #zmat = Rmat @ dhtmp
                #logger.debug("cond(zmat)={}".format(la.cond(zmat)))
                #tmat, heinv = self.precondition(zmat)
                #logger.debug("zmat.shape={}".format(zmat.shape))
                #logger.debug("tmat.shape={}".format(tmat.shape))
                #logger.debug("heinv.shape={}".format(heinv.shape))
            #else:
            dhi = np.delete(dh, far, axis=0)
            zmat = Rmat @ dhi
            logger.debug("cond(zmat)={}".format(la.cond(zmat)))
            tmat, heinv = self.precondition(zmat)
            logger.debug("zmat.shape={}".format(zmat.shape))
            logger.debug("tmat.shape={}".format(tmat.shape))
            logger.debug("heinv.shape={}".format(heinv.shape))
            gmat = pf @ tmat
            logger.debug("gmat.shape={}".format(gmat.shape))
            #gvec = pf[i,:] @ tmat
            #logger.debug("gvec.shape={}".format(gvec.shape))
            if not self.incremental:
                Rinv = np.diag(np.diag(rinv) * Rwf_loc)
                Rinv = np.delete(Rinv, far, axis=0)
                Rinv = np.delete(Rinv, far, axis=1)
                args_j = (xc, i, pf, yi, yiloc, tmat, gmat, heinv, Rinv)
            else:
                obi = np.delete(ob, far)
                di = Rmat @ obi
                args_j = (di, tmat, zmat, heinv)
            x0 = np.zeros(pf.shape[1])
            x = x0.copy()
            irest = 0 # restart counter
            flg = -1  # optimization result flag
            minimize = Minimize(x0.size, self.calc_j, jac=self.calc_grad_j, hess=self.calc_hess,
                            args=args_j, iprint=iprint, method=method, cgtype=cgtype,
                            maxiter=maxiter, restart=restart)
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
                #if self.model=="z08":
                #    xmax = max(np.abs(np.min(x)),np.max(x))
                #    logger.debug("resx max={}".format(xmax))
                #    if xmax < 1000:
                #        self.cost_j(1000, xf.shape[1], x, icycle, *args_j)
                #    else:
                #        xmax = int(np.ceil(xmax*0.001)*1000)
                #        logger.info("resx max={}".format(xmax))
                #        self.cost_j(xmax, xf.shape[1], x, icycle, *args_j)
                #elif self.model=="l96":
                #    self.cost_j(200, xf.shape[1], x, icycle, *args_j)
            else:
                x, flg = minimize(x0)
            wk = tmat @ x
            logger.debug(f"wk={wk.shape}")
            wlist.append(wk)
            xa[i] = xc[i] + gmat[i,:] @ x
            if self.ltlm:
                #xtmp = np.zeros_like(xc)
                #xtmp[i] = xc[i]
                #logger.debug("xtmp={}".format(xtmp))
                dhtmp = self.obs.dh_operator(yiloc,xa) @ pf
            else:
                #xtmp = np.zeros_like(xc)
                #xtmp[i] = xc[i]
                #xetmp = np.zeros_like(pf)
                #xetmp[i,:] = xc[i] + pf[i,:]
                xetmp = xa[:,None] + pf
                dhtmp = self.obs.h_operator(yiloc,xetmp) - self.obs.h_operator(yiloc,xa)[:, None]
            zmat = Rmat @ dhtmp
            logger.debug("cond(zmat)={}".format(la.cond(zmat)))
            tmat, heinv = self.precondition(zmat)
            #xa[i] = xc[i] + gmat[i] @ x
            #if self.ltlm:
            #    dh = self.obs.dh_operator(yloc,xa) @ pf
            #else:
            #    dh = self.obs.h_operator(yloc, xa[:, None] + pf) - self.obs.h_operator(yloc, xa)[:, None]
            #dhi = np.delete(dh, far, axis=0)
            #zmat = Rmat @ dhi
            #tmat, heinv = self.precondition(zmat)
            pa[i,:] = pf[i,:] @ tmat 
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
        #if save_dh:
        if save_w:
            np.save("{}_pa_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), fpa)
            np.save("{}_ua_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), u)
        return u, fpa, pa, innv, chi2, ds
