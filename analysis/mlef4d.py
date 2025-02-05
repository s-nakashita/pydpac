import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
import copy
from .chi_test import Chi
from .minimize import Minimize

zetak = []
alphak = []
logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')
        
class Mlef4d():
# 4-dimensional MLEF
# Reference(En4DVar) Liu et al. 2008: "An ensemble-based four-dimensional variational data assimilation scheme. Part I: Technical formulation and preliminary test," Mon. Wea. Rev., 136, 3363-3373.
    def __init__(self, state_size, nmem, obs, step, nt, a_window, 
                nvars=1,ndims=1,
                linf=False, infl_parm=1.0,
                lloc=False, iloc=None, lsig=-1.0, calc_dist=None, calc_dist1=None,
                incremental=True,ltlm=False, model="model"):
        # necessary parameters
        self.pt = "4dmlef" # DA type (prefix 4d + MLEF)
        self.ndim = state_size # state size
        self.nmem = nmem # ensemble size
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.step = step # forward model
        self.nt = nt     # assimilation interval
        self.a_window = a_window # assimilation window length
        # optional parameters
        # for 2 or more variables
        self.nvars = nvars
        # for 2 or more dimensional data
        self.ndims = ndims
        # inflation
        self.linf = linf # True->Apply inflation False->Not apply
        self.infl_parm = infl_parm # inflation parameter
        # localization
        self.lloc = lloc # True->Apply localization, False->Not apply
        self.iloc = iloc # iloc = None->No localization
                         #      <=0   ->R-localization (lmlef4d.py)
                         #      = 1   ->Eigen value decomposition of localized Pf
                         #      = 2   ->Modulated ensemble
        self.lsig = lsig # localization parameter
        if self.lloc:
            if self.iloc is None:
                self.iloc = 1
        else:
            self.iloc = None
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
        self.rs = np.random.default_rng() # random generator
        # tangent linear
        self.ltlm = ltlm # True->Use tangent linear approximation False->Not use
        # incremental form
        self.incremental = incremental
        # forecast model name
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig}")
        logger.info(f"linf={self.linf} infl_parm={self.infl_parm}")
        logger.info(f"lsig={self.lsig} iloc={self.iloc}")
        logger.info(f"ltlm={self.ltlm}")
        from .mlef import Mlef
        self.mlef = Mlef(self.ndim,self.nmem,self.obs,
                #nvars=self.nvars,ndims=self.ndims,
                #linf=self.linf,infl_parm=self.infl_parm,
                #iloc=self.iloc,lsig=self.lsig,
                calc_dist=self.calc_dist,calc_dist1=self.calc_dist1,
                #ltlm=self.ltlm,incremental=self.incremental,model=self.model
                )
        if self.iloc is not None:
            if self.iloc <= 0:
                from .lmlef4d import Lmlef4d
                self.lmlef = Lmlef4d(self.nmem,self.obs,self.step,self.nt,self.a_window,
                nvars=self.nvars,ndims=self.ndims,
                linf=self.linf,infl_parm=self.infl_parm,
                iloc=self.iloc,lsig=self.lsig,calc_dist1=self.calc_dist1,
                ltlm=self.ltlm,incremental=self.incremental,model=self.model)
            else:
                self.l_mat, self.l_sqrt, self.nmode, self.enswts \
                = self.mlef.loc_mat(self.lsig, self.ndim, self.ndim)
                np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.pt), self.l_mat)
        logger.info(f"nt={self.nt} a_window={self.a_window}")

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
        if self.linf:
            logger.info("==inflation==, alpha={}".format(self.infl_parm))
            rho = 1.0 / self.infl_parm
        c = np.zeros((zmat[0].shape[1], zmat[0].shape[1]))
        for l in range(len(zmat)):
            c = c + zmat[l].transpose() @ zmat[l]
        lam, v = la.eigh(c)
        D = np.diag(1.0/(np.sqrt(lam + np.full(lam.size,rho))))
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
        zetak.append(copy.copy(xk))
        if alpha is not None:
            alphak.append(alpha)

    def calc_j(self, zeta, *args):
        if not self.incremental:
            xc, y, yloc, tmat, gmat, zmat, rmat = args
        else:
            d, tmat, zmat = args
        nmem = zeta.size
        if not self.incremental:
            x = xc + gmat @ zeta
        w = tmat @ zeta
        j = 0.5 * (w.transpose() @ w)
        for l in range(len(zmat)):
            if not self.incremental:
                ob = y[l] - self.obs.h_operator(yloc[l], x)
                ob = rmat @ ob
                for k in range(self.nt):
                    x = self.step(x)
            else:
                ob = zmat[l] @ w - d[l]
            j += 0.5 * (ob.transpose() @ ob)
        return j
    

    def calc_grad_j(self, zeta, *args):
        if not self.incremental:
            xc, y, yloc, tmat, gmat, zmat, rmat = args
        else:
            d, tmat, zmat = args
        nmem = zeta.size
        if not self.incremental:
            x = xc + gmat @ zeta
        w = tmat @ zeta
        #xl = x[:, None] + pf
        g = tmat @ w
        for l in range(len(zmat)): 
            if not self.incremental:
                ob = self.obs.h_operator(yloc[l], x) - y[l]
                ob = rmat @ ob
                #if self.ltlm:
                #    dh = self.obs.dh_operator(yloc[l], x) @ (xl - x[:, None])
                #else:
                #    dh = self.obs.h_operator(yloc[l], x[:,None]+pf) - hx[:, None]
                #ob = dh.transpose() @ rmat.transpose() @ ob
                ob = zmat[l].transpose() @ ob
                for k in range(self.nt):
                    x = self.step(x)
                #    xl = self.step(xl) 
            else:
                ob = zmat[l] @ w - d[l]
                ob = zmat[l].transpose() @ ob
            g = g + tmat @ ob
        return g

    def calc_hess(self, zeta, *args):
        if not self.incremental:
            xc, y, yloc, tmat, gmat, zmat, rmat = args
        else:
            d, tmat, zmat = args
        #x = xc + gmat @ zeta
        #xl = x[:, None] + pf
        hess = np.eye(zeta.size)
        for l in range(len(zmat)):
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
        return self.mlef.dfs(zmat)

    def pfloc(self, sqrtpf, l_mat, save_dh, icycle,\
        op="linear",pt="4dmlefbe",model="model"):
        return self.mlef.pfloc(sqrtpf,l_mat,save_dh,icycle,\
            op=op,pt=pt,model=model)
    
    def pfmod(self, sqrtpf, l_sqrt, save_dh, icycle,\
        op="linear",pt="4dmlefbm",model="model"):
        return self.mlef.pfmod(sqrtpf,l_sqrt,save_dh,icycle,\
            op=op,pt=pt,model=model)

    def __call__(self, xb, pb, y, yloc, method="CG", cgtype=None,
        gtol=1e-6, maxiter=None,
        disp=False, save_hist=False, save_dh=False, icycle=0):
        if self.iloc is not None and self.iloc <= 0:
            return self.lmlef(xb,pb,y,yloc,
            method=method,cgtype=cgtype,gtol=gtol,maxiter=maxiter,
            disp=disp,save_hist=save_hist,save_dh=save_dh,icycle=icycle)
        else:
            global zetak, alphak
            zetak = []
            alphak = []
            logger.debug(f"obsloc={yloc.shape}")
            logger.debug(f"obssize={y.shape}")
            r, rmat, rinv = self.obs.set_r(yloc[0])
            rmatall = np.diag(np.concatenate([np.diag(rmat) for i in range(len(y))]))
            logger.debug(f"rmatall={rmatall.shape}")
            xf = xb[:, 1:]
            xc = xb[:, 0]
            nmem = xf.shape[1]
            chi2_test = Chi(y.size, nmem, rmatall)
            pf = xf - xc[:, None]
            #if self.linf:
            #    logger.info("==inflation==, alpha={}".format(self.infl_parm))
            #    pf *= self.infl_parm
            fpf = pf @ pf.T
            if save_dh:
                np.save("{}_pf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), fpf)
                np.save("{}_spf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), pf)
            if self.iloc is not None:
                logger.info("==localization==, lsig={}".format(self.lsig))
                pf_orig = pf.copy()
                if self.iloc == 1:
                    pf, wts = self.pfloc(pf_orig, self.l_mat, save_dh, icycle,\
                        op=self.op,pt=self.pt,model=self.model)
                elif self.iloc == 2:
                    pf = self.pfmod(pf_orig, self.l_sqrt, save_dh, icycle, \
                        op=self.op,pt=self.pt,model=self.model)
                logger.info("pf.shape={}".format(pf.shape))
                xf = xc[:, None] + pf
            logger.debug("norm(pf)={}".format(la.norm(pf)))
            logger.debug("r={}".format(np.diag(r)))
            xl = np.zeros((xf.shape[0], xf.shape[1]+1))
            xl[:, 0] = xc 
            xl[:, 1:] = xf
            xlc = xl[:, 0]
            pl = np.zeros_like(pf)
            pl = pf[:,:]
            zmat = [] # observation perturbations
            d = [] # normalized innovations
            for l in range(min(self.a_window, y.shape[0])):
                if self.ltlm:
                    logger.debug("dhdx={}".format(self.obs.dh_operator(yloc[l],xlc)))
                    dh = self.obs.dh_operator(yloc[l],xlc) @ pl
                else:
                    dh = self.obs.h_operator(yloc[l],xl[:,1:]) - self.obs.h_operator(yloc[l],xlc)[:, None]
                zmat.append(rmat @ dh)
                ob = y[l] - self.obs.h_operator(yloc[l],xlc)
                d.append(rmat @ ob)
                if save_dh:
                    np.save("{}_dh_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), dh)
                    np.save("{}_d_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), d[l]) 
                for k in range(self.nt):
                    xl = self.step(xl)
                xlc = xl[:, 0]
                pl = xl[:, 1:] - xlc[:, None]
            logger.info("save_dh={}".format(save_dh))
            logger.debug("cond(zmat)={}".format(la.cond(zmat[0])))
            tmat, heinv = self.precondition(zmat)
            logger.debug("pf.shape={}".format(pf.shape))
            logger.debug("tmat.shape={}".format(tmat.shape))
            logger.debug("heinv.shape={}".format(heinv.shape))
            gmat = pf @ tmat
            logger.debug("gmat.shape={}".format(gmat.shape))
            x0 = np.zeros(xf.shape[1])
            if not self.incremental:
                args_j = (xc, y, yloc, tmat, gmat, zmat, rmat)
            else:
                args_j = (d, tmat, zmat)
            iprint = np.zeros(2, dtype=np.int32)
            options = {'iprint':iprint, 'method':method, 'cgtype':cgtype, \
                    'gtol':gtol, 'disp':disp, 'maxiter':maxiter}
            minimize = Minimize(x0.size, self.calc_j, jac=self.calc_grad_j, hess=self.calc_hess,
                            args=args_j, **options)
            logger.info("save_hist={}".format(save_hist))
            if save_hist:
                x, flg = minimize(x0, callback=self.callback)
                jh = np.zeros(len(zetak))
                gh = np.zeros(len(zetak))
                for i in range(len(zetak)):
                    jh[i] = self.calc_j(np.array(zetak[i]), *args_j)
                    g = self.calc_grad_j(np.array(zetak[i]), *args_j)
                    gh[i] = np.sqrt(g.transpose() @ g)
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
            xa = xc + gmat @ x
            if save_dh:
                np.save("{}_dx_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), gmat@x)
            zmat = [] # observation perturbations
            d = [] # normalized innovation vectors
            xl = np.zeros((xf.shape[0], xf.shape[1]+1))
            xl[:, 1:] = xa[:, None] + pf 
            xl[:, 0] = xa
            xlc = xa
            pl = np.zeros_like(pf)
            pl = pf[:,:]
            for l in range(min(self.a_window, y.shape[0])):
                if self.ltlm:
                    logger.debug("dhdx={}".format(self.obs.dh_operator(yloc[l],xlc)))
                    dh = self.obs.dh_operator(yloc[l],xlc) @ pl
                else:
                    dh = self.obs.h_operator(yloc[l],xl[:,1:]) - self.obs.h_operator(yloc[l],xlc)[:, None]
                zmat.append(rmat @ dh)
                ob = y[l] - self.obs.h_operator(yloc[l], xlc)
                d.append(rmat@ob)
                for k in range(self.nt):
                    xl = self.step(xl)
                xlc = xl[:, 0]
                pl = xl[:, 1:] - xlc[:, None]
            logger.debug("cond(zmat)={}".format(la.cond(zmat[0])))
            tmat, heinv = self.precondition(zmat)
            zmat = np.concatenate(zmat,axis=0)
            d = np.concatenate(d)
            logger.info("zmat shape={}".format(zmat.shape))
            logger.info("d shape={}".format(d.shape))
            self.innv, self.chi2 = chi2_test(zmat, d)
            self.ds = self.dfs(zmat)
            logger.info("dfs={}".format(self.ds))
            pa = pf @ tmat 
            if self.iloc is not None:
                # random sampling
                ptrace = np.sum(np.diag(pa @ pa.T))
                rvec = self.rs.standard_normal(size=(pf.shape[1], nmem))
                #for l in range(len(wts)):
                #    rvec[l*nmem:(l+1)*nmem,:] = rvec[l*nmem:(l+1)*nmem,:] * wts[l] / np.sum(wts)
                rvec_mean = np.mean(rvec, axis=0)
                rvec = rvec - rvec_mean[None,:]
                rvec_stdv = np.sqrt((rvec**2).sum(axis=0) / (pf.shape[1]-1))
                rvec = rvec / rvec_stdv[None,:]
                logger.debug("rvec={}".format(rvec[:,0]))
                pa = pf @ tmat @ rvec / np.sqrt(nmem-1)
                trace = np.sum(np.diag(pa @ pa.T))
                logger.info("standard deviation ratio = {}".format(np.sqrt(ptrace / trace)))
                if np.sqrt(ptrace / trace) > 1.05:
                    pa *= np.sqrt(ptrace / trace)
            #if self.linf:
            #    logger.info("==inflation==, alpha={}".format(self.infl_parm))
            #    pa *= self.infl_parm

            u = np.zeros_like(xb)
            u[:, 0] = xa
            u[:, 1:] = xa[:, None] + pa
            fpa = pa @ pa.T
            if save_dh:
                np.save("{}_pa_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), fpa)
                np.save("{}_ua_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), u)
            return u, fpa #, pa, innv, chi2, ds
