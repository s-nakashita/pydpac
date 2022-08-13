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
        
class Mles():
# 4-dimensional MLEF
# Reference(En4DVar) Liu et al. 2008: "An ensemble-based four-dimensional variational data assimilation scheme. Part I: Technical formulation and preliminary test," Mon. Wea. Rev., 136, 3363-3373.
    def __init__(self, pt, state_size, nmem, obs, step, nt, window_l, 
                nvars=1,ndims=1,
                linf=False, infl_parm=1.0,
                iloc=None, lsig=-1.0, calc_dist=None, calc_dist1=None,
                ltlm=False, model="model"):
        # necessary parameters
        self.pt = pt # DA type (prefix 4d + MLEF)
        self.ndim = state_size # state size
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
        self.iloc = iloc # iloc = None->No localization
                         #      = 0   ->R-localization (mles_rloc.py)
                         #      = 1   ->Eigen value decomposition of localized Pf
                         #      = 2   ->Modulated ensemble
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
        logger.info(f"linf={self.linf} iloc={self.iloc} ltlm={self.ltlm}")
        if self.iloc is not None:
            self.l_mat, self.l_sqrt, self.nmode, self.enswts \
            = self.loc_mat(self.lsig, self.ndim, self.ndim)
            np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.pt), self.l_mat)
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

    def pfloc(self, sqrtpf, save_dh, icycle):
        nmode = min(100, self.ndim)
        logger.info(f"== Pf localization, nmode={nmode} ==")
        pf = sqrtpf @ sqrtpf.T
        pf = pf * self.l_mat
        #if save_dh:
        #    np.save("{}_lpf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), pf)
        lam, v = la.eigh(pf)
        lam = lam[::-1]
        lam[lam < 0.0] = 0.0
        v = v[:,::-1]
        if save_dh:
            np.save("{}_lpfeig_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), lam)
        logger.info("pf eigen value = {}".format(lam))
        pf = v[:,:nmode] @ np.diag(lam[:nmode]) @ v[:,:nmode].T
        spf = v[:,:nmode] @ np.diag(np.sqrt(lam[:nmode])) 
        logger.info("pf - spf@spf.T={}".format(np.mean(pf - spf@spf.T)))
        if save_dh:
            np.save("{}_lpf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), pf)
            np.save("{}_lspf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), spf)
        return spf, np.sqrt(lam[:nmode])
    
    def pfmod(self, sqrtpf, save_dh, icycle):
        nmem = sqrtpf.shape[1]
        logger.info(f"== modulated ensemble, nmode={self.nmode} ==")
        
        spf = np.empty((self.ndim, nmem*self.nmode), sqrtpf.dtype)
        for l in range(self.nmode):
            for k in range(nmem):
                m = l*nmem + k
                spf[:, m] = self.l_sqrt[:, l]*sqrtpf[:, k]
        if save_dh:
            np.save("{}_lspf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), spf)
            fullpf = spf @ spf.T
            np.save("{}_lpf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), fullpf)
        return spf

    def loc_mat(self, sigma, nx, ny):
        dist = np.zeros((nx,ny))
        l_mat = np.zeros_like(dist)
        #for j in range(nx):
        #    for i in range(ny):
        #        dist[j,i] = min(abs(j-i),nx-abs(j-i))
        for i in range(ny):
            dist[:, i] = self.calc_dist(float(i))
        d0 = 2.0 * np.sqrt(10.0/3.0) * sigma
        l_mat = np.exp(-dist**2/(2.0*sigma**2))
        l_mat[dist>d0] = 0

        lam, v = la.eigh(l_mat)
        lam = lam[::-1]
        lam[lam < 1.e-10] = 1.e-10
        lamsum = np.sum(lam)
        v = v[:,::-1]
        nmode = 1
        thres = 0.99
        frac = 0.0
        while frac < thres:
            frac = np.sum(lam[:nmode]) / lamsum
            nmode += 1
        nmode = min(nmode, nx, ny)
        logger.info("contribution rate = {}".format(np.sum(lam[:nmode])/np.sum(lam)))
        l_sqrt = v[:,:nmode] @ np.diag(np.sqrt(lam[:nmode]/frac))
        return l_mat, l_sqrt, nmode, np.sqrt(lam[:nmode])

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
        if self.iloc is not None:
            logger.info("==localization==, lsig={}".format(self.lsig))
            pf_orig = pf.copy()
            if self.iloc == 1:
                pf, wts = self.pfloc(pf_orig, save_dh, icycle)
            elif self.iloc == 2:
                pf = self.pfmod(pf_orig, save_dh, icycle)
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
        for l in range(min(self.window_l, y.shape[0])):
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
        #args_j = (xc, pf, y, yloc, tmat, gmat, heinv, rinv)
        args_j = (d, tmat, zmat, heinv)
        iprint = np.zeros(2, dtype=np.int32)
        options = {'gtol':gtol, 'disp':disp, 'maxiter':maxiter}
        minimize = Minimize(x0.size, self.calc_j, jac=self.calc_grad_j, hess=self.calc_hess,
                            args=args_j, iprint=iprint, method=method, cgtype=cgtype,
                            maxiter=maxiter)
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
        for l in range(min(self.window_l, y.shape[0])):
            if self.ltlm:
                logger.debug("dhdx={}".format(self.obs.dh_operator(yloc[l],xlc)))
                dh = self.obs.dh_operator(yloc[l],xlc) @ pl
            else:
                dh = self.obs.h_operator(yloc[l],xl[:,1:]) - self.obs.h_operator(yloc[l],xlc)[:, None]
            zmat.append(rmat @ dh)
            ob = y[l] - self.obs.h_operator(yloc[l], xlc)
            d.append(ob)
            for k in range(self.nt):
                xl = self.step(xl)
            xlc = xl[:, 0]
            pl = xl[:, 1:] - xlc[:, None]
        logger.debug("cond(zmat)={}".format(la.cond(zmat[0])))
        tmat, heinv = self.precondition(zmat)
        logger.info("zmat shape={}".format(np.array(zmat).shape))
        logger.info("d shape={}".format(np.array(d).shape))
        #innv, chi2 = chi2_test(zmat, d)
        ds = self.dof(zmat)
        logger.info("dof={}".format(ds))
        pa = pf @ tmat 
        if self.iloc is not None:
            # random sampling
            ptrace = np.sum(np.diag(pa @ pa.T))
            rvec = np.random.randn(pf.shape[1], nmem)
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
