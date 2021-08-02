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
    def __init__(self, pt, nmem, obs, infl, lsig, 
                 linf, lloc, ltlm,
                 step, nt, window_l, model="model"):
        self.pt = pt # DA type (prefix 4d + MLEF)
        self.nmem = nmem # ensemble size
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.infl_parm = infl # inflation parameter
        self.lsig = lsig # localization parameter
        self.linf = linf # True->Apply inflation False->Not apply
        self.lloc = lloc # True->Apply localization False->Not apply
        self.ltlm = ltlm # True->Use tangent linear approximation False->Not use
        self.step = step 
        self.nt = nt
        self.window_l = window_l
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig} infl_parm={self.infl_parm} lsig={self.lsig}")
        logger.info(f"linf={self.linf} lloc={self.lloc} ltlm={self.ltlm}")
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
        nmem = sqrtpf.shape[1]
        pf = sqrtpf @ sqrtpf.T
        #if save_dh:
        #    np.save("{}_pf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), pf)
        #    np.save("{}_spf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), sqrtpf)
        dist, l_mat = self.loc_mat(self.lsig, pf.shape[0], pf.shape[1])
        if save_dh:
            np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.pt), l_mat)
        pf = pf * l_mat
        if save_dh:
            np.save("{}_lpf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), pf)
        lam, v = la.eigh(pf)
        lam = lam[::-1]
        v = v[:,::-1]
        if save_dh:
            np.save("{}_lpfeig_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), lam)
        logger.info("pf eigen value = {}".format(lam))
        pf = v[:,:nmem] @ np.diag(lam[:nmem]) @ v[:,:nmem].T
        spf = v[:,:nmem] @ np.diag(np.sqrt(lam[:nmem])) 
        logger.info("pf - spf@spf.T={}".format(np.mean(pf - spf@spf.T)))
        if save_dh:
            np.save("{}_lpfr_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), pf)
            np.save("{}_lspf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), spf)
        return spf

    def loc_mat(self, sigma, nx, ny):
        dist = np.zeros((nx,ny))
        l_mat = np.zeros_like(dist)
        for j in range(nx):
            for i in range(ny):
                dist[j,i] = min(abs(j-i),nx-abs(j-i))
        d0 = 2.0 * np.sqrt(10.0/3.0) * sigma
        l_mat = np.exp(-dist**2/(2.0*sigma**2))
        l_mat[dist>d0] = 0
        return dist, l_mat 

    def __call__(self, xb, pb, y, yloc, method="LBFGS", cgtype=None,
        gtol=1e-6, maxiter=None,
        disp=False, save_hist=False, save_dh=False, icycle=0):
        global zetak, alphak
        zetak = []
        alphak = []
        logger.debug(f"obsloc={yloc.shape}")
        logger.debug(f"obssize={y.shape}")
        r, rmat, rinv = self.obs.set_r(y.shape[1])
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
        if self.lloc:
            logger.info("==localization==, lsig={}".format(self.lsig))
            pf = self.pfloc(pf, save_dh, icycle)
            xf = xc[:, None] + pf
        logger.debug("norm(pf)={}".format(la.norm(pf)))
        logger.debug("r={}".format(np.diag(r)))
        xl = np.zeros_like(xb)
        xl = xb[:,:]
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
        xl = np.zeros_like(xb)
        xl[:, 1:] = xa[:, None] + pf 
        xl[:, 0] = xa
        xlc = xa
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
        if save_dh:
            np.save("{}_pa_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), pa)
            ua = np.zeros((xa.size, nmem+1))
            ua[:, 0] = xa
            ua[:, 1:] = xa[:, None] + pa
            np.save("{}_ua_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), ua)
        if self.linf:
            logger.info("==inflation==, alpha={}".format(self.infl_parm))
            pa *= self.infl_parm

        u = np.zeros_like(xb)
        u[:, 0] = xa
        u[:, 1:] = xa[:, None] + pa
        fpa = pa @ pa.T
        #return u, fpa, pa, innv, chi2, ds
        return u, fpa, ds
