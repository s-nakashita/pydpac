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
        
class Lmlef4d():
# 4-dimensional MLEF with Observation space localization
# Reference: Yokota et al. (2016,SOLA)
    def __init__(self, nmem, obs, step, nt, a_window,
        nvars=1,ndims=1,
        linf=False, infl_parm=1.0, 
        iloc=0, lsig=-1.0, calc_dist1=None, 
        ltlm=False, incremental=True, model="model"):
        # necessary parameters
        self.pt = "4dmlef" # DA type
        self.nmem = nmem # ensemble size
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.step = step # forward model
        self.nt = nt # assimilation interval
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
        from .lmlef import Lmlef
        self.lmlef = Lmlef(self.nmem, self.obs, \
            iloc=self.iloc,calc_dist1=self.calc_dist1)
        logger.info(f"R-localization type : {self.pt}")
        logger.info(f"nt={self.nt} a_window={self.a_window}")

    def calc_pf(self, xf, pa, cycle):
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
                xc, y, yloc, tmat, gmat, zmat, rmat = args
            else:
            # incremental form
                d, tmat, zmat = args
            nmem = v.size
            w = tmat @ v
            j = 0.5 * w.transpose() @ w
            if not self.incremental:
                x = xc + gmat @ v
            for l in range(len(zmat)):
                if not self.incremental:
                    ob = y[l] - self.obs.h_operator(yloc[l], x)
                    ob = rmat[l] @ ob
                    for k in range(self.nt):
                        x = self.step(x)
                else:
                    ob = zmat[l]@w - d[l]
                j += 0.5 * np.dot(ob,ob)
        else: #Y
            # control variable v=w
            d_, zmat = args # local normalized innovation = R^{-1/2}(y-Hx), and local Z
            j = 0.5 * np.dot(v,v) / self.infl_parm
            for l in range(len(d_)):
                j += 0.5 * np.dot(d_[l],d_[l])
        return j

    def calc_grad_j(self, v, *args):
        if self.iloc == -1: #CW
            # control variable v=zeta
            if not self.incremental:
                xc, y, yloc, tmat, gmat, zmat, rmat = args
            else:
            # incremental form
                d, tmat, zmat = args
            nmem = v.size
            w = tmat @ v
            grad = tmat @ w
            if not self.incremental:
                x = xc + gmat @ v
            for l in range(len(zmat)):
                if not self.incremental:
                    hx = self.obs.h_operator(yloc[l], x)
                    ob = hx - y[l]
                    ob = rmat[l] @ ob
                #    if self.ltlm:
                #        dh = self.obs.dh_operator(yloc, x) @ pf
                #    else:
                #        dh = self.obs.h_operator(yloc, x[:, None] + pf) - hx[:, None]
                #    ob = dh.transpose() @ rmat[l].transpose() @ ob
                    ob = zmat[l].transpose() @ ob
                    for k in range(self.nt):
                        x = self.step(x)
                else:
                    ob = zmat[l] @ w - d[l]
                    ob = zmat[l].transpose() @ ob
                grad = grad + tmat @ ob
        else: #Y
            # control variable v=w
            d_, zmat = args
            grad = v / self.infl_parm
            for l in range(len(d_)):
                grad -= zmat[l].transpose() @ d_[l]
        return grad

    def calc_hess(self, v, *args):
        if self.iloc == -1: #CW
            # control variable v=zeta
            if not self.incremental:
                xc, y, yloc, tmat, gmat, zmat, rmat = args
            else:
            # incremental form
                d, tmat, zmat = args
            hess = np.eye(v.size)
            for l in range(len(zmat)):
                hess += zmat[l].transpose() @ zmat[l]
            hess = tmat @ hess @ tmat
        else: #Y
            # control variable v=w
            d_, zmat = args
            hess = np.eye(v.size) / self.infl_parm 
            for l in range(len(d_)):
                hess += zmat[l].transpose() @ zmat[l]
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
        return self.lmlef.dof(zmat)

    def r_loc(self, sigma, obsloc, xloc):
        return self.lmlef.r_loc(sigma, obsloc, xloc)

    def __call__(self, xb, pb, y, yloc,
        method="CG", cgtype=1, gtol=1e-6, maxiter=None, restart=False, maxrest=20, update_ensemble=False,
        disp=False, save_hist=False, save_dh=False, icycle=0):
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
        
        xa = xc.copy()
        pa = np.zeros_like(pf)
        xl = np.zeros((xf.shape[0], xf.shape[1]+1))
        xl[:, 0] = xc 
        xl[:, 1:] = xf
        xlc = xl[:, 0]
        pl = np.zeros_like(pf)
        pl = pf[:,:]
        dxflist = [] # background perturbations
        dhlist = [] # observation perturbations
        dlist = [] # innovations
        for l in range(min(self.a_window, y.shape[0])):
            dxflist.append(pl)
            if self.ltlm:
                logger.debug("dhdx={}".format(self.obs.dh_operator(yloc[l],xlc)))
                dh = self.obs.dh_operator(yloc[l],xlc) @ pl
            else:
                dh = self.obs.h_operator(yloc[l],xl[:,1:]) - self.obs.h_operator(yloc[l],xlc)[:, None]
            dhlist.append(dh)
            ob = y[l] - self.obs.h_operator(yloc[l],xlc)
            dlist.append(ob)
            if save_dh:
                np.save("{}_dh_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), dh)
                np.save("{}_d_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), ob) 
            for k in range(self.nt):
                xl = self.step(xl)
            xlc = xl[:, 0]
            pl = xl[:, 1:] - xlc[:, None]
        logger.info("save_dh={}".format(save_dh))
        logger.info("save_hist={}".format(save_hist))
        if self.iloc == -1: #CW
            logger.info("==R-localization, {}==, lsig={}".format(self.loctype[self.iloc],self.lsig))
            iprint = np.zeros(2, dtype=np.int32)
            options = {'gtol':gtol, 'disp':disp, 'maxiter':maxiter}
            for i in range(xc.size):
                zmat = []
                di = []
                yi = []
                yiloc = []
                ri = []
                for l in range(len(dhlist)):
                    far, Rwf_loc = self.r_loc(self.lsig, yloc[l], float(i))
                    logger.debug(f"Number of assimilated obs.={(y[l].size - len(far))*self.a_window}")
                    Rmat = np.diag(np.diag(rmat) * np.sqrt(Rwf_loc))
                    Rmat = np.delete(Rmat, far, axis=0)
                    Rmat = np.delete(Rmat, far, axis=1)
                    dh = dhlist[l]
                    ob = dlist[l]
                    dhi = np.delete(dh, far, axis=0)
                    zmat.append(Rmat @ dhi)
                    if not self.incremental:
                        yi.append(np.delete(y[l], far))
                        yiloc.append(np.delete(yloc[l], far))
                        ri.append(Rmat)
                    obi = np.delete(ob, far)
                    di.append(Rmat @ obi)
                logger.debug("cond(zmat)={}".format(la.cond(zmat)))
                tmat, heinv = self.precondition(zmat)
                logger.debug("tmat.shape={}".format(tmat.shape))
                logger.debug("heinv.shape={}".format(heinv.shape))
                #gvec = pf[i,:] @ tmat
                gmat = pf @ tmat
                logger.debug("gmat.shape={}".format(gmat.shape))
                x0 = np.zeros(pf.shape[1])
                if not self.incremental:
                    args_j = (xc, yi, yiloc, tmat, gmat, zmat, ri)
                else:
                    args_j = (di, tmat, zmat)
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
                xa[i] = xc[i] + gmat[i] @ x
                zmat = [] # observation perturbations
                d = [] # normalized innovation vectors
                xlc = xa
                for l in range(min(self.a_window, y.shape[0])):
                    far, Rwf_loc = self.r_loc(self.lsig, yloc[l], float(i))
                    Rmat = np.diag(np.diag(rmat) * np.sqrt(Rwf_loc))
                    Rmat = np.delete(Rmat, far, axis=0)
                    Rmat = np.delete(Rmat, far, axis=1)
                    pl = dxflist[l]
                    if self.ltlm:
                        logger.debug("dhdx={}".format(self.obs.dh_operator(yloc[l],xlc)))
                        dh = self.obs.dh_operator(yloc[l],xlc) @ pl
                    else:
                        dh = self.obs.h_operator(yloc[l],xlc[:,None]+pl) - self.obs.h_operator(yloc[l],xlc)[:, None]
                    dhi = np.delete(dh, far, axis=0)
                    zmat.append(Rmat @ dhi)
                    for k in range(self.nt):
                        xlc = self.step(xlc)
                tmat, heinv = self.precondition(zmat)
                pa[i,:] = pf[i,:] @ tmat 
        else: #Y
            logger.info("==R-localization, {}==, lsig={}".format(self.loctype[self.iloc],self.lsig))
            if maxiter is None:
                maxiter = 5
            iter=0
            w = np.zeros((xa.size,nmem))
            while iter < maxiter:
                gnorm = 0.0
                for i in range(xa.size):
                    zmat = []
                    d_ = []
                    for l in range(len(dhlist)):
                        far, Rwf_loc = self.r_loc(self.lsig, yloc[l], float(i))
                        logger.debug(f"Number of assimilated obs.={y[l].size - len(far)}")
                        Rmat = np.diag(np.diag(rmat) * np.sqrt(Rwf_loc))
                        Rmat = np.delete(Rmat, far, axis=0)
                        Rmat = np.delete(Rmat, far, axis=1)
                        dh = dhlist[l]
                        ob = dlist[l]
                        dhi = np.delete(dh, far, axis=0)
                        zmat.append(Rmat @ dhi)
                        obi = np.delete(ob, far)
                        d_.append(Rmat @ obi)
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
                xlc = xa
                dhlist = [] # observation perturbations
                dlist = [] # innovations
                for l in range(min(self.a_window, y.shape[0])):
                    pl = dxflist[l]
                    if self.ltlm:
                        logger.debug("dhdx={}".format(self.obs.dh_operator(yloc[l],xlc)))
                        dh = self.obs.dh_operator(yloc[l],xlc) @ pl
                    else:
                        dh = self.obs.h_operator(yloc[l],xlc[:,None]+pl) - self.obs.h_operator(yloc[l],xlc)[:, None]
                    dhlist.append(dh)
                    ob = y[l] - self.obs.h_operator(yloc[l],xlc)
                    dlist.append(ob)
                    for k in range(self.nt):
                        xlc = self.step(xlc)
                iter += 1
            ## update ensemble
            for i in range(xa.size):
                zmat = []
                for l in range(len(dhlist)):
                    far, Rwf_loc = self.r_loc(self.lsig, yloc[l], float(i))
                    Rmat = np.diag(np.diag(rmat) * np.sqrt(Rwf_loc))
                    Rmat = np.delete(Rmat, far, axis=0)
                    Rmat = np.delete(Rmat, far, axis=1)
                    dh = dhlist[l]
                    dhi = np.delete(dh, far, axis=0)
                    zmat.append(Rmat @ dhi)
                tmat, heinv = self.precondition(zmat)
                pa[i,:] = pf[i,:] @ tmat
        # statistical evaluation
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
        innv, chi2 = chi2_test(zmat, d)
        ds = self.dof(zmat)
        logger.info("dof={}".format(ds))
        if save_dh:
            np.save("{}_dx_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), xa - xc)
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
        return u, fpa, pa, innv, chi2, ds
