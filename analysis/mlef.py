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
        
class Mlef():

    def __init__(self, pt, nmem, obs, infl, lsig, 
                 linf, lloc, ltlm, model="model"):
        self.pt = pt # DA type (MLEF or GRAD)
        self.nmem = nmem # ensemble size
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.infl_parm = infl # inflation parameter
        self.lsig = lsig # localization parameter
        self.linf = linf # True->Apply inflation False->Not apply
        self.lloc = lloc # True->Apply localization False->Not apply
        self.ltlm = ltlm # True->Use tangent linear approximation False->Not use
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig} infl_parm={self.infl_parm} lsig={self.lsig}")
        logger.info(f"linf={self.linf} lloc={self.lloc} ltlm={self.ltlm}")

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
        #ob = y - self.obs.h_operator(yloc, x)
        #j = 0.5 * (zeta.transpose() @ heinv @ zeta + ob.transpose() @ rinv @ ob)
        w = tmat @ zeta
        j = 0.5 * (zeta.transpose() @ heinv @ zeta + (zmat@w - d).transpose() @ (zmat@w - d))
        return j
    

    def calc_grad_j(self, zeta, *args):
        #xc, pf, y, yloc, tmat, gmat, heinv, rinv = args
        d, tmat, zmat, heinv = args
        nmem = zeta.size
        w = tmat @ zeta
        #x = xc + gmat @ zeta
        #hx = self.obs.h_operator(yloc, x)
        #ob = y - hx
        #if self.ltlm:
        #    dh = self.obs.dh_operator(yloc, x) @ pf
        #else:
        #    dh = self.obs.h_operator(yloc, x[:, None] + pf) - hx[:, None]
        #return heinv @ zeta - tmat @ dh.transpose() @ rinv @ ob
        return heinv @ zeta + tmat @ zmat.transpose() @ (zmat@w - d)

    def calc_hess(self, zeta, *args):
        #xc, pf, y, yloc, tmat, gmat, heinv, rinv = args
        d, tmat, zmat, heinv = args
        #x = xc + gmat @ zeta
        #if self.ltlm:
        #    dh = self.obs.dh_operator(yloc, x) @ pf
        #else:
        #    dh = self.obs.h_operator(yloc, x[:, None] + pf) - self.obs.h_operator(yloc, x)[:, None]
        #hess = tmat @ (np.eye(zeta.size) + dh.transpose() @ rinv @ dh) @ tmat
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

    def __call__(self, xb, pb, y, yloc, method="CGF", cgtype=1,
        gtol=1e-6, maxiter=None,
        disp=False, save_hist=False, save_dh=False, icycle=0):
        global zetak, alphak
        zetak = []
        alphak = []
        r, rmat, rinv = self.obs.set_r(y.size)
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
        if self.ltlm:
            logger.debug("dhdx={}".format(self.obs.dhdx(xc)))
            dh = self.obs.dh_operator(yloc,xc) @ pf
        else:
            dh = self.obs.h_operator(yloc,xf) - self.obs.h_operator(yloc,xc)[:, None]
        ob = y - self.obs.h_operator(yloc,xc)
        if save_dh:
            np.save("{}_dh_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), dh)
            np.save("{}_d_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), ob)
        logger.info("save_dh={}".format(save_dh))
        zmat = rmat @ dh
        d = rmat @ ob
        logger.debug("cond(zmat)={}".format(la.cond(zmat)))
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
        return u, fpa, pa, innv, chi2, ds
