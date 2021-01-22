import sys
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
from .chi_test import Chi

zetak = []
logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')
        
class Mlef():

    def __init__(self, pt, obs, infl, lsig, 
                 linf, lloc, ltlm, model="model"):
        self.pt = pt # DA type (MLEF or GRAD)
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

    def precondition(self,zmat):
        u, s, vt = la.svd(zmat)
        v = vt.transpose()
        is2r = 1 / (1 + s**2)
        #c = zmat.transpose() @ zmat
        #s, v = la.eigh(c)
        #s[s<0] = 0.0
        #is2r = 1 / (1 + s)
        #vt = v.transpose()
        tmat = v @ np.diag(np.sqrt(is2r)) @ vt
        heinv = v @ np.diag(is2r) @ vt
        logger.debug("tmat={}".format(tmat))
        logger.debug("heinv={}".format(heinv))
        logger.info("singular value ={}".format(s))
        return tmat, heinv

    def callback(self, xk):
        global zetak
        logger.debug("xk={}".format(xk))
        zetak.append(xk)

    def calc_j(self, zeta, *args):
        xc, pf, y, tmat, gmat, heinv, rinv = args
        nmem = zeta.size
        x = xc + gmat @ zeta
        ob = y - self.obs.h_operator(x)
        j = 0.5 * (zeta.transpose() @ heinv @ zeta + ob.transpose() @ rinv @ ob)
        logger.debug("zeta.shape={}".format(zeta.shape))
        logger.debug("j={} zeta={}".format(j, zeta))
        return j
    

    def calc_grad_j(self, zeta, *args):
        xc, pf, y, tmat, gmat, heinv, rinv = args
        nmem = zeta.size
        x = xc + gmat @ zeta
        hx = self.obs.h_operator(x)
        ob = y - hx
        #if self.pt == "grad":
        if self.ltlm:
            dh = self.obs.dhdx(x) @ pf
        else:
            dh = self.obs.h_operator(x[:, None] + pf) - hx[:, None]
        return tmat @ zeta - dh.transpose() @ rinv @ ob
        
    def cost_j(self, nx, nmem, xopt, icycle, *args):
        xc, pf, y, tmat, gmat, heinv, rinv= args
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

    def chi2_test(self, zmat, heinv, rmat, d):
        p = d.size
        G_inv = np.eye(p) - zmat @ heinv @ zmat.T
        innv = rmat @ d[:,None]
        return innv.T @ G_inv @ innv / p

    def dof(self, zmat):
        u, s, vt = la.svd(zmat)
        ds = np.sum(s**2/(1.0+s**2))
        return ds

    def pfloc(self, sqrtpf, save_dh, icycle):
        nmem = sqrtpf.shape[1]
        pf = sqrtpf @ sqrtpf.T
        if save_dh:
            np.save("{}_pf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), pf)
            np.save("{}_spf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), sqrtpf)
        dist, l_mat = self.loc_mat(self.lsig, pf.shape[0], pf.shape[1])
        if save_dh:
            np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.pt), l_mat)
        pf = pf * l_mat
        lam, v = la.eig(pf)
        lam[nmem:] = 0.0
        logger.debug("eigen value = {}".format(lam))
        pf = v @ np.diag(lam) @ v.T
        spf = v[:,:nmem] @ np.diag(np.sqrt(lam[:nmem])) 
        #spf0 = v @ np.diag(np.sqrt(lam)) @ v.T
        #spf = spf0[:,:nmem]
        logger.debug("pf - spf@spf.T={}".format(pf - spf@spf.T))
        if save_dh:
            np.save("{}_lpf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), pf)
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

    def __call__(self, xb, pb, y, gtol=1e-6, 
        disp=False, save_hist=False, save_dh=False, icycle=0):
        global zetak
        zetak = []
        r, rmat, rinv = self.obs.set_r(y.size)
        xf = xb[:, 1:]
        xc = xb[:, 0]
        nmem = xf.shape[1]
        chi2_test = Chi(y.size, nmem, rmat)
        pf = xf - xc[:, None]
        if self.linf:
            logger.info("==inflation==, alpha={}".format(self.infl_parm))
            pf *= self.infl_parm
        if self.lloc:
            logger.info("==localization==, lsig={}".format(self.lsig))
            pf = self.pfloc(pf, save_dh, icycle)
            xf = xc[:, None] + pf
        logger.debug("norm(pf)={}".format(la.norm(pf)))
        logger.debug("r={}".format(np.diag(r)))
        #if self.pt == "grad":
        if self.ltlm:
            logger.debug("dhdx.shape={}".format(self.obs.dhdx(xc).shape))
            dh = self.obs.dhdx(xc) @ pf
        else:
            dh = self.obs.h_operator(xf) - self.obs.h_operator(xc)[:, None]
        if save_dh:
            np.save("{}_dh_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), dh)
            ob = y - self.obs.h_operator(xc)
            np.save("{}_d_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), ob)
        logger.info("save_dh={}".format(save_dh))
        zmat = rmat @ dh
        logger.debug("cond(zmat)={}".format(la.cond(zmat)))
        tmat, heinv = self.precondition(zmat)
        logger.debug("pf.shape={}".format(pf.shape))
        logger.debug("tmat.shape={}".format(tmat.shape))
        logger.debug("heinv.shape={}".format(heinv.shape))
        gmat = pf @ tmat
        logger.debug("gmat.shape={}".format(gmat.shape))
        x0 = np.zeros(xf.shape[1])
        args_j = (xc, pf, y, tmat, gmat, heinv, rinv)
        logger.info("save_hist={}".format(save_hist))
        if save_hist:
            g = self.calc_grad_j(x0, *args_j)
            res = spo.minimize(self.calc_j, x0, args=args_j, method='BFGS', \
                jac=self.calc_grad_j, options={'gtol':gtol, 'disp':disp}, callback=self.callback)
            jh = np.zeros(len(zetak))
            gh = np.zeros(len(zetak))
            for i in range(len(zetak)):
                jh[i] = self.calc_j(np.array(zetak[i]), *args_j)
                g = self.calc_grad_j(np.array(zetak[i]), *args_j)
                gh[i] = np.sqrt(g.transpose() @ g)
            np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), jh)
            np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), gh)
            if self.model=="z08":
                xmax = max(np.abs(np.min(res.x)),np.max(res.x))
                logger.debug("resx max={}".format(xmax))
                if xmax < 1000:
                    self.cost_j(1000, xf.shape[1], res.x, icycle, *args_j)
                else:
                    xmax = int(xmax*0.01+1)*100
                    logger.debug("resx max={}".format(xmax))
                    self.cost_j(xmax, xf.shape[1], res.x, icycle, *args_j)
            elif self.model=="l96":
                self.cost_j(200, xf.shape[1], res.x, icycle, *args_j)
        else:
            res = spo.minimize(self.calc_j, x0, args=args_j, method='BFGS', \
                jac=self.calc_grad_j, options={'gtol':gtol, 'disp':disp})
        logger.info("success={} message={}".format(res.success, res.message))
        logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
            res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))
        xa = xc + gmat @ res.x
        if save_dh:
            np.save("{}_dx_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), gmat@res.x)
        #if self.pt == "grad":
        if self.ltlm:
            dh = self.obs.dhdx(xa) @ pf
        else:
            dh = self.obs.h_operator(xa[:, None] + pf) - self.obs.h_operator(xa)[:, None]
        zmat = rmat @ dh
        logger.debug("cond(zmat)={}".format(la.cond(zmat)))
        tmat, heinv = self.precondition(zmat)
        d = y - self.obs.h_operator(xa)
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
        #if infl:
        #    logger.info("==inflation==")
        #    pa *= self.infl_parm

        u = np.zeros_like(xb)
        u[:, 0] = xa
        u[:, 1:] = xa[:, None] + pa
        return u, pa, innv, chi2, ds
