import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import copy
from .chi_test import Chi
from .minimize import Minimize

zetak = []
alphak = []
logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')
        
class Mlefw():

    def __init__(self, pt, state_size, nmem, obs, 
        linf=False, infl_parm=1.0, 
        iloc=None, lsig=-1.0, ss=False, gain=False,
        l_mat=None, l_sqrt=None,
        calc_dist=None, calc_dist1=None, 
        ltlm=False, incremental=False, model="model"):
        # necessary parameters
        self.pt = pt # DA type (MLEF or GRAD)
        self.ndim = state_size # state size
        self.nmem = nmem # ensemble size
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        # optional parameters
        # inflation
        self.linf = linf # True->Apply inflation False->Not apply
        self.infl_parm = infl_parm # inflation parameter
        # localization
        self.iloc = iloc # iloc = None->No localization
                         #      = 0   ->R-localization (mlef_rloc.py)
                         #      = 1   ->Eigen value decomposition of localized Pf
                         #      = 2   ->Modulated ensemble
        self.lsig = lsig # localization parameter
        self.ss = ss     # ensemble reduction method : True->Use stochastic sampling
        self.gain = gain # ensemble reduction method : True->Use reduced gain (Bishop et al. 2017)
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
        self.rs = np.random.RandomState()
        # tangent linear
        self.ltlm = ltlm # True->Use tangent linear approximation False->Not use
        # incremental form
        self.incremental = incremental
        # forecast model name
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"ndim={self.ndim} nmem={self.nmem}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig} infl_parm={self.infl_parm} lsig={self.lsig}")
        logger.info(f"linf={self.linf} iloc={self.iloc} ltlm={self.ltlm} incremental={self.incremental}")
        if self.iloc is not None:
            if l_mat is None or l_sqrt is None:
                self.l_mat, self.l_sqrt, self.nmode, self.enswts \
                = self.loc_mat(self.lsig, self.ndim, self.ndim)
            else:
                self.l_mat = l_mat
                self.l_sqrt = l_sqrt
                self.nmode = l_sqrt.shape[1]
            np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.pt), self.l_mat)

    def calc_pf(self, xf, **kwargs):
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
        logger.debug("eigen value ={}".format(lam))
        #print(f"rank(zmat)={lam[lam>1.0e-10].shape[0]}")
        return tmat, heinv

    def callback(self, xk, alpha=None):
        global wk, alphak
        logger.debug("xk={}".format(xk))
        wk.append(copy.copy(xk))
        if alpha is not None:
            alphak.append(alpha)

    def calc_j(self, w, *args):
        if not self.incremental:
            xc, pf, y, yloc, rinv = args
            x = xc + pf @ w
            ob = y - self.obs.h_operator(yloc, x)
            jb = 0.5 * w.transpose() @ w
            jo = 0.5 * ob.transpose() @ rinv @ ob
        else:
        ## incremental form
            d, zmat = args
            jb = 0.5 * w.transpose() @ w 
            jo = 0.5 * (zmat@w - d).transpose() @ (zmat@w - d)
        logger.info(f"jb:{jb:.6e} jo:{jo:.6e}")
        j = jb + jo
        return j

    def calc_grad_j(self, w, *args):
        if not self.incremental:
            xc, pf, y, yloc, rinv = args
            x = xc + pf @ w
            hx = self.obs.h_operator(yloc, x)
            ob = y - hx
            if self.ltlm:
                dh = self.obs.dh_operator(yloc, x) @ pf
            else:
                dh = self.obs.h_operator(yloc, x[:, None] + pf) - hx[:, None]
            grad = w - dh.transpose() @ rinv @ ob
        else:
        ## incremental form
            d, zmat = args
            grad = w + zmat.transpose() @ (zmat@w - d)
        return grad 

    def calc_hess(self, w, *args):
        if not self.incremental:
            xc, pf, y, yloc, rinv = args
            x = xc + pf @ w
            if self.ltlm:
                dh = self.obs.dh_operator(yloc, x) @ pf
            else:
                dh = self.obs.h_operator(yloc, x[:, None] + pf) - self.obs.h_operator(yloc, x)[:, None]
            hess = np.eye(w.size) + dh.transpose() @ rinv @ dh
        else:
        ## incremental form
            d, zmat = args
            hess = np.eye(w.size) + zmat.transpose() @ zmat
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
        # Zupanski, D. et al., (2007) Applications of information theory in ensemble data assimilation
        # Eq. (10)
        u, s, vt = la.svd(zmat)
        ds = np.sum(s**2/(1.0+s**2))
        return ds

    def pfloc(self, sqrtpf, save_dh, icycle):
        #nmode = min(100, self.ndim)
        #logger.info(f"== Pf localization, nmode={nmode} ==")
        pf = sqrtpf @ sqrtpf.T
        pf = pf * self.l_mat
        #if save_dh:
        #    np.save("{}_lpf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), pf)
        lam, v = la.eigh(pf)
        lam = lam[::-1]
        lam[lam < 0.0] = 0.0
        lamsum = lam.sum()
        v = v[:,::-1]
        if save_dh:
            np.save("{}_lpfeig_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), lam)
        logger.info("pf eigen value = {}".format(lam))
        nmode = 1
        thres = 0.99
        frac = 0.0
        while frac < thres:
            frac = np.sum(lam[:nmode]) / lamsum
            nmode += 1
        nmode = min(nmode, self.ndim)
        logger.info(f"== eigen value decomposition of Pf, nmode={nmode} ==")
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

    def __call__(self, xb, pb, y, yloc, r=None, rmat=None, rinv=None,
        method="NCG", cgtype=None,
        gtol=1e-6, maxiter=None, restart=False, maxrest=20, update_ensemble=False,
        disp=False, save_hist=False, save_dh=False, icycle=0):
        global wk, alphak
        wk = []
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
        if self.iloc is not None:
            logger.info("==localization==, lsig={}".format(self.lsig))
            pf_orig = pf.copy()
            if self.iloc == 1:
                pf, wts = self.pfloc(pf_orig, save_dh, icycle)
            elif self.iloc == 2:
                pf = self.pfmod(pf_orig, save_dh, icycle)
            logger.info("pf.shape={}".format(pf.shape))
            xf = xc[:, None] + pf
        #logger.debug("norm(pf)={}".format(la.norm(pf)))
        #logger.debug("r={}".format(np.diag(r)))
        if self.ltlm:
            dh = self.obs.dh_operator(yloc,xc) @ pf
        else:
            dh = self.obs.h_operator(yloc,xc[:, None]+pf) - self.obs.h_operator(yloc,xc)[:, None]
        ob = y - self.obs.h_operator(yloc,xc)
        if save_dh:
            np.save("{}_dh_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), dh)
            np.save("{}_d_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), ob)
        logger.info("save_dh={}".format(save_dh))
        zmat = rmat @ dh
        d = rmat @ ob
        #logger.debug("cond(zmat)={}".format(la.cond(zmat)))
        #tmat, heinv = self.precondition(zmat)
        logger.debug("pf.shape={}".format(pf.shape))
        #logger.debug("tmat.shape={}".format(tmat.shape))
        #logger.debug("heinv.shape={}".format(heinv.shape))
        #gmat = pf @ tmat
        #logger.debug("gmat.shape={}".format(gmat.shape))
        w0 = np.zeros(pf.shape[1])
        w = w0.copy()
        if not self.incremental:
            args_j = (xc, pf, y, yloc, rinv)
        else:
            args_j = (d, zmat)
        iprint = np.ones(2, dtype=np.int32)
        irest = 0 # restart counter
        flg = -1  # optimization result flag
        options = {'gtol':gtol, 'disp':disp, 'maxiter':maxiter}
        minimize = Minimize(w0.size, self.calc_j, jac=self.calc_grad_j, hess=self.calc_hess,
                            args=args_j, iprint=iprint, method=method, cgtype=cgtype,
                            maxiter=maxiter, restart=restart)
        logger.info("save_hist={}".format(save_hist))
        if save_hist:
            w, flg = minimize(w0, callback=self.callback)
            jh = np.zeros(len(wk))
            gh = np.zeros(len(wk))
            for i in range(len(wk)):
                jh[i] = self.calc_j(np.array(wk[i]), *args_j)
                g = self.calc_grad_j(np.array(wk[i]), *args_j)
                gh[i] = np.sqrt(g.transpose() @ g)
            np.save("{}_w_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), np.array(wk))
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
            w, flg = minimize(w0)
        xa = xc + pf @ w
        if save_dh:
            np.save("{}_dx_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), xa-xc)
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
        pa = pf @ tmat
        if self.iloc is not None:
            nmem2 = pf.shape[1]
            ptrace = np.sum(np.diag(pa @ pa.T))
            if self.ss:
                # random sampling
                rvec = self.rs.standard_normal(size=(nmem2, nmem))
                #for l in range(len(wts)):
                #    rvec[l*nmem:(l+1)*nmem,:] = rvec[l*nmem:(l+1)*nmem,:] * wts[l] / np.sum(wts)
                rvec_mean = np.mean(rvec, axis=0)
                rvec = rvec - rvec_mean[None,:]
                rvec_stdv = np.sqrt((rvec**2).sum(axis=0) / (nmem2-1))
                rvec = rvec / rvec_stdv[None,:]
                logger.debug("rvec={}".format(rvec[:,0]))
                pa = pf @ tmat @ rvec / np.sqrt(nmem-1)
            elif self.gain:
                if self.ltlm:
                    dh = self.obs.dh_operator(yloc,xa) @ pf_orig
                else:
                    dh = self.obs.h_operator(yloc, xa[:, None] + pf_orig) - self.obs.h_operator(yloc, xa)[:, None]
                zmat_orig = rmat @ dh
                u, s, vt = la.svd(zmat, full_matrices=False)
                logger.debug(f"s.shape={s.shape}")
                logger.debug(f"u.shape={u.shape}")
                logger.debug(f"vt.shape={vt.shape}")
                sp = s**2 + 1.0
                D = (1.0 - np.sqrt(1.0/sp))/s
                nsig = D.size
                reducedgain = pf @ vt.transpose() @ np.diag(D) @ u.transpose()
                pa = pf_orig - reducedgain @ zmat_orig
            else: # tmat computed from original forecast ensemble
                if self.ltlm:
                    dh = self.obs.dh_operator(yloc,xa) @ pf_orig
                else:
                    dh = self.obs.h_operator(yloc, xa[:, None] + pf_orig) - self.obs.h_operator(yloc, xa)[:, None]
                zmat_orig = rmat @ dh
                tmat, heinv = self.precondition(zmat_orig)
                pa = pf_orig @ tmat
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
        return u, fpa, pa, innv, chi2, ds
