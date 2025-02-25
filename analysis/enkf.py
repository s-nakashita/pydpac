import logging
from logging import getLogger
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
from scipy.linalg import cho_solve, cho_factor
from .chi_test import Chi
from .infladap import infladap
from .inflfunc import inflfunc

fileConfig("./logging_config.ini")
logger = getLogger('anl')

class EnKF():

    def __init__(self, da, state_size, nmem, obs, 
        nvars=1,ndims=1,
        linf=False, iinf=None, infl_parm=1.0, 
        lloc=False, iloc=None, lsig=-1.0, ss=False, getkf=False, bthres=0.99,
        l_mat=None, l_sqrt=None, 
        calc_dist=None, calc_dist1=None, 
        ltlm=False, model="model"):
        # necessary parameters
        self.da = da.lower() # DA type (ETKF, PO, SRF, EAKF, or LETKF)
        self.ndim = state_size # state size
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
        self.linf = linf # inflation switch
        self.infltype = {-99:'No',-3:'adap.pre-mul.D21',-2:'adap.pre-mul.L09',-1:'fix.pre-mul',0:'post-mul',1:'add',2:'RTPP',3:'RTPS',4:'mul-lin'}
        self.iinf = iinf # iinf = None->No inflation
                         #      = -3  ->Adaptive pre-multiplicative inflation (Duc et al. 2021)
                         #      = -2  ->Adaptive pre-multiplicative inflation
                         #      = -1  ->Fixed pre-multiplicative inflation
                         #      = 0   ->Post-multiplicative inflation
                         #      = 1   ->Additive inflation
                         #      = 2   ->RTPP(Relaxation To Prior Perturbations)
                         #      = 3   ->RTPS(Relaxation To Prior Spread)
                         #      >= 4  ->Multiplicative linear inflation (Duc et al. 2020)
        self.infl_parm = infl_parm # inflation parameter
        if self.iinf is None:
            if self.linf:
                self.iinf = -1
            else:
                self.iinf = -99
        if self.iinf == -2:
            self.infladap = infladap()
            self.infl_parm_pre = np.full(self.ndim, self.infl_parm)
        paramtype = self.iinf - 4
        self.inflfunc = inflfunc("mult",paramtype=paramtype)
        # localization
        self.lloc = lloc # localization switch
        self.loctype = {None:'No',0:'R-loc',1:'EVD',2:'Modulation'}
        self.iloc = iloc # iloc = None->No localization
                         #      = 0   ->K-localization(R-localization for LETKF)
                         #      = 1   ->Eigen value decomposition of localized Pf
                         #      = 2   ->Modulated ensemble
        self.lsig = lsig # localization parameter
        # B-localization parameter
        self.ss = ss     # ensemble reduction method : True->Use stochastic sampling
        self.getkf = getkf # ensemble reduction method : True->Gain ETKF (Bishop et al. 2017)
        self.bthres = bthres # variance threshold for modulation and EVD
        if self.lloc:
            if self.iloc is None:
                self.iloc = 0
        else:
            self.iloc = None
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
        self.rs = np.random.default_rng() # random generator
        # tangent linear
        self.ltlm = ltlm # True->Use tangent linear approximation False->Not use
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"pt={self.da} op={self.op} obserr={self.sig} infl_parm={self.infl_parm} lsig={self.lsig}")
        logger.info(f"inf={self.infltype[self.iinf]} loc={self.loctype[self.iloc]} ltlm={self.ltlm}")
        #if self.iloc is not None:
        if self.iloc == 1 or self.iloc == 2:
            if l_mat is None or l_sqrt is None:
                self.l_mat, self.l_sqrt, self.nmode, self.enswts\
                = self.b_loc(self.lsig, self.ndim)
            else:
                self.l_mat = l_mat
                self.l_sqrt = l_sqrt
                self.nmode = l_sqrt.shape[1]
            np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), self.l_mat)

    def calc_pf(self, xf, **kwargs):
        dxf = xf - np.mean(xf,axis=1)[:, None]
        pf = dxf @ dxf.transpose() / (self.nmem-1)
        return pf

    def cost_j(self, w, *args):
        xf, dxf, y, yloc, rinv = args
        x = xf + dxf @ w 
        ob = y - self.obs.h_operator(yloc, x)
        jb = 0.5 * w.transpose() @ w * (self.nmem - 1)
        jo = 0.5 * ob.transpose() @ rinv @ ob
        logger.info(f"jb:{jb:.6e} jo:{jo:.6e}")
        j = jb + jo 
        return j

    def __call__(self, xf, pf, y, yloc, R=None, rmat=None, rinv=None,
        save_hist=False, save_dh=False, save_w=False, icycle=0, evalout=False):
        #xf = xb[:]
        xf_ = np.mean(xf, axis=1)
        logger.debug(f"obsloc={yloc}")
        logger.debug(f"obssize={y.size}")
        JH = self.obs.dh_operator(yloc, xf_)
        if (R is None) or (rmat is None) or (rinv is None):
            logger.info("set R")
            R, rmat, rinv = self.obs.set_r(yloc)
        else:
            logger.info("use input R")
        nmem = xf.shape[1]
        chi2_test = Chi(y.size, nmem, rmat)
        dxf = xf - xf_[:,None]
        logger.debug(xf.shape)
        xloc = np.arange(xf_.size)

        pf = dxf @ dxf.T / (nmem-1)
        logger.info("pf max={:.3e} min={:.3e}".format(np.max(pf),np.min(pf)))
        if self.iinf >= 3:
            stdv_f = np.sqrt(np.diag(pf))
        if save_dh:
            np.save("{}_pf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), pf)
            np.save("{}_spf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxf)
            wk = np.zeros(self.nmem)
            args = (xf_, dxf, y, yloc, rinv)
            jlist = []
            jlist.append(self.cost_j(wk, *args))
        if (self.iloc == 1 or self.iloc == 2) and self.da != "letkf":
            logger.info("==B-localization==, lsig={}".format(self.lsig))
            dxf_orig = dxf.copy()
            xf_orig = xf.copy()
            if self.iloc == 1:
                dxf = self.pfloc(dxf_orig, save_dh, icycle)
            elif self.iloc == 2:
                dxf = self.pfmod(dxf_orig, save_dh, icycle)
            nmem2 = dxf.shape[1]
            pf = dxf @ dxf.T / (nmem2-1)
            xf = xf_[:, None] + dxf
            logger.info("dxf.shape={}".format(dxf.shape))
        else:
            nmem2 = nmem
            dxf_orig = dxf
            xf_orig = xf
        
        if self.ltlm:
            dy = JH @ dxf
            if (self.iloc == 1 or self.iloc == 2):
                dy_orig = JH @ dxf_orig
            else:
                dy_orig = dy
        else:
            dy = self.obs.h_operator(yloc, xf) - np.mean(self.obs.h_operator(yloc, xf), axis=1)[:, None]
            if (self.iloc == 1 or self.iloc == 2):
                dy_orig = self.obs.h_operator(yloc, xf_orig) - np.mean(self.obs.h_operator(yloc, xf_orig), axis=1)[:, None]
            else:
                dy_orig = dy
        d = y - np.mean(self.obs.h_operator(yloc, xf), axis=1)
        if save_dh:
            np.save("{}_dxf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxf)
            np.save("{}_dh_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dy)
            np.save("{}_d_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), d)
        logger.info("save_dh={} cycle{}".format(save_dh, icycle))
        d_ = rmat @ d
        dy_ = rmat @ dy / np.sqrt(nmem-1)
        
        if self.iinf == -1:
            if self.da != "letkf" and self.da != "etkf":
                logger.info("==pre-multiplicative inflation==, alpha={}".format(self.infl_parm))
                dxf *= self.infl_parm
            #xf = xf_[:, None] + dxf
        
        if self.da == "etkf":
            if self.iinf == -1:
                logger.info("==pre-multiplicative inflation==, alpha={}".format(self.infl_parm))
                rho = 1.0 / self.infl_parm
            elif self.iinf == -2:
                self.infl_parm = self.infladap(self.infl_parm, d_, dy_)
                logger.info("==adaptive pre-multiplicative inflation==, alpha={:.2f}".format(self.infl_parm))
                rho = 1.0 / self.infl_parm
            else:
                rho = 1.0
            C = dy_.T @ dy_ 
            ga2f, vf = la.eigh(C)
            nrank = np.sum(ga2f>1.0e-10)
            ga2 = ga2f[::-1]
            v = vf[:,::-1]
            ga = np.where(ga2>1.0e-10,np.sqrt(ga2),0.0)
            u = np.dot(dy_,v[:,:nrank])/ga[:nrank]
            #u, ga, vt = la.svd(dy_,full_matrices=False)
            #v = vt.transpose()
            #ga2 = ga*ga
            dtilde = np.dot(u.transpose(), d_)
            logger.info("ga = {}".format(ga))
            if self.iinf==-3:
                logger.info("==singular value adaptive inflation==")
                gainf = np.zeros_like(ga)
                gainf[:nrank] = self.inflfunc.est(dtilde,ga[:nrank])
                logger.info("ga inf = {}".format(gainf))
            lam = 1.0 / (np.sqrt(ga2 + np.full(ga.size, rho)))
            logger.info("lam = {}".format(lam))
            if self.iinf>=4:
                laminf = self.inflfunc(lam,alpha1=self.infl_parm)
                logger.info("lam inf = {}".format(laminf))
            elif self.iinf==-3:
                laminf = self.inflfunc.g2f(ga,gainf)
                logger.info("lam inf = {}".format(laminf))
            else:
                laminf = lam.copy()
            self.inflfunc.pdr(dtilde, ga[:nrank], lam[:nrank], laminf[:nrank])
            #A = (nmem2-1)*A + dy.T @ rinv @ dy
            #lam, v = la.eigh(A)
            ##logger.info("eigen values={}".format(lam))
            Dinv = np.diag(lam*lam)
            TT = v @ (Dinv/(nmem2-1)) @ v.T
            T = v @ np.sqrt(np.diag(laminf)) @ v.T

            K = dxf @ TT @ dy.T @ rinv
            if save_dh:
                np.save("{}_K_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K)
                np.save("{}_dx_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K@d)
                wk = TT @ dy.T @ rinv @ d
                jlist.append(self.cost_j(wk, *args))
                logger.debug(len(jlist))
                np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(self.model, self.op, self.da, icycle), jlist)
            if self.iloc == 0: # K-localization
                logger.info("==K-localization==, lsig={}".format(self.lsig))
                l_mat = self.k_loc(sigma=self.lsig, obsloc=yloc, xloc=xloc)
                K = K * l_mat
                if save_dh:
                    np.save("{}_Kloc_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K)
                    np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), l_mat)
            #dumu, dums, dumvt = la.svd(K)
            #logger.info("singular values of K={}".format(dums))
            #print(f"rank(kmat)={dums[np.abs(dums)>1.0e-10].shape[0]}")
            xa_ = xf_ + K @ d
            dxa = dxf @ T
            if self.iinf == 2:
                logger.info("==RTPP==, alpha={}".format(self.infl_parm))
                dxa = (1.0 - self.infl_parm)*dxa + self.infl_parm*dxf

            if (self.iloc == 1 or self.iloc == 2):
                ptrace = np.sum(np.diag(dxa @ dxa.T / (nmem2-1)))
                if save_dh:
                    np.save("{}_dxaorig_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxa)
                if self.ss:
                    logger.info("random sampling")
                    rvec = self.rs.standard_normal(size=(nmem2, nmem))
                    rvec_mean = np.mean(rvec, axis=0)
                    rvec = rvec - rvec_mean[None, :]
                    rvec_stdv = np.sqrt((rvec**2).sum(axis=0)/(nmem2-1))
                    rvec = rvec / rvec_stdv[None, :]
                    logger.debug("rvec={}".format(rvec[:, 0]))
                    dxa = dxf @ T @ rvec / np.sqrt(nmem2-1)
                    dxa = dxa - dxa.mean(axis=1)[:, None]
                elif self.getkf:
                    logger.info("Gain ETKF")
                    u, s, vt = la.svd(rmat @ dy, full_matrices=False)
                    logger.debug(f"s.shape={s.shape}")
                    logger.debug(f"u.shape={u.shape}")
                    logger.debug(f"vt.shape={vt.shape}")
                    sp = s**2 + nmem2-1
                    D = (1.0 - np.sqrt((nmem2-1)/sp))#/(s**2)
                    nsig = D.size
                    #rK = dxf @ vt[:nsig,:].transpose() @ np.diag(D) @ vt[:nsig,:] @ dy.transpose() @ rinv
                    rK = np.dot(dxf, vt.transpose()) * D
                    rK = np.dot(np.dot(rK, (u/s).transpose()), rmat)
                    dxa = dxf_orig - rK @ dy_orig
                else:
                    scalefact = np.sqrt(float(nmem2-1)/float(nmem-1))*self.l_sqrt[:,0]
                    dxa = dxf @ T[:,:nmem] / scalefact[:, None]
                trace = np.sum(np.diag(dxa @ dxa.T / (nmem-1)))
                logger.info("standard deviation ratio = {}".format(np.sqrt(ptrace / trace)))
                if np.sqrt(ptrace / trace) > 1.0:
                    dxa *= np.sqrt(ptrace / trace)
            if self.iinf == 0:
                logger.info("==multiplicative inflation==, alpha={}".format(self.infl_parm))
                dxa *= self.infl_parm
            if self.iinf == 1:
                logger.info("==additive inflation==, alpha={}".format(self.infl_parm))
                dxa += np.random.randn(dxa.shape[0], dxa.shape[1])*self.infl_parm
            if self.iinf == 3:
                pa = dxa @ dxa.T / (nmem-1)
                stdv_a = np.sqrt(np.diag(pa))
                logger.info("==RTPS, alpha={}".format(self.infl_parm))
                beta = ((1.0 - self.infl_parm)*stdv_a + self.infl_parm*stdv_f)/stdv_a
                logger.info(f"beta={beta}")
                dxa = dxa * beta[:, None]
            if save_dh:
                np.save("{}_dxa_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxa)
            xa = dxa + xa_[:,None]
            
        elif self.da=="po":
            err = self.rs.standard_normal(size=(y.size,nmem))
            err = err - err.mean(axis=1)[:, None]
            err_var = np.sum(err**2, axis=1)/(nmem-1)
            err = np.sqrt(np.diag(R))[:, None] * err / np.sqrt(err_var.mean())
            ## original
            #K1 = dxf @ dy.T / (nmem2-1)
            #K2 = dy @ dy.T / (nmem2-1) + R
            #eigK, vK = la.eigh(K2)
            #eigK = eigK[::-1]
            #vK = vK[:,::-1]
            #neig = eigK[eigK>1.e-10].size
            #logger.info(f"neig={neig}")
            #logger.debug("eigenvalues of K2")
            #logger.debug(eigK)
            #K2inv = vK[:, :neig] @ np.diag(1.0/eigK[:neig]) @ vK[:,:neig].transpose()
            ##K2inv = la.inv(K2)
            ##K2inv = cho_solve(cho_factor(K2), np.eye(y.size))
            #K = K1 @ K2inv
            # like ETKF
            A = (nmem2-1)*np.eye(nmem2) + dy.T @ rinv @ dy
            lam, v = la.eigh(A)
            #logger.info("eigen values={}".format(lam))
            Dinv = np.diag(1.0/lam)
            TT = v @ Dinv @ v.T
            K = dxf @ TT @ dy.T @ rinv
            #K = dxf @ dy.T @ la.inv(dy @ dy.T + (nmem-1)*R)
            if save_dh:
                np.save("{}_K_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K)
            if self.iloc == 0:
                logger.info("==K-localization== lsig={}".format(self.lsig))
                l_mat = self.k_loc(sigma=self.lsig, obsloc=yloc, xloc=xloc)
                K = K * l_mat
                if save_dh:
                    np.save("{}_Kloc_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), K)
                    np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), l_mat)
            xa_ = xf_ + K @ d
            dxa = dxf_orig - K @ (dy_orig + err)
            #if self.ltlm:
            #    xa = xf_orig + K @ (Y - JH @ xf_orig)
            #else:
            #    HX = self.obs.h_operator(yloc, xf_orig)
            #    xa = xf_orig + K @ (Y - HX)
            if save_dh:
                np.save("{}_dxa_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxa)
            xa = dxa + xa_[:, None]

        elif self.da=="srf":
            xa = xf_orig
            x0 = xf
            xa_ = np.mean(x0, axis=1)
            dx0 = dxf
            dxa = dxf_orig
            if self.iloc == 0:
                logger.info("==K-localization== lsig={}".format(self.lsig))
                l_mat = self.k_loc(sigma=self.lsig, obsloc=yloc, xloc=xloc)
                if save_dh:
                    np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), l_mat)
            for i, obloc, ob in zip(np.arange(y.size), yloc, y):
                logger.debug(f"{i}, {obloc}, {ob}")
                logger.debug(f"{np.atleast_1d(obloc)}")
                logger.debug(f"xa.shape={x0.shape}")
                logger.debug(f"xa_.shape={xa_.shape}")
                dyi = dy[i,].reshape(1,-1)
                di = d[i]
                logger.debug(f"dyi.shape={dyi.shape}")
                d1 = dyi @ dyi.T / (nmem2-1) + R[i,i] 
                k1 = dxf @ dyi.T / (nmem2-1) /d1
                logger.debug(f"k1={np.squeeze(k1)}")
                if self.iloc == 0: # K-localization
                    k1 = k1 * l_mat[:,i].reshape(k1.shape) # change later
                k1_ = k1 * np.sqrt(d1) / (np.sqrt(d1) + np.sqrt(R[i,i])) # change later
                xa_ = xa_.reshape(k1.shape) + k1 * di
                dxa = dxa - k1_@ dyi
                #x0 = xa_ + dx0
                if (self.iloc == 1 or self.iloc == 2):
                    if not self.ss:
                        dxa = dxa - k1_@ dyi_orig
                        xa = xa_ + dxa
            xa = xa_ + dxa
            if (self.iloc == 1 or self.iloc == 2):
                if self.ss:
                    if save_dh:
                        np.save("{}_dxaorig_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxa)
                    logger.info("random sampling")
                    ptrace = np.sum(np.diag(dx0 @ dx0.T / (nmem2-1)))
                    rvec = self.rs.standard_normal(size=(nmem2, nmem))
                    rvec_mean = np.mean(rvec, axis=0)
                    rvec = rvec - rvec_mean[None, :]
                    rvec_stdv = np.sqrt((rvec**2).sum(axis=0)/(nmem2-1))
                    rvec = rvec / rvec_stdv[None, :]
                    logger.debug("rvec={}".format(rvec[:, 0]))
                    dxa = dx0 @ rvec / np.sqrt(nmem2-1)
                    dxa = dxa - dxa.mean(axis=1)[:, None]
                    logger.debug(f"dxa.mean={dxa.mean(axis=1)}")
                    trace = np.sum(np.diag(dxa @ dxa.T / (nmem-1)))
                    logger.info("standard deviation ratio = {}".format(np.sqrt(ptrace / trace)))
                    #if np.sqrt(ptrace / trace) > 1.0:
                    #    dxa *= np.sqrt(ptrace / trace)
                    xa = xa_ + dxa
                    logger.debug(f"xa_ - xa.mean={np.squeeze(xa_) - xa.mean(axis=1)}")
            #else:
            #    xa = x0
            logger.debug(f"xa.mean={xa.mean(axis=1)}")
            if save_dh:
                np.save("{}_dxa_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxa)
            xa_ = np.squeeze(xa_)
        
        elif self.da=="eakf":
            # local least-square formulation (Anderson 2003, MWR)
            xa = xf_orig
            xa_ = np.mean(xf, axis=1)
            x0 = np.zeros_like(xa_)
            dx0 = np.zeros_like(dxf)
            dxa = dxf_orig
            if self.iloc == 0:
                logger.info("==K-localization== lsig={}".format(self.lsig))
                l_mat = self.k_loc(sigma=self.lsig, obsloc=yloc, xloc=xloc)
                if save_dh:
                    np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.da), l_mat)
            for i, obloc, ob in zip(np.arange(y.size), yloc, y):
                logger.debug(f"{i}, {obloc}, {ob}")
                logger.debug(f"{np.atleast_1d(obloc)}")
                logger.debug(f"xa.shape={x0.shape}")
                logger.debug(f"xa_.shape={xa_.shape}")
                dyi = dy[i,].reshape(1,-1)
                di = d[i]
                dzf = np.vstack([dxf,dyi])
                logger.debug(f"dzf.shape={dzf.shape}")
                s = dzf @ dyi.T / (nmem2-1)
                s = s.ravel()
                if self.iloc == 0: # K-localization
                    s[:-1] = s[:-1] * l_mat[:,i].ravel()
                d1 = s[-1] + R[i,i]
                alpha = np.sqrt(R[i,i] / d1)
                dz_ = s / d1 * di
                s_ = (alpha - 1.0) / s[-1] * s
                x0 = x0 + dz_[:-1]
                dx0 = dx0 + np.outer(s_[:-1], dyi)
            xa_ = xa_ + x0
            dxa = dxa + dx0
            xa = xa_[:,None] + dxa
            if (self.iloc == 1 or self.iloc == 2):
                #if self.ss:
                if save_dh:
                    np.save("{}_dxaorig_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxa)
                logger.info("random sampling")
                ptrace = np.sum(np.diag(dxa @ dxa.T / (nmem2-1)))
                rvec = self.rs.standard_normal(size=(nmem2, nmem))
                rvec_mean = np.mean(rvec, axis=0)
                rvec = rvec - rvec_mean[None, :]
                rvec_stdv = np.sqrt((rvec**2).sum(axis=0)/(nmem2-1))
                rvec = rvec / rvec_stdv[None, :]
                logger.debug("rvec={}".format(rvec[:, 0]))
                dxa = dxa @ rvec / np.sqrt(nmem2-1)
                dxa = dxa - dxa.mean(axis=1)[:, None]
                logger.debug(f"dxa.mean={dxa.mean(axis=1)}")
                trace = np.sum(np.diag(dxa @ dxa.T / (nmem-1)))
                logger.info("standard deviation ratio = {}".format(np.sqrt(ptrace / trace)))
                #if np.sqrt(ptrace / trace) > 1.0:
                #    dxa *= np.sqrt(ptrace / trace)
                xa = xa_ + dxa
                logger.debug(f"xa_ - xa.mean={np.squeeze(xa_) - xa.mean(axis=1)}")
            logger.debug(f"xa.mean={xa.mean(axis=1)}")
            if save_dh:
                np.save("{}_dxa_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxa)

        elif self.da=="letkf":
            sigma = self.lsig
            xa = np.zeros_like(xf)
            xa_ = np.zeros_like(xf_)
            dxa = np.zeros_like(dxf)
            rho = 1.0
            if self.iinf == -1:
                logger.info("==pre-multiplicative inflation==, alpha={}".format(self.infl_parm))
                rho = 1.0 / self.infl_parm
            if self.iloc==0:
                logger.info("==R-localization==, lsig={}".format(self.lsig))
            wlist = []
            Wlist = []
            for i in range(self.ndim):
                far, Rwf_loc = self.r_loc(sigma, yloc, float(i))
                logger.debug("analysis grid={} number of assimilated obs.={}".format(i, (y.size - len(far))))
                if self.iloc==0:
                    wgtmat = np.diag(np.sqrt(Rwf_loc))
                    di = wgtmat @ d_
                    dyi = wgtmat @ dy_
                else:
                    di = d_.copy()
                    dyi = dy_.copy()
                di = np.delete(di,far)
                dyi = np.delete(dyi,far,axis=0)
                if self.iinf == -2:
                    self.infl_parm = self.infladap(self.infl_parm_pre[i], di, dyi)
                    logger.debug("==adaptive pre-multiplicative inflation==, alpha={:.2f}".format(self.infl_parm))
                    rho = 1.0 / self.infl_parm
                u, ga, vt = la.svd(dyi,full_matrices=False)
                v = vt.transpose()
                ga2 = ga*ga
                lam = 1.0 / np.sqrt(ga2 + np.full(ga.size, rho))
                if self.iinf>=4:
                    logger.debug("lam = {}".format(lam))
                    laminf = self.inflfunc(lam,alpha1=self.infl_parm)
                    logger.debug("lam inf = {}".format(laminf))
                else:
                    laminf = lam.copy()
                D_inv = np.diag(lam*lam)
                pa_ = v @ D_inv @ v.T
                wk = pa_ @ dyi.T @ di / np.sqrt(nmem-1)
                logger.debug(f"wk={wk.shape}")
                wlist.append(wk)
                xa_[i] = xf_[i] + dxf[i] @ wk
                sqrtPa = v @ np.sqrt(np.diag(laminf)) @ v.T #* np.sqrt(nmem-1)
                Wlist.append(sqrtPa)
                for ivar in range(self.nvars):
                    xa_[self.ndim*ivar+i] = xf_[self.ndim*ivar+i] + dxf[self.ndim*ivar+i] @ wk
                    dxa[self.ndim*ivar+i] = dxf[self.ndim*ivar+i] @ sqrtPa
                    xa[self.ndim*ivar+i] = np.full(nmem,xa_[self.ndim*ivar+i]) + dxa[self.ndim*ivar+i]
            if save_w:
                logger.debug(f"wlist={np.array(wlist).shape}")
                logger.debug(f"Wlist={np.array(Wlist).shape}")
                np.save("wa_{}_{}_cycle{}.npy".format(self.op, self.da, icycle), np.array(wlist))
                np.save("Wmat_{}_{}_cycle{}.npy".format(self.op, self.da, icycle), np.array(Wlist))
        pa = dxa@dxa.T/(nmem-1)
        spa = dxa / np.sqrt(nmem-1)
        #if save_dh:
        if save_w:
            ua = np.zeros((xa_.size,nmem+1))
            ua[:,0] = xa_
            ua[:,1:] = xa
            np.save("{}_ua_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), ua)
            np.save("{}_pa_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), pa)

        if self.ltlm:
            dh = self.obs.dh_operator(yloc, xa_) @ dxf / np.sqrt(nmem2-1)
        else:
            x1 = xa_[:, None] + dxf / np.sqrt(nmem2-1)
            dh = self.obs.h_operator(yloc, x1) - np.mean(self.obs.h_operator(yloc, x1), axis=1)[:, None]
        zmat = rmat @ dh
        d = y - np.mean(self.obs.h_operator(yloc, xa))
        self.innv, self.chi2 = chi2_test(zmat, d)
        self.ds = self.dfs(dy,nmem2)
        logger.info("dfs={}".format(self.ds))
        
        #u = np.zeros_like(xb)
        #u = xa[:,:]
        if evalout:
            infl_mat = np.dot(zmat,zmat.T)
            evalb, _ = la.eigh(infl_mat)
            self.eval = evalb[::-1] / (1.0 + evalb[::-1])
        return xa, pa #, spa, innv, chi2, ds

    def b_loc(self, sigma, nx):
        if sigma < 0.0:
            loc_scale = 1.0
        else:
            loc_scale = sigma
        dist = np.zeros((nx,nx))
        l_mat = np.ones((self.nvars*nx,self.nvars*nx))
        # distance threshold
        dist0 = loc_scale * np.sqrt(10.0/3.0) * 2.0
        logger.debug(dist0)
        #for j in range(nx):
        #    for i in range(ny):
        #        dist[j,i] = min(abs(j-i),nx-abs(j-i))
        for i in range(nx):
            dist[:, i] = self.calc_dist(float(i))
        l_tmp = np.exp(-0.5*(dist/loc_scale)**2)
        logger.debug(dist[dist>dist0])
        l_tmp[dist>dist0] = 0
        for ivar in range(self.nvars):
            l_mat[ivar*nx:(ivar+1)*nx,ivar*nx:(ivar+1)*nx] = l_tmp[:,:]
        
        lam, v = la.eigh(l_mat)
        lam = lam[::-1]
        lam[lam < 1.e-10] = 1.e-10
        lamsum = np.sum(lam)
        v = v[:,::-1]
        nmode = 1
        #thres = 0.85
        frac = 0.0
        while frac < self.bthres:
            frac = np.sum(lam[:nmode]) / lamsum
            nmode += 1
        nmode = min(nmode, nx)
        logger.info("contribution rate = {}".format(np.sum(lam[:nmode])/np.sum(lam)))
        l_sqrt = v[:,:nmode] @ np.diag(np.sqrt(lam[:nmode]/frac))
        return l_mat, l_sqrt, nmode, np.sqrt(lam[:nmode])

    def k_loc(self, sigma, obsloc, xloc):
        if sigma < 0.0:
            loc_scale = 1.0
        else:
            loc_scale = sigma
        nx = xloc.size
        if obsloc.ndim == 1:
            nobs = obsloc.size
        else:
            nobs = obsloc.shape[0]
        dist = np.zeros((nx, nobs))
        # distance threshold
        dist0 = loc_scale * np.sqrt(10.0/3.0) * 2.0
        logger.debug(dist0)
        #for j in range(nx):
        #    for i in range(nobs):
        #        dist[j, i] = min(abs(obsloc[i]-j), nx-abs(obsloc[i]-j))
        for i in range(nobs):
            dist[:, i] = self.calc_dist(obsloc[i])
        l_mat = np.exp(-0.5*(dist/loc_scale)**2)
        logger.debug(dist[dist>dist0])
        l_mat[dist>dist0] = 0
        return l_mat        

    def r_loc(self, sigma, obsloc, xloc):
        if sigma < 0.0:
            loc_scale = 1.0e5
        else:
            loc_scale = sigma
        if obsloc.ndim == 1:
            nobs = obsloc.size
        else:
            nobs = obsloc.shape[0]
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

    def pfloc(self, dxf_orig, save_dh, icycle):
        nmem = dxf_orig.shape[1]
        #nmode = min(100, self.ndim)
        pf = dxf_orig @ dxf_orig.T / (nmem - 1)
        pf = pf * self.l_mat
        logger.info("lpf max={} min={}".format(np.max(pf),np.min(pf)))
        if save_dh:
            np.save("{}_lpf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), pf)
        lam, v = la.eigh(pf)
        lam = lam[::-1]
        lam[lam < 0.0] = 0.0
        lamsum = lam.sum()
        v = v[:,::-1]
        if save_dh:
            np.save("{}_lpfeig_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), lam)
        logger.info("pf eigen value = {}".format(lam))
        nmode = 1
        #thres = 0.99
        frac = 0.0
        while frac < self.bthres:
            frac = np.sum(lam[:nmode]) / lamsum
            nmode += 1
        nmode = min(nmode, self.ndim)
        logger.info(f"== eigen value decomposition of Pf, nmode={nmode} ==")
        pf = v[:,:nmode] @ np.diag(lam[:nmode]) @ v[:,:nmode].T
        dxf = v[:,:nmode] @ np.diag(np.sqrt(lam[:nmode])) * np.sqrt(nmode-1)
        logger.info("pf - spf@spf.T={}".format(np.mean(pf - dxf@dxf.T/(nmode-1))))
        if save_dh:
            np.save("{}_lpfr_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), pf)
            np.save("{}_lspf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxf)
        return dxf

    def pfmod(self, dxf_orig, save_dh, icycle):
        nmem = dxf_orig.shape[1]
        logger.info(f"== modulated ensemble, nmode={self.nmode} ==")
        
        dxf = np.empty((self.ndim, nmem*self.nmode), dxf_orig.dtype)
        for l in range(self.nmode):
            for k in range(nmem):
                m = l*nmem + k
                dxf[:, m] = self.l_sqrt[:, l]*dxf_orig[:, k]
        dxf = dxf * np.sqrt((nmem*self.nmode-1)/(nmem-1))
        if save_dh:
            np.save("{}_lspf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), dxf)
            fullpf = dxf @ dxf.T / (nmem*self.nmode - 1)
            np.save("{}_lpf_{}_{}_cycle{}.npy".format(self.model, self.op, self.da, icycle), fullpf)
        return dxf

    def dfs(self, dy, nmem):
        # Zupanski, D. et al., (2007) Applications of information theory in ensemble data assimilation
        # Eq. (10)
        zmat = dy / self.sig / np.sqrt(nmem-1)
        u, s, vt = la.svd(zmat)
        ds = np.sum(s**2/(1.0+s**2))
        return ds
