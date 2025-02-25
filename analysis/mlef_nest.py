import logging
from logging import getLogger
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import copy
from .chi_test import Chi
from .minimize import Minimize
from .trunc1d import Trunc1d
from scipy.interpolate import interp1d

zetak = []
alphak = []
fileConfig("./logging_config.ini")
logger = getLogger('anl')
        
class Mlef_nest():

    def __init__(self, state_size, nmem, obs, ix_gm, ix_lam,
        pt="mlef_nest", ntrunc=None, ftrunc=None, cyclic=False, crosscov=False,
        nvars=1,ndims=1,
        linf=False, infl_parm=1.0, infl_parm_lrg=1.0,
        iloc=None, lsig=-1.0, ss=False, getkf=False,
        l_mat=None, l_sqrt=None,
        calc_dist=None, calc_dist1=None, 
        ltlm=False, incremental=True, model="model"):
        # necessary parameters
        self.ndim = state_size # state size
        self.nmem = nmem # ensemble size
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        self.ix_gm = ix_gm # GM grid
        self.ix_lam = ix_lam # LAM grid
        i0 = np.argmin(np.abs(self.ix_gm-self.ix_lam[0]))
        if self.ix_gm[i0]<self.ix_lam[0]: i0+=1
        i1 = np.argmin(np.abs(self.ix_gm-self.ix_lam[-1]))
        if self.ix_gm[i1]>self.ix_lam[-1]: i1-=1
        self.i0 = i0 # GM first index within LAM domain
        self.i1 = i1 # GM last index within LAM domain
        self.nv = self.i1 - self.i0 + 1
        self.ntrunc = ntrunc # truncation number for GM
        self.ftrunc = ftrunc # truncation wavenumber for GM
        self.trunc_operator = Trunc1d(self.ix_lam,ntrunc=self.ntrunc,ftrunc=self.ftrunc,cyclic=False,ttype='s',nghost=0)
        self.nv = min(self.nv,self.trunc_operator.ix_trunc.size)
        # optional parameters
        self.pt = pt # DA type 
        # for 2 or more variables
        self.nvars = nvars
        # for 2 or more dimensional data
        self.ndims = ndims
        # inflation
        self.linf = linf # True->Apply inflation False->Not apply
        self.infl_parm = infl_parm # inflation parameter
        self.infl_parm_lrg = infl_parm_lrg # inflation parameter for large-scale error cov.
        # localization (TODO: implementation)
        self.iloc = iloc # iloc = None->No localization
                         #      <=0   ->R-localization (in lmlef.py)
                         #      = 1   ->Eigen value decomposition of localized Pf
                         #      = 2   ->Modulated ensemble
        self.lsig = lsig # localization parameter
        self.ss = ss     # ensemble reduction method : True->Use stochastic sampling
        self.getkf = getkf # ensemble reduction method : True->Use reduced gain (Bishop et al. 2017)
        if calc_dist is None:
            self.calc_dist = self._calc_dist
        else:
            self.calc_dist = calc_dist # distance calculation routine
        if calc_dist1 is None:
            self.calc_dist1 = self._calc_dist1
        else:
            self.calc_dist1 = calc_dist1 # distance calculation routine
        self.rs = np.random.default_rng() # random generator
        # tangent linear
        self.ltlm = ltlm # True->Use tangent linear approximation False->Not use
        # incremental form
        self.incremental = incremental
        if self.op == "linear":
            self.incremental = True
        # cross-covariance
        self.crosscov = crosscov
        # forecast model name
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"ndim={self.ndim} nmem={self.nmem}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig} infl_parm={self.infl_parm} lsig={self.lsig} infl_parm_lrg={self.infl_parm_lrg}")
        logger.info(f"linf={self.linf} iloc={self.iloc} ltlm={self.ltlm} incremental={self.incremental}")
        logger.info(f"crosscov={self.crosscov}")
        if self.iloc is not None:
          #if self.iloc <= 0:
          #  from .lmlef import Lmlef
          #  self.lmlef = Lmlef(self.nmem,self.obs,
          #  nvars=self.nvars,ndims=self.ndims,
          #  linf=self.linf,infl_parm=self.infl_parm,
          #  iloc=self.iloc,lsig=self.lsig,calc_dist1=self.calc_dist1,
          #  ltlm=self.ltlm,incremental=self.incremental,model=self.model)
          #else:
            if l_mat is None or l_sqrt is None:
                self.l_mat, self.l_sqrt, self.nmode, self.enswts \
                = self.loc_mat(self.lsig, self.ndim, self.ndim)
            else:
                self.l_mat = l_mat
                self.l_sqrt = l_sqrt
                self.nmode = l_sqrt.shape[1]
            np.save("{}_rho_{}_{}.npy".format(self.model, self.op, self.pt), self.l_mat)

    def _calc_dist(self, i):
        dist = np.zeros(self.ndim)
        for j in range(self.ndim):
            dist[j] = min(abs(j-i),self.ndim-abs(j-i))
        return dist
    
    def _calc_dist1(self, i, j):
        return min(abs(j-i),self.ndim-abs(j-i))

    def calc_pf(self, xf, **kwargs):
        spf = xf[:, 1:] - xf[:, 0].reshape(-1,1)
        pf = spf @ spf.transpose()
        logger.debug(f"pf max{np.max(pf)} min{np.min(pf)}")
        return pf

    def precondition(self,zmat,qmat,save=False,icycle=0):
        #u, s, vt = la.svd(zmat)
        #v = vt.transpose()
        #is2r = 1 / (1 + s**2)
        nk = zmat.shape[1]
        rho = 1.0
        rho_lrg = 1.0
        #if self.linf:
        #    logger.info("==inflation==, alpha={}".format(self.infl_parm))
        #    rho = 1.0 / self.infl_parm
        #    logger.info("==inflation(lrg)==, alpha={}".format(self.infl_parm_lrg))
        #    rho_lrg = 1.0 / self.infl_parm_lrg
        if not self.crosscov:
            ## Q = (JH_1P^B)^\dag JH_2P^b
            ## Hess = (I + Z^T Z + Q^T Q)
            hessmi = zmat.transpose() @ zmat + rho_lrg * qmat.transpose() @ qmat
            if save:
                hess = rho*np.eye(hessmi.shape[0]) + hessmi
                np.save("{}_hess_{}_{}_cycle{}.npy".format(self.model,self.op,self.pt,icycle), hess)
            lam, c = la.eigh(hessmi)
            D = np.diag(1.0/np.sqrt(lam + np.full(lam.size,rho)))
            ct = c.transpose()
            tmat = c @ D @ ct
            heinv = tmat @ tmat.T
        else:
            ## Q = (P^b\\ JH_1P^B)^\dag(P^b\\ JH_2P^b)
            ## Hess = (Q^T Q + Z^T Z)
            zmatc = np.vstack((np.sqrt(rho_lrg)*qmat,zmat))
            e, s, ct = la.svd(zmatc,full_matrices=False)
            if save:
                hess = zmatc.transpose() @ zmatc
                np.save("{}_hess_{}_{}_cycle{}.npy".format(self.model,self.op,self.pt,icycle), hess)
            lam = s*s
            ndof = np.sum((lam-1.0)>-1.0e-5)
            #ndof = np.sum(lam>1.0e-10)
            logger.info(f"precondition: ndof={ndof}")
            if ndof < s.size:
                e = e[:,:ndof]
                s = s[:ndof]
                ct = ct[:ndof,:]
            c = ct.transpose()
            D = np.diag(1.0/s)
            if zmatc.shape[0]>zmatc.shape[1]:
                tmat = c @ D @ ct
            else:
                tmat = np.eye(c.shape[0]) - c @ (np.diag(np.ones(D.shape[0]))-D) @ ct
            heinv = tmat @ tmat.transpose()
        logger.debug("precondition: tmat={}".format(tmat))
        logger.debug("precondition: heinv={}".format(heinv))
        logger.info("precondition: eigenvalue ={}".format(lam))
        #print(f"rank(zmat)={lam[lam>1.0e-10].shape[0]}")
        if save:
            np.save("{}_tmat_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), tmat)
            np.save("{}_heinv_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), heinv)
        return tmat, heinv

    def callback(self, xk, alpha=None):
        global zetak, alphak
        logger.debug("xk={}".format(xk))
        zetak.append(copy.copy(xk))
        if alpha is not None:
            alphak.append(alpha)

    def calc_j(self, zeta, *args, return_each=False):
        if not self.crosscov:
            ## Q = (JH_1P^B)^\dag JH_2P^b
            ## dk = (JH_1P^B)^\dag (H_1(x^B) - H_2(x^b))
            if not self.incremental:
                xc, pf, y, yloc, tmat, gmat, dk, qmat, heinv, rinv = args
                x = xc + gmat @ zeta
                w = tmat @ zeta
                ob = y - self.obs.h_operator(yloc, x)
                jb = 0.5 * zeta.transpose() @ heinv @ zeta
                jo = 0.5 * ob.transpose() @ rinv @ ob
                jk = 0.5 * (qmat@w - dk).transpose() @ (qmat@w - dk)
            #    j = 0.5 * (zeta.transpose() @ heinv @ zeta + ob.transpose() @ rinv @ ob)
            else:
            ## incremental form
                d, tmat, zmat, dk, qmat, heinv = args
                w = tmat @ zeta
                jb = 0.5 * zeta.transpose() @ heinv @ zeta 
                jo = 0.5 * (zmat@w - d).transpose() @ (zmat@w - d)
                jk = 0.5 * (qmat@w - dk).transpose() @ (qmat@w - dk)
                #j = 0.5 * (zeta.transpose() @ heinv @ zeta + (zmat@w - d).transpose() @ (zmat@w - d))
            if return_each:
                return jb,jo,jk
            else:
                logger.info(f"jb:{jb:.6e} jo:{jo:.6e} jk:{jk:.6e}")
                j = jb + jo + jk
            return j
        else:
            ## Q = (P^b\\ JH_1P^B)^\dag (P^b\\ JH_2P^b)
            ## dk = (P^b\\ JH_1P^B)^\dag (0\\ H_1(x^B) - H_2(x^b))
            if not self.incremental:
                xc, pf, y, yloc, tmat, gmat, dk, qmat, heinv, rinv = args
                x = xc + gmat @ zeta
                w = tmat @ zeta
                ob = y - self.obs.h_operator(yloc, x)
                jbv = 0.5 * (qmat@w - dk).transpose() @ (qmat@w - dk)
                jo  = 0.5 * ob.transpose() @ rinv @ ob
            else:
                d, tmat, zmat, dk, qmat, heinv = args
                w = tmat @ zeta
                jbv = 0.5 * (qmat@w - dk).transpose() @ (qmat@w - dk) 
                jo  = 0.5 * (zmat@w - d).transpose() @ (zmat@w - d)
            if return_each:
                return jbv, jo
            else:
                logger.info(f"jbv:{jbv:.6e} jo:{jo:.6e}")
                j = jbv + jo
                return j

    def calc_grad_j(self, zeta, *args):
        if not self.crosscov:
            ## Q = (JH_1P^B)^\dag JH_2P^b
            ## dk = (JH_1P^B)^\dag (H_1(x^B) - H_2(x^b))
            if not self.incremental:
                xc, pf, y, yloc, tmat, gmat, dk, qmat, heinv, rinv = args
                x = xc + gmat @ zeta
                w = tmat @ zeta
                hx = self.obs.h_operator(yloc, x)
                ob = y - hx
                if self.ltlm:
                    dh = self.obs.dh_operator(yloc, x) @ pf
                else:
                    dh = self.obs.h_operator(yloc, x[:, None] + pf) - hx[:, None]
                grad = heinv @ zeta \
                    - tmat @ dh.transpose() @ rinv @ ob \
                    + tmat @ qmat.transpose() @ (qmat@w - dk)
            else:
            ## incremental form
                d, tmat, zmat, dk, qmat, heinv = args
                w = tmat @ zeta
                grad = heinv @ zeta \
                    + tmat @ zmat.transpose() @ (zmat@w - d) \
                    + tmat @ qmat.transpose() @ (qmat@w - dk)
        else:
            ## Q = (P^b\\ JH_1P^B)^\dag (P^b\\ JH_2P^b)
            ## dk = (P^b\\ JH_1P^B)^\dag (0\\ H_1(x^B) - H_2(x^b))
            if not self.incremental:
                xc, pf, y, yloc, tmat, gmat, dk, qmat, heinv, rinv = args
                x = xc + gmat @ zeta
                w = tmat @ zeta
                hx = self.obs.h_operator(yloc, x)
                ob = y - hx
                if self.ltlm:
                    dh = self.obs.dh_operator(yloc, x) @ pf
                else:
                    dh = self.obs.h_operator(yloc, x[:, None] + pf) - hx[:, None]
                grad = tmat @ qmat.transpose() @ (qmat@w - dk) \
                    - tmat @ dh.transpose() @ rinb @ ob
            else:
            ## incremental form
                d, tmat, zmat, dk, qmat, heinv = args
                w = tmat @ zeta
                grad = tmat @ qmat.transpose() @ (qmat@w - dk) \
                    + tmat @ zmat.transpose() @ (zmat@w - d)
        logger.info(f"|dj|:{np.sqrt(np.dot(grad,grad)):.6e}")
        return grad 

    def calc_hess(self, zeta, *args):
        if not self.crosscov:
            ## Q = (JH_1P^B)^\dag JH_2P^b
            ## dk = (JH_1P^B)^\dag (H_1(x^B) - H_2(x^b))
            if not self.incremental:
                xc, pf, y, yloc, tmat, gmat, dk, qmat, heinv, rinv = args
                x = xc + gmat @ zeta
                w = tmat @ zeta
                if self.ltlm:
                    dh = self.obs.dh_operator(yloc, x) @ pf
                else:
                    dh = self.obs.h_operator(yloc, x[:, None] + pf) - self.obs.h_operator(yloc, x)[:, None]
                hess = tmat @ (np.eye(zeta.size) \
                    + dh.transpose() @ rinv @ dh \
                    + qmat.transpose() @ qmat) @ tmat
            else:
            ## incremental form
                d, tmat, zmat, dk, qmat, heinv = args
                hess = tmat @ (np.eye(zeta.size) \
                    + zmat.transpose() @ zmat \
                    + qmat.transpose() @ qmat) @ tmat
        else:
            ## Q = (P^b\\ JH_1P^B)^\dag (P^b\\ JH_2P^b)
            ## dk = (P^b\\ JH_1P^B)^\dag (0\\ H_1(x^B) - H_2(x^b))
            if not self.incremental:
                xc, pf, y, yloc, tmat, gmat, dk, qmat, heinv, rinv = args
                x = xc + gmat @ zeta
                w = tmat @ zeta
                if self.ltlm:
                    dh = self.obs.dh_operator(yloc, x) @ pf
                else:
                    dh = self.obs.h_operator(yloc, x[:, None] + pf) - self.obs.h_operator(yloc, x)[:, None]
                hess = tmat @ (qmat.transpose() @ qmat \
                    + dh.transpose() @ rinv @ dh) @ tmat
            else:
            ## incremental form
                d, tmat, zmat, dk, qmat, heinv = args
                hess = tmat @ (qmat.transpose() @ qmat \
                    + zmat.transpose() @ zmat) @ tmat
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

    def pfloc(self, sqrtpf, l_mat, save_dh, icycle,\
        op="linear",pt="mlefbe",model="model"):
        #nmode = min(100, self.ndim)
        #logger.info(f"== Pf localization, nmode={nmode} ==")
        ndim,nmem = sqrtpf.shape
        pf = sqrtpf @ sqrtpf.T
        pf = pf * l_mat
        #if save_dh:
        #    np.save("{}_lpf_{}_{}_cycle{}.npy".format(model, op, pt, icycle), pf)
        lam, v = la.eigh(pf)
        lam = lam[::-1]
        lam[lam < 0.0] = 0.0
        lamsum = lam.sum()
        v = v[:,::-1]
        if save_dh:
            np.save("{}_lpfeig_{}_{}_cycle{}.npy".format(model, op, pt, icycle), lam)
        logger.info("pf eigen value = {}".format(lam))
        nmode = 0
        thres = 0.99
        frac = 0.0
        while frac < thres:
            nmode += 1
            frac = np.sum(lam[:nmode]) / lamsum
        nmode = min(nmode, self.ndim)
        logger.info(f"== eigen value decomposition of Pf, nmode={nmode} ==")
        pf = v[:,:nmode] @ np.diag(lam[:nmode]) @ v[:,:nmode].T
        spf = v[:,:nmode] @ np.diag(np.sqrt(lam[:nmode])) 
        logger.info("pf - spf@spf.T={}".format(np.mean(pf - spf@spf.T)))
        if save_dh:
            np.save("{}_lpf_{}_{}_cycle{}.npy".format(model, op, pt, icycle), pf)
            np.save("{}_lspf_{}_{}_cycle{}.npy".format(model, op, pt, icycle), spf)
        return spf, np.sqrt(lam[:nmode])
    
    def pfmod(self, sqrtpf, l_sqrt, save_dh, icycle,\
        op="linear",pt="mlefbm",model="model"):
        ndim,nmem = sqrtpf.shape
        nmode= l_sqrt.shape[1]
        logger.info(f"== modulated ensemble, nmode={nmode} ==")
        
        spf = np.empty((ndim, nmem*nmode), sqrtpf.dtype)
        for l in range(nmode):
            for k in range(nmem):
                m = l*nmem + k
                spf[:, m] = l_sqrt[:, l]*sqrtpf[:, k]
        if save_dh:
            np.save("{}_lspf_{}_{}_cycle{}.npy".format(model, op, pt, icycle), spf)
            fullpf = spf @ spf.T
            np.save("{}_lpf_{}_{}_cycle{}.npy".format(model, op, pt, icycle), fullpf)
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
        if logger.isEnabledFor(logging.DEBUG):
            for i in range(l_mat.shape[1]):
                logger.debug(f"{i:02d} lmat {l_mat[:,i]}")

        lam, v = la.eigh(l_mat)
        lam = lam[::-1]
        logger.debug(f"lam {lam}")
        lam[lam < 1.e-10] = 0.0
        lamsum = np.sum(lam)
        logger.debug(f"lamsum {lamsum}")
        v = v[:,::-1]
        nmode = 0
        thres = 0.99
        frac = 0.0
        while frac < thres:
            nmode += 1
            frac = np.sum(lam[:nmode]) / lamsum
        nmode = min(nmode, nx, ny)
        logger.info("# mode {} : contribution rate = {}".format(\
        nmode,np.sum(lam[:nmode])/np.sum(lam)))
        l_sqrt = v[:,:nmode] @ np.diag(np.sqrt(lam[:nmode]))
        if logger.isEnabledFor(logging.DEBUG):
            for i in range(l_sqrt.shape[1]):
                logger.debug(f"{i:02d} lsq {l_sqrt[:,i]}")
            l_tmp = l_sqrt @ l_sqrt.transpose()
            for i in range(l_tmp.shape[1]):
                logger.debug(f"{i:02d} lmat {l_tmp[:,i]}")
        return l_mat, l_sqrt, nmode, np.sqrt(lam[:nmode])

    def __call__(self, xb, pb, y, yloc, xg, r=None, rmat=None, rinv=None,
        method="CG", cgtype=1,
        gtol=1e-6, maxiter=None, restart=False, maxrest=20, update_ensemble=False,
        disp=False, save_hist=False, save_dh=False, icycle=0, evalout=False):
        if self.iloc is not None and self.iloc <= 0:
        #    return self.lmlef(xb,pb,y,yloc,r=r,rmat=rmat,rinv=rinv,
        #    method=method,cgtype=cgtype,gtol=gtol,maxiter=maxiter,restart=restart,maxrest=maxrest,update_ensemble=update_ensemble,
        #    disp=disp,save_hist=save_hist,save_dh=save_dh,icycle=icycle)
            print("not implemented yet")
            exit()
        else:
            global zetak, alphak
            zetak = []
            alphak = []
            if (r is None) or (rmat is None) or (rinv is None):
                logger.info("set R")
                r, rmat, rinv = self.obs.set_r(yloc)
            else:
                logger.info("use input R")
            logger.debug("r={}".format(r))
            logger.debug("r^[-1/2]={}".format(rmat))
            xf = xb[:, 1:]
            xc = xb[:, 0]
            nmem = xf.shape[1]
            chi2_test = Chi(y.size, nmem, rmat)
            pf = xf - xc[:, None]
            if self.linf:
                logger.info("==inflation==, alpha={}".format(self.infl_parm))
                pf *= np.sqrt(self.infl_parm)
            fpf = pf @ pf.T
            if save_dh:
                np.save("{}_uf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), xb)
                np.save("{}_pf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), fpf)
                np.save("{}_spf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), pf)
            if self.iloc is not None:
                logger.info("==localization==, lsig={}".format(self.lsig))
                pf_orig = pf.copy()
                if self.iloc == 1:
                    pf, wts = self.pfloc(pf_orig, self.l_mat, save_dh, icycle,\
                        op=self.op,pt=self.pt,model=self.model)
                elif self.iloc == 2:
                    pf = self.pfmod(pf_orig, self.l_sqrt, save_dh, icycle,\
                        op=self.op,pt=self.pt,model=self.model)
                logger.info("pf.shape={}".format(pf.shape))
                xf = xc[:, None] + pf
            #logger.debug("norm(pf)={}".format(la.norm(pf)))
            #logger.debug("r={}".format(np.diag(r)))
            ## global ensemble
            xgc = xg[:, 0]
            xgf = xg[:, 1:]
            pfg = xgf - xgc[:, None]
            if self.linf:
                logger.info("==inflation(lrg)==, alpha={}".format(self.infl_parm_lrg))
                pfg *= np.sqrt(self.infl_parm_lrg)
            x_gm2lam = interp1d(self.ix_gm,xgc)
            xens_gm2lam = interp1d(self.ix_gm,pfg,axis=0)
            dk = self.trunc_operator(x_gm2lam(self.ix_lam) - xc)
            zbmat = self.trunc_operator(pf)
            zvmat = self.trunc_operator(xens_gm2lam(self.ix_lam))
            logger.info(f"zv.shape={zvmat.shape}")
            logger.info(f"zb.shape={zbmat.shape}")
            if save_dh:
                np.save("{}_zbmat_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), zbmat)
                np.save("{}_dk_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), dk)
                np.save("{}_svmat_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), zvmat)
            if not self.crosscov:
                ## Q = (JH_1X^B)^\dag JH_2P^b
                ## dk = (JH_1X^B)^\dag (H_1(x^B) - H_2(x^b))
                u, s, vt = la.svd(zvmat)
                logger.info(f"s={s}")
                ndof = int(np.sum(s>1.0e-10))
                logger.info(f"ndof={ndof}")
                vsqrtinv = vt[:ndof,:].transpose() @ np.diag(1.0/s[:ndof]) @ u[:,:ndof].transpose()
                qmat = vsqrtinv @ zbmat
                dk = vsqrtinv @ dk
            else:
                ## Q = (P^b\\ JH_1P^B)^\dag (P^b\\ JH_2P^b)
                ## dk = (P^b\\ JH_1P^B)^\dag (0\\ H_1(x^B) - H_2(x^b))
                dxc1 = np.vstack((pf,zvmat))
                dxc2 = np.vstack((pf,zbmat))
                logger.info(f"dxc1.shape={dxc1.shape}")
                logger.info(f"dxc2.shape={dxc2.shape}")
                if save_dh:
                    np.save("{}_dxc1_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), dxc1)
                    np.save("{}_dxc2_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), dxc2)
                u, s, vt = la.svd(dxc1)
                logger.info(f"s={s}")
                ndof = int(np.sum(s>1.0e-10))
                logger.info(f"ndof={ndof}")
                vsqrtinv = vt[:ndof,:].transpose() @ np.diag(1.0/s[:ndof]) @ u[:,:ndof].transpose()
                qmat = vsqrtinv @ dxc2
                dk = vsqrtinv @ np.hstack((np.zeros(pf.shape[0]),dk))
            logger.info(f"qmat.shape={qmat.shape}")
            logger.info(f"dk.shape={dk.shape}")
            if save_dh:
                np.save("{}_qmat_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), qmat)
                np.save("{}_dk2_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), dk)
            ## observation
            if self.ltlm:
                logger.debug("dhdx={}".format(self.obs.dhdx(xc)))
                dh = self.obs.dh_operator(yloc,xc) @ pf
            else:
                dh = self.obs.h_operator(yloc,xc[:, None]+pf) - self.obs.h_operator(yloc,xc)[:, None]
            ob = y - self.obs.h_operator(yloc,xc)
            logger.debug("ob={}".format(ob))
            logger.debug("dh={}".format(dh))
            if save_dh:
                np.save("{}_dh_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), dh)
                np.save("{}_d_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), ob)
            logger.info("save_dh={}".format(save_dh))
            zmat = rmat @ dh
            d = rmat @ ob
            #logger.debug("cond(zmat)={}".format(la.cond(zmat)))
            tmat, heinv = self.precondition(zmat,qmat,save=save_dh,icycle=icycle)
            logger.debug("pf.shape={}".format(pf.shape))
            logger.debug("tmat.shape={}".format(tmat.shape))
            logger.debug("heinv.shape={}".format(heinv.shape))
            gmat = pf @ tmat
            logger.debug("gmat.shape={}".format(gmat.shape))
            x0 = np.zeros(pf.shape[1])
            x = x0.copy()
            if not self.incremental:
                args_j = (xc, pf, y, yloc, tmat, gmat, dk, qmat, heinv, rinv)
            else:
                args_j = (d, tmat, zmat, dk, qmat, heinv)
            iprint = np.zeros(2, dtype=np.int32)
            irest = 0 # restart counter
            flg = -1  # optimilation result flag
            options = {'iprint':iprint, 'method':method, 'cgtype':cgtype, \
                    'gtol':gtol, 'disp':disp, 'maxiter':maxiter, 'restart':restart}
            minimize = Minimize(x0.size, self.calc_j, jac=self.calc_grad_j, hess=self.calc_hess,
                            args=args_j, **options)
            logger.info("save_hist={}".format(save_hist))
            if restart:
                if save_hist:
                    jh = []
                    gh = []
                while irest < maxrest:
                    zetak = []
                    xold = x 
                    if save_hist:
                        x, flg = minimize(x0, callback=self.callback)
                        for i in range(len(zetak)):
                            #jh.append(self.calc_j(np.array(zetak[i]), *args_j))
                            # calculate jb and jo separately
                            if not self.crosscov:
                                jb,jo,jk = self.calc_j(np.array(zetak[i]), *args_j, return_each=True)
                                jh.append([jb,jo,jk])
                            else:
                                jbv, jo = self.calc_j(np.array(zetak[i]), *args_j, return_each=True)
                                jh.append([jbv,jo])
                            g = self.calc_grad_j(np.array(zetak[i]), *args_j)
                            gh.append(np.sqrt(g.transpose() @ g))
                    else:
                        x, flg = minimize(x0)
                    irest += 1
                    if flg == 0:
                        logger.info("Converged at {}th restart".format(irest))
                        break
                    xup = x - xold
                    if np.sqrt(np.dot(xup,xup)) < 1e-10:
                        logger.info("Stagnation at {}th restart : solution not updated"
                        .format(irest))
                        break
                    xa = xc + gmat @ x
                    xc = xa
                    if self.ltlm:
                        dh = self.obs.dh_operator(yloc, xc) @ pf
                    else:
                        dh = self.obs.h_operator(yloc, xc[:, None]+pf) - self.obs.h_operator(yloc, xc)[:, None]
                    zmat = rmat @ dh
                    tmat, heinv = self.precondition(zmat)
                    gmat = pf @ tmat
                    dk = self.trunc_operator(x_gm2lam(self.ix_lam) - xc)
                    dk = vsqrtinv @ dk
                    if update_ensemble:
                        pf = pf @ tmat
                    # update arguments
                    if not self.incremental:
                        args_j = (xc, pf, y, yloc, tmat, gmat, dk, qmat, heinv, rinv)
                    else:
                        d = rmat @ (y - self.obs.h_operator(yloc,xc))
                        args_j = (d, tmat, zmat, dk, qmat, heinv)
                    x0 = np.zeros(pf.shape[1])
                    # reload minimize class
                    minimize = Minimize(x0.size, self.calc_j, jac=self.calc_grad_j, hess=self.calc_hess,
                            args=args_j, iprint=iprint, method=method, cgtype=cgtype,
                            maxiter=maxiter, restart=restart)
                if save_hist:
                    logger.debug("zetak={} alpha={}".format(len(zetak), len(alphak)))
                    np.savetxt("{}_jh_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), jh)
                    np.savetxt("{}_gh_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), gh)
                    if len(alphak)>0: np.savetxt("{}_alpha_{}_{}_cycle{}.txt".format(self.model, self.op, self.pt, icycle), alphak)
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
                if save_hist:
                    x, flg = minimize(x0, callback=self.callback)
                    jh = np.zeros((len(zetak),3))
                    gh = np.zeros(len(zetak))
                    for i in range(len(zetak)):
                        #jh[i] = self.calc_j(np.array(zetak[i]), *args_j)
                        # calculate jb, jo and jk separately
                        if not self.crosscov:
                            jb, jo, jk = self.calc_j(np.array(zetak[i]), *args_j, return_each=True)
                            jh[i,0] = jb
                            jh[i,1] = jo
                            jh[i,2] = jk
                        else:
                            jbv, jo = self.calc_j(np.array(zetak[i]), *args_j, return_each=True)
                            jh[i,0] = jbv
                            jh[i,1] = jo
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
            #logger.debug("cond(zmat)={}".format(la.cond(zmat)))
            tmat, heinv = self.precondition(zmat, qmat)
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
                elif self.getkf:
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
            if evalout:
                infl_mat = np.dot(zmat,zmat.T)
                evalb, _ = la.eigh(infl_mat)
                eval = evalb[::-1] / (1.0 + evalb[::-1])
                return u, fpa, pa, innv, chi2, ds, eval
            else:
                return u, fpa, pa, innv, chi2, ds
