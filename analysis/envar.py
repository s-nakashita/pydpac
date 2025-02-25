import logging
from logging import getLogger
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import copy
from .chi_test import Chi
from .minimize import Minimize
from .infladap import infladap
from .inflfunc import inflfunc

zetak = []
alphak = []
fileConfig("./logging_config.ini")
logger = getLogger('anl')
        
class EnVAR():

    def __init__(self, state_size, nmem, obs, pt="envar",
        nvars=1,ndims=1,
        linf=False, iinf=None, infl_parm=1.0, 
        lloc=False, iloc=None, lsig=-1.0, ss=False, getkf=False,
        l_mat=None, l_sqrt=None,
        calc_dist=None, calc_dist1=None, 
        ltlm=False, incremental=True, model="model"):
        # essential parameters
        self.ndim = state_size # state size
        self.nmem = nmem # ensemble size
        self.obs = obs # observation operator
        self.op = obs.get_op() # observation type
        self.sig = obs.get_sig() # observation error standard deviation
        # optional parameters
        self.pt = pt # DA type 
        # for 2 or more variables
        self.nvars = nvars
        # for 2 or more dimensional data
        self.ndims = ndims
        # inflation
        self.linf = linf # inflation switch
        self.infltype = {-99:'No',-3:'adap.pre-mul.D21',-2:'adap.pre-mul.L09',-1:'fix.pre-mul',0:'post-mul',1:'add',2:'RTPP',3:'RTPS',4:'mul-lin'}
        self.iinf = iinf # iinf = None->No inflation
                         #      = -3  ->Adaptive pre-multiplicative inflation (Duc et al. 2021)
                         #      = -2  ->Adaptive pre-multiplicative inflation (Liu et al. 2009)
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
        paramtype = self.iinf - 4
        self.inflfunc = inflfunc("mult",paramtype=paramtype)
        # localization
        self.lloc = lloc # localization switch
        self.loctype = {None:'No',0:'R-loc',1:'EVD',2:'Modulation'}
        self.iloc = iloc # iloc = None->No localization
                         #      <=0   ->R-localization (TODO: implement)
                         #      = 1   ->Eigen value decomposition of localized Pf
                         #      = 2   ->Modulated ensemble
        self.lsig = lsig # localization parameter
        self.ss = ss     # ensemble reduction method : True->Use stochastic sampling
        self.getkf = getkf # ensemble reduction method : True->Use reduced gain (Bishop et al. 2017)
        if self.lloc:
            if self.iloc is None:
                self.iloc = 1
        else:
            self.iloc = None
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
        # forecast model name
        self.model = model
        logger.info(f"model : {self.model}")
        logger.info(f"ndim={self.ndim} nmem={self.nmem}")
        logger.info(f"pt={self.pt} op={self.op} sig={self.sig} infl_parm={self.infl_parm} lsig={self.lsig}")
        logger.info(f"inf={self.infltype[self.iinf]} loc={self.loctype[self.iloc]} ltlm={self.ltlm} incremental={self.incremental}")
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
        dxf = xf[:, 1:] - np.mean(xf, axis=1)[:, None]
        pf = dxf @ dxf.transpose() / (self.nmem - 1)
        logger.debug(f"pf max{np.max(pf)} min{np.min(pf)}")
        return pf

    def precondition(self,zmat,d=None,first=True):
        #u, s, vt = la.svd(zmat)
        #v = vt.transpose()
        #is2r = 1 / (1 + s**2)
        nk = zmat.shape[1]
        rho = 1.0
        if self.iinf<=-1 and self.iinf>-3:
            logger.info("==pre-multiplicative inflation==, alpha={}".format(self.infl_parm))
            rho = 1.0 / self.infl_parm
        c = zmat.transpose() @ zmat
        ga2f, vf = la.eigh(c)
        nrank = np.sum(ga2f>1.0e-10)
        logger.info("ga2full = {}".format(ga2f))
        logger.info(f"size={ga2f.size} rank={nrank}")
        ga2 = ga2f[::-1]
        v = vf[:,::-1]
        logger.info(f"v.shape={v.shape}")
        ga = np.where(ga2>1.0e-10,np.sqrt(ga2),0.0)
        logger.info("ga = {}".format(ga))
        u = np.dot(zmat,v[:,:nrank])/ga[:nrank]
        if self.iinf==-3 and not first:
            logger.info("==singular value adaptive inflation==")
            d_ = np.dot(u.transpose(),d)
            gainf = np.zeros_like(ga)
            gainf[:nrank] = self.inflfunc.est(d_,ga[:nrank])
            logger.info("ga inf ={}".format(gainf))
        lam = 1.0/(np.sqrt(ga2 + np.full(ga2.size,rho)))
        logger.info("lam ={}".format(lam))
        if self.iinf>=4 and not first:
            logger.info(f"==singular value inflation==, alpha={self.infl_parm}")
            laminf = self.inflfunc(lam,alpha1=self.infl_parm)
            logger.info("lam inf ={}".format(laminf))
        elif self.iinf==-3 and not first:
            laminf = self.inflfunc.g2f(ga,gainf)
            logger.info("lam inf ={}".format(laminf))
        else:
            laminf = lam.copy()
        if d is not None:
            d_ = np.dot(u.transpose(),d)
            self.inflfunc.pdr(d_, ga[:nrank], lam[:nrank], laminf[:nrank])
        D = np.diag(laminf)
        vt = v.transpose()
        tmat = v @ D @ vt
        heinv = tmat @ tmat.T
        logger.debug("tmat={}".format(tmat))
        logger.debug("heinv={}".format(heinv))
        logger.debug("eigen value ={}".format(lam))
        #print(f"rank(zmat)={lam[lam>1.0e-10].shape[0]}")
        return tmat, heinv

    def callback(self, xk, alpha=None):
        global zetak, alphak
        logger.debug("xk={}".format(xk))
        zetak.append(copy.copy(xk))
        if alpha is not None:
            alphak.append(alpha)

    def calc_j(self, zeta, *args, return_each=False):
        nk = zeta.size
        if not self.incremental:
            xc, pf, y, yloc, tmat, gmat, heinv, rinv = args
            x = xc + gmat @ zeta
            ob = y - self.obs.h_operator(yloc, x)
            jb = 0.5 * zeta.transpose() @ heinv @ zeta
            jo = 0.5 * ob.transpose() @ rinv @ ob
        #    j = 0.5 * (zeta.transpose() @ heinv @ zeta + ob.transpose() @ rinv @ ob)
        else:
        ## incremental form
            d, tmat, zmat, heinv = args
            w = tmat @ zeta
            jb = 0.5 * zeta.transpose() @ heinv @ zeta 
            jo = 0.5 * (zmat@w - d).transpose() @ (zmat@w - d)
            #j = 0.5 * (zeta.transpose() @ heinv @ zeta + (zmat@w - d).transpose() @ (zmat@w - d))
        if return_each:
            return jb, jo
        else:
            logger.info(f"jb:{jb:.6e} jo:{jo:.6e}")
            j = jb + jo
            return j

    def calc_grad_j(self, zeta, *args):
        nk = zeta.size
        if not self.incremental:
            xc, pf, y, yloc, tmat, gmat, heinv, rinv = args
            x = xc + gmat @ zeta
            hx = self.obs.h_operator(yloc, x)
            ob = y - hx
            if self.ltlm:
                dy = self.obs.dh_operator(yloc, x) @ pf
            else:
                dy = self.obs.h_operator(yloc, x[:, None] + pf) - hx[:, None]
            grad = heinv @ zeta - tmat @ dy.transpose() @ rinv @ ob
        else:
        ## incremental form
            d, tmat, zmat, heinv = args
            w = tmat @ zeta
            grad = heinv @ zeta + tmat @ zmat.transpose() @ (zmat@w - d)
        logger.info(f"|dj|:{np.sqrt(np.dot(grad,grad)):.6e}")
        return grad 

    def calc_hess(self, zeta, *args):
        nk = zeta.size
        if not self.incremental:
            xc, pf, y, yloc, tmat, gmat, heinv, rinv = args
            x = xc + gmat @ zeta
            if self.ltlm:
                dy = self.obs.dh_operator(yloc, x) @ pf
            else:
                dy = self.obs.h_operator(yloc, x[:, None] + pf) - self.obs.h_operator(yloc, x)[:, None]
            hess = tmat @ (np.eye(zeta.size) + dy.transpose() @ rinv @ dy) @ tmat
        else:
        ## incremental form
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

    def dfs(self, zmat):
        # Zupanski, D. et al., (2007) Applications of information theory in ensemble data assimilation
        # Eq. (10)
        u, s, vt = la.svd(zmat)
        ds = np.sum(s**2/(1.0+s**2))
        return ds

    def pfloc(self, sqrtpf, l_mat, save_dh, icycle,\
        op="linear",pt="envarbe",model="model"):
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
        op="linear",pt="envarbm",model="model"):
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

    def __call__(self, xb, pb, y, yloc, r=None, rmat=None, rinv=None,
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
                logger.debug("set R")
                r, rmat, rinv = self.obs.set_r(yloc)
            else:
                logger.debug("use input R")
            logger.debug("r={}".format(r))
            logger.debug("r^[-1/2]={}".format(rmat))
            xf = xb.copy()
            xf_ = np.mean(xb,axis=1)
            nmem = xf.shape[1]
            chi2_test = Chi(y.size, nmem, rmat)
            dxf = xf - xf_[:, None]
            #if self.iinf==-1:
            #    logger.info("==pre-multiplicative inflation==, alpha={}".format(self.infl_parm))
            #    dxf *= np.sqrt(self.infl_parm)
            pf = dxf / np.sqrt(nmem-1)
            fpf = dxf @ dxf.T / (nmem-1)
            if self.iinf == 3:
                stdv_f = np.sqrt(np.diag(fpf))
            if save_dh:
                np.save("{}_uf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), xb)
                np.save("{}_pf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), fpf)
                np.save("{}_spf_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), pf)
            if self.iloc is not None:
                logger.info("==localization==, lsig={}".format(self.lsig))
                dxf_orig = dxf.copy()
                pf_orig = dxf_orig / np.sqrt(nmem-1)
                if self.iloc == 1:
                    pf, wts = self.pfloc(pf_orig, self.l_mat, save_dh, icycle,\
                        op=self.op,pt=self.pt,model=self.model)
                elif self.iloc == 2:
                    pf = self.pfmod(pf_orig, self.l_sqrt, save_dh, icycle,\
                        op=self.op,pt=self.pt,model=self.model)
                logger.info("pf.shape={}".format(pf.shape))
                dxf = pf * np.sqrt(pf.shape[1]-1)
                xf = xf_[:, None] + dxf
            #logger.debug("norm(pf)={}".format(la.norm(pf)))
            #logger.debug("r={}".format(np.diag(r)))
            if self.ltlm:
                logger.debug("dhdx={}".format(self.obs.dhdx(xc)))
                dy = self.obs.dh_operator(yloc,xf_) @ dxf
            else:
                dy = self.obs.h_operator(yloc,xf_[:, None]+dxf) - self.obs.h_operator(yloc,xf_)[:, None]
            ob = y - self.obs.h_operator(yloc,xf_)
            logger.debug("ob={}".format(ob))
            logger.debug("dy={}".format(dy))
            if save_dh:
                np.save("{}_dh_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), dy)
                np.save("{}_d_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), ob)
            logger.info("save_dh={}".format(save_dh))
            zmat = rmat @ dy / np.sqrt(nmem-1)
            d = rmat @ ob
            if self.iinf == -2:
                logger.info("==adaptive pre-multiplicative inflation==")
                self.infl_parm = self.infladap(self.infl_parm, d, zmat)
            #    dxf *= np.sqrt(self.infl_parm)
            #    zmat *= np.sqrt(self.infl_parm)
            #logger.debug("cond(zmat)={}".format(la.cond(zmat)))
            tmat, heinv = self.precondition(zmat)
            logger.debug("dxf.shape={}".format(dxf.shape))
            logger.debug("tmat.shape={}".format(tmat.shape))
            logger.debug("heinv.shape={}".format(heinv.shape))
            gmat = pf @ tmat
            logger.debug("gmat.shape={}".format(gmat.shape))
            x0 = np.zeros(pf.shape[1])
            x = x0.copy()
            if not self.incremental:
                args_j = (xf_, pf, y, yloc, tmat, gmat, heinv, rinv)
            else:
                args_j = (d, tmat, zmat, heinv)
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
                            jb,jo = self.calc_j(np.array(zetak[i]), *args_j, return_each=True)
                            jh.append([jb,jo])
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
                    xa_ = xf_ + gmat @ x
                    xf_ = xa_
                    if self.ltlm:
                        dy = self.obs.dh_operator(yloc, xf_) @ dxf
                    else:
                        dy = self.obs.h_operator(yloc, xf_[:, None]+dxf) - self.obs.h_operator(yloc, xf_)[:, None]
                    zmat = rmat @ dy / np.sqrt(nmem-1)
                    tmat, heinv = self.precondition(zmat)
                    gmat = pf @ tmat
                    if update_ensemble:
                        pf = pf @ tmat
                    # update arguments
                    if not self.incremental:
                        args_j = (xf_, pf, y, yloc, tmat, gmat, heinv, rinv)
                    else:
                        d = rmat @ (y - self.obs.h_operator(yloc,xf_))
                        args_j = (d, tmat, zmat, heinv)
                    x0 = np.zeros(dxf.shape[1])
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
                    jh = np.zeros((len(zetak),2))
                    gh = np.zeros(len(zetak))
                    for i in range(len(zetak)):
                        #jh[i] = self.calc_j(np.array(zetak[i]), *args_j)
                        # calculate jb and jo separately
                        jb, jo = self.calc_j(np.array(zetak[i]), *args_j, return_each=True)
                        jh[i,0] = jb
                        jh[i,1] = jo
                        g = self.calc_grad_j(np.array(zetak[i]), *args_j)
                        gh[i] = np.sqrt(g.transpose() @ g)
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
                    x, flg = minimize(x0)
            xa_ = xf_ + gmat @ x
            if save_dh:
                np.save("{}_dx_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), gmat@x)
            if not self.incremental:
                if self.ltlm:
                    dy = self.obs.dh_operator(yloc,xa_) @ dxf
                else:
                    dy = self.obs.h_operator(yloc, xa_[:, None] + dxf) - self.obs.h_operator(yloc, xa_)[:, None]
                zmat = rmat @ dy / np.sqrt(nmem-1)
            #logger.debug("cond(zmat)={}".format(la.cond(zmat)))
            tmat, heinv = self.precondition(zmat, d=d, first=False)
            d = y - self.obs.h_operator(yloc, xa_)
            logger.debug("zmat shape={}".format(zmat.shape))
            logger.debug("d shape={}".format(d.shape))
            self.innv, self.chi2 = chi2_test(zmat, d)
            self.ds = self.dfs(zmat)
            logger.debug("dfs={}".format(self.ds))
            pa = pf @ tmat 
            dxa = pa * np.sqrt(nmem-1)
            if self.iinf == 2:
                logger.info("==RTPP==, alpha={}".format(self.infl_parm))
                dxa = (1.0 - self.infl_parm)*dxa + self.infl_parm*dxf
            
            if self.iloc is not None:
                nmem2 = dxf.shape[1]
                ptrace = np.sum(np.diag(dxa @ dxa.T)/(nmem2-1))
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
                    dxa = dxf @ tmat @ rvec / np.sqrt(nmem2-1)
                    dxa = dxa - dxa.mean(axis=1)[:,None]
                elif self.getkf:
                    if self.ltlm:
                        dy_orig = self.obs.dh_operator(yloc,xa_) @ dxf_orig
                    else:
                        dy_orig = self.obs.h_operator(yloc, xa_[:, None] + dxf_orig) - self.obs.h_operator(yloc, xa_)[:, None]
                    zmat_orig = rmat @ dy_orig
                    u, s, vt = la.svd(zmat, full_matrices=False)
                    logger.debug(f"s.shape={s.shape}")
                    logger.debug(f"u.shape={u.shape}")
                    logger.debug(f"vt.shape={vt.shape}")
                    sp = s**2 + (nmem2-1)
                    D = (1.0 - np.sqrt((nmem2-1)/sp))/s
                    nsig = D.size
                    reducedgain = dxf @ vt.transpose() @ np.diag(D) @ u.transpose()
                    dxa = dxf_orig - reducedgain @ zmat_orig
                else: # tmat computed from original forecast ensemble
                    if self.ltlm:
                        dy_orig = self.obs.dh_operator(yloc,xa_) @ dxf_orig
                    else:
                        dy_orig = self.obs.h_operator(yloc, xa_[:, None] + dxf_orig) - self.obs.h_operator(yloc, xa_)[:, None]
                    zmat_orig = rmat @ dy_orig
                    tmat, heinv = self.precondition(zmat_orig)
                    dxa = dxf_orig @ tmat
                trace = np.sum(np.diag(dxa @ dxa.T)/(nmem-1))
                logger.info("standard deviation ratio = {}".format(np.sqrt(ptrace / trace)))
                if np.sqrt(ptrace / trace) > 1.05:
                    dxa *= np.sqrt(ptrace / trace)
            if self.iinf == 0:
                logger.info("==multiplicative inflation==, alpha={}".format(self.infl_parm))
                dxa *= np.sqrt(self.infl_parm)
            if self.iinf == 1:
                logger.info("==additive inflation==, alpha={}".format(self.infl_parm))
                dxa += np.random.randn(dxa.shape[0], dxa.shape[1])*self.infl_parm
            if self.iinf == 3:
                fpa = dxa @ dxa.T / (nmem-1)
                stdv_a = np.sqrt(np.diag(fpa))
                logger.info("==RTPS, alpha={}".format(self.infl_parm))
                beta = ((1.0 - self.infl_parm)*stdv_a + self.infl_parm*stdv_f)/stdv_a
                logger.info(f"beta={beta}")
                dxa = dxa * beta[:, None]

            u = np.zeros_like(xb)
            u = xa_[:, None] + dxa
            fpa = pa @ pa.T
            if save_dh:
                np.save("{}_pa_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), fpa)
                np.save("{}_ua_{}_{}_cycle{}.npy".format(self.model, self.op, self.pt, icycle), u)
            if evalout:
                infl_mat = np.dot(zmat,zmat.T)
                evalb, _ = la.eigh(infl_mat)
                self.eval = evalb[::-1] / (1.0 + evalb[::-1])
            return u, fpa #, pa, innv, chi2, ds
