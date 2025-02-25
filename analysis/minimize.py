try:
    from .lbfgs import lbfgs, lb3
    lbfgs_exist=True
except ImportError:
    try:
        from lbfgs import lbfgs, lb3
        lbfgs_exist=True
    except ImportError:
        lbfgs_exist=False
try:
    from .cgf import cgfam, cvsmod, cgdd
    cgf_exist=True
except ImportError:
    try:
        from cgf import cgfam, cvsmod, cgdd
        cgf_exist=True
    except ImportError:
        cgf_exist=False
try:
    from .file_utility import file_utility
    futil_exist=True
except ImportError:
    try:
        from file_utility import file_utility
        futil_exist=True
    except ImportError:
        futil_exist=False

import numpy as np
import numpy.linalg as la
import scipy.optimize as spo
import logging
from logging.config import fileConfig

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')

# standard status messages of optimizers
_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}
class Minimize():
    def __init__(self, n, func, jac=None, hess=None, prec=None, args=None, 
        iprint=np.array([0,0]), method="LBFGS", cgtype=None, maxiter=None,
        gtol=1.0e-6, disp=False, restart=False, loglevel=0):
        self.n = n
        self.m = min(self.n, 7)
        self.func = func
        self.jac = jac
        self.hess = hess
        self.prec = prec
        self.args = args
        self.method = method
        # self.cgtype = 1 : Fletcher-Reeves
        #               2 : Polak-Ribiere
        #               3 : Positive Polak-Ribiere
        self.cgtype = cgtype
        # for scipy.optimize.minimize
        self.maxiter = maxiter
        self.gtol = gtol
        self.disp = disp
        if restart:
            self.irest = 1
        else:
            self.irest = 0
        self.loglevel = loglevel # 0:info, >0:debug
        # for lbfgs and cgfam
        self.iprint = iprint
        self.xtol = 1.0e-16
        # for lbfgs
        self.diagco = False
        self.diag = np.ones(self.n)
        self.llwork = self.n*(2*self.m+1)+2*self.m
        self.lwork = np.zeros(self.llwork)
        # for cgfam
        self.desc = np.ones(self.n)
        self.lcwork = self.n
        self.cwork = np.zeros(self.lcwork)
        if self.loglevel==0:
            logger.info(f"method={self.method}")
            if (self.method=="cg" or self.method == "CG") and self.cgtype is not None:
                logger.info("%s%s" % ("cgtype: ", "Fletcher-Reeves" if self.cgtype == 1 else
                                              "Polak-Ribiere" if self.cgtype == 2 else
                                              "Positive Polak-Ribiere" if self.cgtype == 3
                                              else ""))
            logger.info(f"restart={self.irest==1}")
        else:
            logger.debug(f"method={self.method}")
            if (self.method=="cg" or self.method == "CG") and self.cgtype is not None:
                logger.debug("%s%s" % ("cgtype: ", "Fletcher-Reeves" if self.cgtype == 1 else
                                              "Polak-Ribiere" if self.cgtype == 2 else
                                              "Positive Polak-Ribiere" if self.cgtype == 3
                                              else ""))
            logger.debug(f"restart={self.irest==1}")
         
    def __call__(self, x0, callback=None):
        if self.method == "LBFGS":
            return self.minimize_lbfgs(x0, callback=callback)
        elif self.method == "CGF":
            return self.minimize_cgf(x0, callback=callback)
        elif self.method == "GD" or self.method == "GDF":
            return self.minimize_gd(x0, callback=callback)
        elif self.method == "EXN" or self.method == "NCG" or self.method == "TNC":
            return self.minimize_newton(x0, callback=callback)
        else:
            return self.minimize_scipy(x0, callback=callback)

    def minimize_gd(self, x0, callback=None):
        from scipy.optimize import line_search
        from scipy.optimize._linesearch import LineSearchWarning
        
        if self.args is not None:
            old_fval = self.func(x0, *self.args)
            gfk = self.jac(x0, *self.args)
        else:
            old_fval = self.func(x0)
            gfk = self.jac(x0)
        k = 0
        xk = x0
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2

        if self.maxiter is None:
            if self.method == "GD":
                maxiter = len(x0) * 200
            elif self.method == "GDF":
                maxiter = 1
        else:
            maxiter = self.maxiter
        warnflag = 0
        pk = -gfk
        gnorm = np.linalg.norm(gfk)
        nfev = 1
        ngev = 1
        alpha0 = 1.0#/gnorm
        info = 0
        while (gnorm > self.gtol) and (k < maxiter):
            if self.method == "GD":
                # line search
                if self.args is not None:
                    alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     line_search(self.func, self.jac, xk, pk, gfk=gfk,
                                 old_fval=old_fval, old_old_fval=old_old_fval,\
                                 args=self.args ,amax=1e20)
                else:
                    alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     line_search(self.func, self.jac, xk, pk, gfk=gfk,
                                 old_fval=old_fval, old_old_fval=old_old_fval,\
                                 amax=1e20)
            else:
                # fixed step-length to 1
                alpha_k = 1.0
                fc = 0
                gc = 0
            #dginit = np.dot(pk, gfk)
            ##logger.debug("dginit={}".format(dginit))
            #[dgout, infoout, xout, alpha_k] = cvsmod(n=self.n, x=xk, maxfev=30, info=info, 
            #    nfev=nfev, f=old_fval, g=gfk, s=pk, stp=alpha0, ftol=1.0e-4,
            #    gtol=0.1, xtol=self.xtol, stpmin=1e-20, stpmax=1e20, wa=self.cwork, 
            #    dginit=dginit)
            #info = infoout
            #logger.debug(xout-x0)
            #logger.debug(xk-x0)
            #alpha_k = 1.0
            #logger.debug("alpha_k={} info={} nfev={}".format(alpha_k, info, nfev))
            if alpha_k is None: # or info >= 2:
                warnflag = 2
                break
            nfev += fc
            ngev += gc
            xk = xk + alpha_k * pk
            #logger.debug(xk)
            if self.args is not None:
                gfk = self.jac(xk, *self.args)
                old_fval = self.func(xk, *self.args)
            else:
                gfk = self.jac(xk)
                old_fval = self.func(xk)
            nfev += 1
            ngev += 1
            #if info == -1:
            #    alpha0 = alpha_k
            #    continue
            pk = -gfk
            gnorm = np.linalg.norm(gfk)
            logger.debug("alpha_k={} gnorm={}".format(alpha_k, gnorm))
            #alpha0 = 1.0
            if callback is not None:
                callback(xk,alpha_k)
            k += 1
        if self.args is not None:
            fval = self.func(xk, *self.args)
        else:
            fval = self.func(xk)
        if warnflag == 2:
            msg = _status_message['pr_loss']
        elif k >= maxiter:
            warnflag = 1
            msg = _status_message['maxiter']
        elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
            warnflag = 3
            msg = _status_message['nan']
        else:
            msg = _status_message['success']

        if self.disp:
            if self.loglevel==0:
                logger.info("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
                logger.info("         Current function value: %f" % fval)
                logger.info("         Iterations: %d" % k)
                logger.info("         Function evaluations: %d" % nfev)
                logger.info("         Gradient evaluations: %d" % ngev)
            else:
                logger.debug("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
                logger.debug("         Current function value: %f" % fval)
                logger.debug("         Iterations: %d" % k)
                logger.debug("         Function evaluations: %d" % nfev)
                logger.debug("         Gradient evaluations: %d" % ngev)
        else:
            if self.loglevel==0:
                logger.info("success={} message={}".format(warnflag==0, msg))
                logger.info("success={} message={}".format(warnflag==0, msg))
            else:
                logger.debug("J={:7.3e} nit={}".format(fval, k))
                logger.debug("J={:7.3e} nit={}".format(fval, k))
        return xk, warnflag

    def minimize_newton(self, w0, callback=None, 
                    delta=1e-10, mu=0.5, c1=1e-4, c2=0.9):
        from scipy.optimize import line_search
        from scipy.optimize._linesearch import LineSearchWarning
        def pcg(g, H, M, delta=1e-10, eps=None, maxiter=30):
            j = 0
            pj = np.zeros_like(g)
            rj = g
            rnorm = np.sqrt(np.dot(rj, rj))
            zj = la.solve(M, rj)
            dj = -zj
            p = -g
            while (j < maxiter):
                # negative curvature test
                hd = H @ dj
                d2norm = np.dot(dj, dj)
                #logger.debug(f"d^T@H@d={np.dot(dj, hd)}")
                if np.dot(dj, hd) <= delta*d2norm:
                    logger.debug("satisfy negative curvature")
                    break
                # conjugate gradient update
                old_rj = np.dot(rj, zj)
                alpha = old_rj / np.dot(dj, hd)
                pj = pj + alpha * dj
                p = pj
                rj = rj + alpha * hd
                rnorm = np.sqrt(np.dot(rj, rj))
                if rnorm < eps:
                    logger.debug(f"truncate, ||rj||={rnorm}, eps={eps}")
                    break
                zj = la.solve(M, rj)
                beta = np.dot(rj, zj) / old_rj
                dj = -zj + beta * dj
                j += 1
            logger.debug(f"linear CG terminate at {j}th iteration")
            return p 

        def back_tracking(fun, wk, pk, gfk, old_fval, nfev, args, c1, c2):
            alpha0 = 1.0
            while (alpha0 > 0.5):
                wtrial = wk + alpha0 * pk
                if args is not None:
                    ftrial = fun(wtrial, *args)
                else:
                    ftrial = fun(wtrial)
                nfev += 1
                if (ftrial - old_fval <= c1*alpha0*np.dot(gfk, pk)) \
                    and (ftrial - old_fval >= c2*alpha0*np.dot(gfk, pk)):
                    break
                alpha0 *= 0.9
            return alpha0

        def quad_itpl(wk, alphab, pk, gfk, old_fval, nfev, ngev):
            # approximate function \phi as quadratic (\alpha_b is the previous step length)
            # \phi(\alpha) = F(w_k + \alpha*p_k)
            # \phi'(\alpha) = \nabla F^T p_k
            # \phi_q(\alpha) = a1*\alpha^2 + a2*\alpha + a3
            # a1 = (\phi(\alpha_b) - \phi'(0)\alpha_b - \phi(0))/\alpha_b^2
            # a2 = \phi'(0)
            # a3 = \phi(0)
            # \alpha = - a2 / (2*a1)
            wtrial = wk + alphab * pk
            if self.args is not None:
                ftrial = self.func(wtrial, *self.args)
                gtrial = self.jac(wtrial, *self.args)
            else:
                ftrial = self.func(wk)
                gtrial = self.jac(wk)
            nfev += 1
            ngev += 1
            # check the strong Wolfe conditions
            logger.debug(f"ftrial={ftrial:.4f} fcurrent={old_fval:.4f}")
            logger.debug(f"gnorm trial={np.sqrt(np.dot(gtrial, gtrial)):.4e} current={np.sqrt(np.dot(gfk, gfk)):.4e} descent={np.sqrt(np.dot(pk, pk)):.4e}")
            logger.debug(f"gtrial={np.abs(np.dot(gtrial, pk)):.4e} gcurrent={np.abs(np.dot(gfk, pk)):.4e}")
            if (ftrial < old_fval + c1 * alphab * np.dot(gfk, pk)) \
                and (np.abs(np.dot(gtrial, pk)) <= c2 * np.abs(np.dot(gfk, pk))):
                logger.debug("strong Wolfe conditions hold")
                return alphab
            # calculate new step length
            a3 = old_fval
            a2 = np.dot(gfk, pk)
            a1 = (ftrial - a2*alphab - a3) / alphab / alphab
            if a1 < 1e-10:
                return alphab
            else:
                return - 0.5 * a2 / a1

        w0 = np.asarray(w0).flatten()
        if self.maxiter is None:
            #maxiter = len(w0)*20
            maxiter = 100
        else:
            maxiter = self.maxiter
    
        if self.args is not None:
            old_fval = self.func(w0, *self.args)
            gfk = self.jac(w0, *self.args)
        else:
            old_fval = self.func(w0)
            gfk = self.jac(w0)
        nfev = 1
        ngev = 1
        k = 0
        wk = w0
        alphab = 1.0

        warnflag = 0
        gnorm = np.sqrt(np.dot(gfk, gfk))
        phik = min(mu,np.sqrt(gnorm))
        old_old_fval = old_fval + gnorm / 2

        if self.prec is not None:
            Mk = self.prec
        else:
            Mk = np.eye(self.n)
        logger.debug("preconditioner={}".format(Mk))

        while (gnorm > self.gtol) and (k < maxiter):
            if self.args is not None:
                Hk = self.hess(wk, *self.args)    
            else:
                Hk = self.hess(wk)    
            #lam, v = la.eigh(Hk)
        #lam[:len(lam)-1] = 0.0
        #Mk = v @ np.diag(lam) @ v.transpose()
            #logger.debug(f"Hessian eigen values=max{lam.max()}, min{lam.min()}")
            if self.method == "NCG" or self.method == "EXN":
                # solve Newton eq.
                pk = la.solve(Hk, -gfk)
            elif self.method == "TNC":
                # truncated Newton-CG
                eps = phik * gnorm
            #eps = 1e-8
                pk = pcg(gfk, Hk, Mk, delta=delta, eps=eps)
            rk = Hk @ pk + gfk
            logger.debug(f"residual:{np.sqrt(np.dot(rk, rk))}")
            #alphak = back_tracking(self.func, wk, pk, gfk, old_fval, nfev, 
            #        self.args, c1, c2)
            if self.method == "NCG":
                #alphak = quad_itpl(wk, alphab, pk, gfk, old_fval, nfev, ngev)
                if self.args is not None:
                    alphak, fc, gc, old_fval, old_old_fval, gfkp1 = \
                         line_search(self.func, self.jac, wk, pk, gfk=gfk,
                                 old_fval=old_fval, old_old_fval=old_old_fval,\
                                 args=self.args ,amax=1e20)
                else:
                    alphak, fc, gc, old_fval, old_old_fval, gfkp1 = \
                         line_search(self.func, self.jac, wk, pk, gfk=gfk,
                                 old_fval=old_fval, old_old_fval=old_old_fval,\
                                 amax=1e20)
                nfev += fc
                ngev += gc
            elif self.method == "EXN": #Exact Newton
                alphak = 1.0
            if alphak is None:
                warnflag = 2
                break
            wkp1 = wk + alphak * pk
            if self.args is not None:
                old_fval = self.func(wkp1, *self.args)
                gfkp1 = self.jac(wkp1, *self.args)
            else:
                old_fval = self.func(wkp1)
                gfkp1 = self.jac(wkp1)
            gfk = gfkp1
            wk = wkp1
            ngev += 1
            gnorm = np.sqrt(np.dot(gfk, gfk))
            nfev += 1
            if callback is not None:
                callback(wk, alphak)
            logger.debug(f"current:{k} gnorm={gnorm} step-length={alphak}")
            k += 1
            phik = min(mu/k, np.sqrt(gnorm))
            alphab = alphak
        if self.args is not None:
            fval = self.func(wk, *self.args)
        else:
            fval = self.func(wk)
        nfev += 1
        if warnflag == 2:
            msg = _status_message['pr_loss']
        elif k >= maxiter:
            warnflag = 1
            msg = _status_message['maxiter']
        elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(wk).any():
            warnflag = 3
            msg = _status_message['nan']
        else:
            msg = _status_message['success']

        if self.disp:
            if self.loglevel==0:
                logger.info("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
                logger.info("         Current function value: %f" % fval)
                logger.info("         Iterations: %d" % k)
                logger.info("         Function evaluations: %d" % nfev)
                logger.info("         Gradient evaluations: %d" % ngev)
            else:
                logger.debug("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
                logger.debug("         Current function value: %f" % fval)
                logger.debug("         Iterations: %d" % k)
                logger.debug("         Function evaluations: %d" % nfev)
                logger.debug("         Gradient evaluations: %d" % ngev)
        else:
            if self.loglevel==0:
                logger.info("success={} message={}".format(warnflag==0, msg))
                logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
                    fval, gnorm, k))
            else:
                logger.debug("success={} message={}".format(warnflag==0, msg))
                logger.debug("J={:7.3e} dJ={:7.3e} nit={}".format( \
                    fval, gnorm, k))
        return wk, warnflag

    def minimize_lbfgs(self, x0, callback=None):
        if not lbfgs_exist:
            raise ImportError('LBFGS cannot be used. Compile analysis/lbfgs.f first.')
        lb3.mp = 10
        lb3.lp = 11
        file_utility.file_open(lb3.mp, "minimize_monitor.log")
        file_utility.file_open(lb3.lp, "minimize.err")
        icall = 0
        iflag = 0

        fval = 0.0
        gval = np.zeros_like(x0)
        if self.args != None:
            fval = self.func(x0, *self.args)
            gval = self.jac(x0, *self.args)
        else:
            fval = self.func(x0)
            gval = self.jac(x0)
        if self.loglevel==0:
            logger.info("initial function value = {:13.6e}".format(fval))
            logger.info("initial gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
        else:
            logger.debug("initial function value = {:13.6e}".format(fval))
            logger.debug("initial gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
#        print("initial function value = {}".format(fval))
#        print("initial gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))

        #x = x0
        xk = x0.copy()
        #logger.info(f"id(x0)={id(x0)} id(x)={id(x)} id(xk)={id(xk)}")
        #alpha = 1.0
        #if callback != None:
        #    callback(x, alpha)
        if self.maxiter is None:
            maxiter = len(x0)*200
        else:
            maxiter = self.maxiter
        ## check stagnation
        #nomove = 0
        gold = gval
        wold = self.lwork
        while icall < maxiter:
            [xk, dk, alphak, self.lwork, oflag] = \
            lbfgs(n=self.n, m=self.m, x=xk, f=fval, g=gval, \
                          diagco=self.diagco, diag=self.diag, \
                          iprint=self.iprint, eps=self.gtol, xtol=self.xtol, w=wold, \
                          iflag=iflag, irest=self.irest)
            iflag = oflag
            #if self.loglevel==0:
            #    logger.info(f"iflag={iflag}")
            #else:
            logger.debug(f"iflag={iflag}")
            #update = np.dot((xk - x),(xk - x))
            #logger.debug(f"update={update}")
            #x = xk[:]
            wold = self.lwork[:]
            if iflag == 3: # line-search terminated successfully but not converged
                icall += 1
                if callback != None:
                    #if self.loglevel==0:
                    #    logger.info(f"icall={icall} xk={xk} alphak={alphak:13.6e}")
                    #else:
                    logger.debug(f"icall={icall} xk={xk} alphak={alphak:13.6e}")
                    callback(xk, alphak)
                if self.irest == 1:
                    # mutual orthogonality check
                    if np.dot(gold, gval) >= 0.2*np.dot(gval, gval):
                        if self.loglevel==0:
                            logger.info("not satisfy mutual orthogonality")
                        else:
                            logger.debug("not satisfy mutual orthogonality")
                        iflag = -2
                        break
                    # downhill check
                    if np.dot(gval, dk) > -0.8*np.dot(gval, gval) \
                        or np.dot(gval, dk) < -1.2*np.dot(gval, gval):
                        if self.loglevel==0:
                            logger.info("not satisfy downhill direction")
                        else:
                            logger.debug("not satisfy downhill direction")
                        iflag = -2
                        break
            elif iflag == 1: # in line-search iteration
                gold = gval
                if self.args != None:
                    fval = self.func(xk, *self.args)
                    gval = self.jac(xk, *self.args)
                else:
                    fval = self.func(xk)
                    gval = self.jac(xk)
            elif iflag == 0:
                if callback != None:
                    callback(xk, alphak)
                if self.loglevel==0:
                    logger.info("minimization success")
                    logger.info("iteration = {}".format(icall))
                    logger.info("final step-length = {:13.6e}".format(alphak))
                    logger.info("final function value = {:13.6e}".format(fval))
                    logger.info("final gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
                else:
                    logger.debug("minimization success")
                    logger.debug("iteration = {}".format(icall))
                    logger.debug("final step-length = {:13.6e}".format(alphak))
                    logger.debug("final function value = {:13.6e}".format(fval))
                    logger.debug("final gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
#                print("iteration = {}".format(icall))
#                print("final function value = {}".format(fval))
#                print("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
                break
            elif iflag < 0:
                if callback != None:
                    callback(xk, alphak)
                if self.loglevel==0:
                    logger.info("minimization failed, FLAG = {}".format(iflag))
                    logger.info("iteration = {}".format(icall))
                    logger.info("final step-length = {:13.6e}".format(alphak))
                    logger.info("final function value = {:13.6e}".format(fval))
                    logger.info("final gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
                else:
                    logger.debug("minimization failed, FLAG = {}".format(iflag))
                    logger.debug("iteration = {}".format(icall))
                    logger.debug("final step-length = {:13.6e}".format(alphak))
                    logger.debug("final function value = {:13.6e}".format(fval))
                    logger.debug("final gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
#                print("iteration = {}".format(icall))
#                print("final function value = {}".format(fval))
#                print("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
                break
        if iflag > 0:
            if self.loglevel==0:
                logger.info("minimization not converged")
                logger.info("current step-length = {:13.6e}".format(alphak))
                logger.info("current function value = {:13.6e}".format(fval))
                logger.info("current gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
            else:
                logger.debug("minimization not converged")
                logger.debug("current step-length = {:13.6e}".format(alphak))
                logger.debug("current function value = {:13.6e}".format(fval))
                logger.debug("current gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
#            print("minimization not converged")
#            print("current function value = {}".format(fval))
#            print("current gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
        file_utility.file_close(lb3.mp)
        file_utility.file_close(lb3.lp)

        return xk, iflag

    def minimize_cgf(self, x0, callback=None):
        if not cgf_exist:
            raise ImportError('CG+ cannot be used. Compile cgfam.f first.')
        cgdd.mp = 10
        cgdd.lp = 11
        file_utility.file_open(cgdd.mp, "minimize_monitor_cg.log")
        file_utility.file_open(cgdd.lp, "minimize_cg.err")
        icall = 0
        iflag = 0

        fval = 0.0
        gval = np.zeros_like(x0)
        if self.args != None:
            fval = self.func(x0, *self.args)
            gval = self.jac(x0, *self.args)
        else:
            fval = self.func(x0)
            gval = self.jac(x0)
        if self.loglevel==0:
            logger.info("initial function value = {:13.6e}".format(fval))
            logger.info("initial gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
        else:
            logger.debug("initial function value = {:13.6e}".format(fval))
            logger.debug("initial gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
#        print("initial function value = {}".format(fval))
#        print("initial gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))

        x = x0.copy()
        xold = x.copy()
        gold = gval.copy()
        gold_old = gold.copy()
        dold = self.desc.copy()
        finish = False
        if callback != None:
            callback(x, 0.0)
        if self.maxiter is None:
            maxiter = len(x0)*200
        else:
            maxiter = self.maxiter
        while icall < maxiter:
            [x, gval, self.desc, gold, alphak, oflag, ofinish] = \
                cgfam(n=self.n, x=xold, f=fval, g=gold, \
                    d=dold, gold=gold_old, \
                    iprint=self.iprint, eps=self.gtol, w=self.cwork, iflag=iflag, \
                    irest=self.irest, method=self.cgtype, finish=finish)
            iflag = oflag
            finish = bool(ofinish==1)
            if self.loglevel==0:
                logger.info(f"iflag={iflag} finish={finish}")
            else:
                logger.debug(f"iflag={iflag} finish={finish}")
            if iflag <= 0:
                break
            if iflag == 2:
                if callback != None:
                    callback(x, alphak)
                gnorm = np.sqrt(np.dot(gval, gval))
                xnorm = np.sqrt(np.dot(x,x))
                xnorm = max(1.0,xnorm)
                if gnorm/xnorm <= self.gtol:
                    finish = True
                icall += 1
                #tlev = self.gtol*(1.0+np.abs(fval))
                #i = 0
                #if (np.abs(gval[i]) > tlev):
                #    continue
                #else:
                #    i += 1
                #if i >= self.n-1:
                #    finish = True
            elif iflag == 1:
                xold = x[:]
                dold = self.desc[:]
                if self.args != None:
                    fval = self.func(x, *self.args)
                    gval = self.jac(x, *self.args)
                else:
                    fval = self.func(x)
                    gval = self.jac(x)
            ## mutual orthogonality check
            #if np.dot(gold, gval) >= 0.2*np.dot(gval, gval):
            #    logger.info("not satisfy mutual orthogonality")
            #    iflag = -2
            #    break
            ## downhill check
            #if np.dot(gval, self.desc) > -0.8*np.dot(gval, gval) \
            #    or np.dot(gval, self.desc) < -1.2*np.dot(gval, gval):
            #    logger.info("not satisfy downhill direction")
            #    iflag = -2
            #    break
            gold_old = gold[:]
            gold = gval[:]
            #if iflag == 1:
            #    icall += 1
            
        if iflag == 0:
            #if callback != None:
            #    callback(x, alphak)
            if self.loglevel==0:
                logger.info("minimization success")
                logger.info("iteration = {}".format(icall))
                logger.info("final step-length = {:13.6e}".format(alphak))
                logger.info("final function value = {:13.6e}".format(fval))
                logger.info("final gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
            else:
                logger.debug("minimization success")
                logger.debug("iteration = {}".format(icall))
                logger.debug("final step-length = {:13.6e}".format(alphak))
                logger.debug("final function value = {:13.6e}".format(fval))
                logger.debug("final gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
#                print("iteration = {}".format(icall))
#                print("final function value = {}".format(fval))
#                print("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
            
        if iflag < 0:
            if callback != None:
                callback(x, alphak)
            if self.loglevel==0:
                logger.info("minimization failed, FLAG = {}".format(iflag))
                logger.info("iteration = {}".format(icall))
                logger.info("final step-length = {:13.6e}".format(alphak))
                logger.info("final function value = {:13.6e}".format(fval))
                logger.info("final gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
            else:
                logger.debug("minimization failed, FLAG = {}".format(iflag))
                logger.debug("iteration = {}".format(icall))
                logger.debug("final step-length = {:13.6e}".format(alphak))
                logger.debug("final function value = {:13.6e}".format(fval))
                logger.debug("final gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
#                print("iteration = {}".format(icall))
#                print("final function value = {}".format(fval))
#                print("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
            
        if iflag > 0:
            if self.loglevel==0:
                logger.info("minimization not converged")
                logger.info("current step-length = {:13.6e}".format(alphak))
                logger.info("current function value = {:13.6e}".format(fval))
                logger.info("current gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
            else:
                logger.debug("minimization not converged")
                logger.debug("current step-length = {:13.6e}".format(alphak))
                logger.debug("current function value = {:13.6e}".format(fval))
                logger.debug("current gradient norm = {:13.6e}".format(np.sqrt(np.dot(gval, gval))))
#            print("minimization not converged")
#            print("current function value = {}".format(fval))
#            print("current gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
        file_utility.file_close(cgdd.mp)
        file_utility.file_close(cgdd.lp)
        return x, iflag

    def minimize_scipy(self, x0, callback=None):
        if self.method == "Nelder-Mead":
            if self.args is not None:
                res = spo.minimize(self.func, x0, args=self.args, method=self.method, \
                   options={'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            else:
                res = spo.minimize(self.func, x0, method=self.method, \
                   options={'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            if self.loglevel==0:
                logger.info("success={} message={}".format(res.success, res.message))
                logger.info("J={:7.3e} nit={}".format(res.fun, res.nit))
            else:
                logger.debug("success={} message={}".format(res.success, res.message))
                logger.debug("J={:7.3e} nit={}".format(res.fun, res.nit))
        elif self.method == "Powell":
            if self.args is not None:
                res = spo.minimize(self.func, x0, args=self.args, method=self.method, \
                   bounds=None, \
                   options={'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            else:
                res = spo.minimize(self.func, x0, method=self.method, \
                   bounds=None, \
                   options={'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            if self.loglevel==0:
                logger.info("success={} message={}".format(res.success, res.message))
                logger.info("J={:7.3e} nit={}".format(res.fun, res.nit))
            else:
                logger.debug("success={} message={}".format(res.success, res.message))
                logger.debug("J={:7.3e} nit={}".format(res.fun, res.nit))
        elif self.method == "dogleg" or self.method == "trust-ncg" \
            or self.method == "trust-krylov" or self.method == "trust-exact" \
            or self.method == "Newton-CG":
            if self.args is not None:
                res = spo.minimize(self.func, x0, args=self.args, method=self.method, \
                   jac=self.jac, hess=self.hess, options={'gtol':self.gtol, 'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            else:
                res = spo.minimize(self.func, x0, method=self.method, \
                   jac=self.jac, hess=self.hess, options={'gtol':self.gtol, 'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            if self.loglevel==0:
                logger.info("success={} message={}".format(res.success, res.message))
                logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
                res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))
            else:
                logger.debug("success={} message={}".format(res.success, res.message))
                logger.debug("J={:7.3e} dJ={:7.3e} nit={}".format( \
                res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))
        else:
            if self.args is not None:
                if self.jac is None:
                    res = spo.minimize(self.func, x0, args=self.args, method=self.method, \
                        jac='2-point', options={'gtol':self.gtol, 'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
                else:
                    res = spo.minimize(self.func, x0, args=self.args, method=self.method, \
                        jac=self.jac, options={'gtol':self.gtol, 'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            else:
                if self.jac is None:
                    res = spo.minimize(self.func, x0, method=self.method, \
                        jac='2-point', options={'gtol':self.gtol, 'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
                else:
                    res = spo.minimize(self.func, x0, method=self.method, \
                        jac=self.jac, options={'gtol':self.gtol, 'disp':self.disp, 'maxiter':self.maxiter}, callback=callback)
            if self.loglevel==0:
                logger.info("success={} message={}".format(res.success, res.message))
                logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
                res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))
            else:
                logger.debug("success={} message={}".format(res.success, res.message))
                logger.debug("J={:7.3e} dJ={:7.3e} nit={}".format( \
                res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))

        if res.success:
            iflag = 0
        else:
            iflag = -1

        return res.x, iflag

if __name__ == "__main__":        
    from scipy.optimize import rosen, rosen_der, rosen_hess
    import time
    import sys
    sys.path.append("~/")

    def sphere(x):
        return np.sum((x-1.0)**2)
    def sphere_der(x):
        return 2*(x-1.0)
    def sphere_hess(x):
        return np.eye(x.size)*2.0

    n = 2
    iprint = np.ones(2, dtype=np.int32)
    iprint[0] = 0
    iprint[1] = 0
    logger.info(iprint)

    args = None
    maxiter = 2000
    # initial guess
    #x0 = np.zeros(n)
    #for i in range(0, n, 2):
    #    x0[i] = -1.2
    #    x0[i+1] = 1.0
    x0 = np.ones(n) * -1.0
        
    method = "LBFGS"
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
     method=method, maxiter=maxiter)
    #minimize = Minimize(n, sphere, jac=sphere_der, args=args, iprint=iprint,
    # method=method, maxiter=None)

    start = time.time()
    #x = minimize.minimize_lbfgs(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    logger.info(f"x={x}")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")

    method = "BFGS"
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
     method=method, maxiter=maxiter)
    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    logger.info(f"x={x}")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")
    """
    method = "BFGS-jacfree"
    minimize = Minimize(n, rosen, jac=None, args=args, iprint=iprint,
     method="BFGS", maxiter=None)
    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")

    method = "CG"
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
     method=method, maxiter=None)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")

    method = "CG-jacfree"
    minimize = Minimize(n, rosen, jac=None, args=args, iprint=iprint,
     method="CG", maxiter=None)
    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")

    method = "Nelder-Mead"
    minimize = Minimize(n, rosen, args=args, iprint=iprint,
     method=method, maxiter=None)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")

    method = "Powell"
    minimize = Minimize(n, rosen, args=args, iprint=iprint,
     method=method, maxiter=None)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")
    """
    method = "GD"
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
     method=method, maxiter=maxiter)
    #minimize = Minimize(n, sphere, jac=sphere_der, args=args, iprint=iprint,
    # method=method, maxiter=10)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    logger.info(f"x={x}")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")

    method = "GDF"
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
     method=method, maxiter=maxiter)
    #minimize = Minimize(n, sphere, jac=sphere_der, args=args, iprint=iprint,
    # method=method, maxiter=10)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    logger.info(f"x={x}")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")
    
    method = "CGF"
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
     method=method, maxiter=maxiter, cgtype=1)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    logger.info(f"x={x}")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")

    method = "CGF"
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
     method=method, maxiter=maxiter, cgtype=2)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    logger.info(f"x={x}")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")

    method = "CGF"
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
     method=method, maxiter=maxiter, cgtype=3)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    logger.info(f"x={x}")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")
    """
    method = "Newton-CG"
    minimize = Minimize(n, rosen, jac=rosen_der, hess=rosen_hess, args=args, iprint=iprint,
     method=method, maxiter=None)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")
    """
    method = "EXN"
    minimize = Minimize(n, rosen, jac=rosen_der, hess=rosen_hess, args=args, iprint=iprint,
     method=method, maxiter=maxiter)
    #minimize = Minimize(n, sphere, jac=sphere_der, hess=sphere_hess, args=args, iprint=iprint,
    # method=method, maxiter=None)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    logger.info(f"x={x}")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")

    method = "NCG"
    minimize = Minimize(n, rosen, jac=rosen_der, hess=rosen_hess, args=args, iprint=iprint,
     method=method, maxiter=maxiter)
    #minimize = Minimize(n, sphere, jac=sphere_der, hess=sphere_hess, args=args, iprint=iprint,
    # method=method, maxiter=None)

    start = time.time()
    #x = minimize.minimize_scipy(x0)
    x, flg = minimize(x0)
    elapsed_time = time.time() - start
    logger.info("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    logger.info(f"x={x}")
    err = np.sqrt(np.mean((x-1.0)**2))
    logger.info(f"err={err}")
