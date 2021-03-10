if __name__ != "__main__":
    from . import lbfgs
import numpy as np
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
    def __init__(self, n, m, func, jac, args, iprint, method, options={}):
        self.n = n
        self.m = m
        self.func = func
        self.jac = jac
        self.args = args
        self.method = method
        # for lbfgs
        self.iprint = iprint
        self.lwork = self.n*(2*self.m+1)+2*self.m
        self.work = np.zeros(self.lwork)
        self.eps = 1.0e-5
        self.xtol = 1.0e-16
        self.diagco = False
        self.diag = np.ones(self.n)
        # for scipy.optimize.minimize
        self.options = {'gtol':1.0e-6, 'disp':False, 'maxiter':None}
        if 'gtol' in options:
            self.options['gtol'] = options['gtol']
        if 'disp' in options:
            self.options['disp'] = options['disp']
        if 'maxiter' in options:
            self.options['maxiter'] = options['maxiter']
        
    def __call__(self, x0, callback=None):
        if self.method == "LBFGS":
            return self.minimize_lbfgs(x0, callback=callback)
        elif self.method == "GD":
            return self.minimize_gd(x0, callback=callback)
        else:
            return self.minimize_scipy(x0, callback=callback)

    def minimize_gd(self, x0, callback=None):
        from scipy.optimize import line_search
        from scipy.optimize.linesearch import LineSearchWarning
        
        if self.args is not None:
            old_fval = self.func(x0, *self.args)
            gfk = self.jac(x0, *self.args)
        else:
            old_fval = self.func(x0)
            gfk = self.jac(x0)
        k = 0
        xk = x0
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2

        if self.options['maxiter'] is None:
            maxiter = len(x0) * 1000
        warnflag = 0
        pk = -gfk
        gnorm = np.linalg.norm(gfk)
        nfev = 1
        ngev = 1
        while (gnorm > self.options['gtol']) and (k < maxiter):
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
            if alpha_k is None:
                warnflag = 2
                break
            logger.debug("alpha_k={}".format(alpha_k))
            nfev += fc
            ngev += gc
            xk = xk + alpha_k * pk
            if self.args is not None:
                gfk = self.jac(xk, *self.args)
            else:
                gfk = self.jac(xk)
            ngev += 1
            pk = -gfk
            gnorm = np.linalg.norm(gfk)
            if callback is not None:
                callback(xk)
            k += 1
        if self.args is not None:
            fval = self.func(xk, *self.args)
        else:
            fval = self.func(xk)
        if warnflag == 2:
            msg = _status_message['pr_loss']
        elif k >= maxiter:
            msg = _status_message['maxiter']
        elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
            warnflag = 3
            msg = _status_message['nan']
        else:
            msg = _status_message['success']

        if self.options['disp']:
            logger.info("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
            logger.info("         Current function value: %f" % fval)
            logger.info("         Iterations: %d" % k)
            logger.info("         Function evaluations: %d" % nfev)
            logger.info("         Gradient evaluations: %d" % ngev)
        else:
            logger.info("success={} message={}".format(warnflag==0, msg))
            logger.info("J={:7.3e} nit={}".format(fval, k))
        return xk

    def minimize_lbfgs(self, x0, callback=None):
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
        logger.info("initial function value = {}".format(fval))
        logger.info("initial gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))

        x = x0.copy()
        if callback != None:
            callback(x)
        for icall in range(1000):
            [xk, oflag] = lbfgs.lbfgs(n=self.n, m=self.m, x=x, f=fval, g=gval, \
                          diagco=self.diagco, diag=self.diag, \
                          iprint=self.iprint, eps=self.eps, xtol=self.xtol, w=self.work, iflag=iflag)
            iflag = oflag
            x = xk[:]
            if callback != None and iflag == 1:
                callback(x)
            if self.args != None:
                fval = self.func(x, *self.args)
                gval = self.jac(x, *self.args)
            else:
                fval = self.func(x)
                gval = self.jac(x)
            if iflag == 0:
                if callback != None:
                    callback(x)
                logger.info("minimization success")
                logger.info("iteration = {}".format(icall))
                logger.info("final function value = {}".format(fval))
                logger.info("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
                break
            if iflag <= 0:
                if callback != None:
                    callback(x)
                logger.info("minimization failed, FLAG = {}".format(iflag))
                logger.info("iteration = {}".format(icall))
                logger.info("final function value = {}".format(fval))
                logger.info("final gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))
                break
        if iflag > 0:
            logger.info("minimization not converged")
            logger.info("current function value = {}".format(fval))
            logger.info("current gradient norm = {}".format(np.sqrt(np.dot(gval, gval))))

        return x    

    def minimize_scipy(self, x0, callback=None):
        if self.method == "Nelder-Mead":
            if self.args is not None:
                res = spo.minimize(self.func, x0, args=self.args, method=self.method, \
                   options={'disp':self.options['disp'], 'maxiter':self.options['maxiter']},
                   callback=callback)
            else:
                res = spo.minimize(self.func, x0, method=self.method, \
                   options={'disp':self.options['disp'], 'maxiter':self.options['maxiter']}, 
                   callback=callback)
            logger.info("success={} message={}".format(res.success, res.message))
            logger.info("J={:7.3e} nit={}".format(res.fun, res.nit))
        else:
            if self.args is not None:
                res = spo.minimize(self.func, x0, args=self.args, method=self.method, jac=self.jac, \
                    options={'gtol':self.options['gtol'], 'disp':self.options['disp'], 'maxiter':self.options['maxiter']}, \
                    callback=callback)
            else:
                res = spo.minimize(self.func, x0, method=self.method, jac=self.jac, 
                    options={'gtol':self.options['gtol'], 'disp':self.options['disp'], 'maxiter':self.options['maxiter']}, 
                    callback=callback)
            logger.info("success={} message={}".format(res.success, res.message))
            logger.info("J={:7.3e} dJ={:7.3e} nit={}".format( \
                res.fun, np.sqrt(res.jac.transpose() @ res.jac), res.nit))

        return res.x

if __name__ == "__main__":        
    from scipy.optimize import rosen, rosen_der
    import lbfgs
    import time

    n = 100
    m = 7
    iprint = np.ones(2, dtype=np.int32)
    iprint[0] = 1
    iprint[1] = 0
    print(iprint)

    args = None
    method = "LBFGS"
    minimize = Minimize(n, m, rosen, rosen_der, args, iprint, method)

    # initial guess
    x0 = np.zeros(n)
    for i in range(0, n, 2):
        x0[i] = -1.2
        x0[i+1] = 1.0
    
    start = time.time()
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")

    method = "BFGS"
    minimize = Minimize(n, m, rosen, rosen_der, args, iprint, method)

    start = time.time()
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")

    method = "CG"
    minimize = Minimize(n, m, rosen, rosen_der, args, iprint, method)

    start = time.time()
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")

    method = "Nelder-Mead"
    minimize = Minimize(n, m, rosen, rosen_der, args, iprint, method)

    start = time.time()
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")

    method = "GD"
    minimize = Minimize(n, m, rosen, rosen_der, args, iprint, method)

    start = time.time()
    x = minimize(x0)
    elapsed_time = time.time() - start
    print("{} elapsed_time:{:7.3e}".format(method, elapsed_time)+"s")
    err = np.sqrt(np.mean((x-1.0)**2))
    print(f"err={err}")