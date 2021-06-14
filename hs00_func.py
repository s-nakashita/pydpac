import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
import numpy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
import time

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('param')

class HS_func():

    def __init__(self, params):
        self.tstep = params["tstep"]
        self.nx, self.dt, self.Ftrue = self.tstep.get_params()
        self.step = params["step"]
        self.nx, self.dt, self.Fmodel = self.step.get_params()
        self.obs = params["obs"]
        self.analysis = params["analysis"]
        self.nobs = params["nobs"]
        self.t0c = params["t0c"]
        self.t0f = params["t0f"]
        self.nt = params["nt"]
        self.na = params["na"]
        self.namax = params["namax"]
        self.a_window = params["a_window"]
        self.op = params["op"]
        self.pt = params["pt"]
        self.ft = params["ft"]
        #self.linf = params["linf"]
        #self.lloc = params["lloc"]
        #self.ltlm = params["ltlm"]
        #self.infl_parm = params["infl_parm"]
        #self.lsig = params["lsig"]
        self.aostype = params["aostype"]
        self.vt = params["vt"]
        logger.info("Truth : nx={} F={} dt={:7.3e}".format(self.nx, self.Ftrue, self.dt))
        logger.info("Forecast : nx={} F={} dt={:7.3e}".format(self.nx, self.Fmodel, self.dt))
        logger.info("nobs={}".format(self.nobs))
        logger.info("t0f={}".format(self.t0f))
        logger.info("nt={} na={}".format(self.nt, self.na))
        logger.info("operator={} perturbation={} sig_obs={} ftype={}".format\
        (self.op, self.pt, self.obs.get_sig(), self.ft))
        logger.info("AOS Type={}".format(self.aostype))
        if self.aostype[:2] == "SV":
            logger.info("optimization time = {}".format(self.vt))
        #logger.info("inflation={} localization={} TLM={}".format(self.linf,self.lloc,self.ltlm))
        #logger.info("infl_parm={} loc_parm={}".format(self.infl_parm, self.lsig))
        logger.info("Assimilation window size = {}".format(self.a_window))
    
    # generate truth
    def gen_true(self):
        xt = np.zeros((self.namax, self.nx))
        x = np.random.normal(0.25*self.Ftrue, scale=0.5*self.Ftrue, size=self.nx)
        tmp = x.copy()
        xt[0, :] = x
        for i in range(self.namax-1):
            for k in range(self.nt):
                tmp = self.tstep(x)
                x = tmp
            xt[i+1, :] = x
        return xt

    # get truth and make observation
    def get_true_and_obs(self):
        truefile = "hs00_ut.npy"
        if not os.path.isfile(truefile):
            logger.info("create truth")
            xt = self.gen_true()
            np.save(truefile,xt)
        else:
            logger.info("read truth")
            xt = np.load(truefile)
        #logger.debug("xt={}".format(xt))

        xloc = np.arange(self.nx)
        lloc = xloc[20:] # land region
        oloc = xloc[:20] # ocean region
        obs_s = self.obs.get_sig()
        oberr = int(obs_s*1e4)
        obsfile="obs_{}_{}.npy".format(self.op, oberr)
        if not os.path.isfile(obsfile):
            logger.info("create obs")
            yobs = np.zeros((self.na,self.nx,2)) # location and value
            obsloc = xloc
            for k in range(self.na):
                yobs[k,:,0] = obsloc[:]
                yobs[k,:,1] = self.obs.h_operator(obsloc, xt[k])
            yobs[:,:,1] = self.obs.add_noise(yobs[:,:,1])
            np.save(obsfile, yobs)
        else:
            logger.info("read obs")
            yobs = np.load(obsfile)
        
        return xt, yobs

    # get AOS observations
    def get_aos(self, yobs, cycle, aos):
        yloc = yobs[cycle:cycle+self.a_window,20:,0]
        y = yobs[cycle:cycle+self.a_window,20:,1]
        if self.aostype == "NO":
            logger.info("no supplementary observation")
        elif self.aostype == "RO":
            logger.info("random location")
            #np.random.seed(int(time.time()))
            aloc = np.ones(self.a_window)
            aobs = np.zeros(self.a_window)
            for i in range(self.a_window):
                obsloc = yobs[cycle+i,:,0].tolist()
                aloc[i] = np.random.choice(obsloc[:20], size=1)
                aobs[i] = yobs[cycle+i,obsloc.index(aloc[i]),1]
            yloc = np.concatenate((yloc, aloc[:, None]), axis=1)
            y = np.concatenate((y, aobs[:, None]), axis=1)
        else:
            logger.info(f"AOS type : {self.aostype}")
            aloc = np.ones(self.a_window)
            aobs = np.zeros(self.a_window)
            for i in range(self.a_window):
                obsloc = yobs[cycle+i,:,0].tolist()
                aloc[i] = aos
                aobs[i] = yobs[cycle+i,obsloc.index(aloc[i]),1]
            yloc = np.concatenate((yloc, aloc[:, None]), axis=1)
            y = np.concatenate((y, aobs[:, None]), axis=1)
        return yloc, y

    # calculate singular vector by lanczos method
    def lanczos(self, x, gnorm=None, vnorm=None):
        if gnorm is not None:
            v0 = np.random.normal(0.0, scale=0.1, size=gnorm.shape[1])
        else:
            v0 = np.random.normal(0.0, scale=0.1, size=self.nx)
        scale0 = np.sqrt(np.mean(v0**2))
        v = v0
        vt= x.shape[0]
        logger.debug("v shape={}".format(v.shape))
        for k in range(200):
            if gnorm is not None:
                tmp = gnorm @ v0
            else:
                tmp = v0
            logger.debug("tmp shape={}".format(tmp.shape))
            for i in range(vt):
                tmp = self.step.step_t(x[i], tmp)
            logger.debug("tmp shape={}".format(tmp.shape))
            if vnorm is not None:
                tmp = vnorm.T @ vnorm @ tmp
            for i in range(vt):
                tmp = self.step.step_adj(x[vt-i-1], tmp)
            logger.debug("tmp shape={}".format(tmp.shape))
            if gnorm is not None:
                v = gnorm.T @ tmp
            else:
                v = tmp
            logger.debug("v shape={}".format(v.shape))
            scale = np.sqrt(np.mean(v**2))
            v = v * scale0 / scale
            d = np.sqrt(np.mean((v-v0)**2))
            #logger.debug("diff {}".format(np.sqrt(np.mean((v-v0)**2))))
            if d < 1e-6:
                logger.info("lanczos converged at {}".format(k))
                if gnorm is not None:
                    return gnorm @ v
                else:
                    return v
            v0 = v
        logger.info("not converged {}".format(d))
        if gnorm is not None:
            return gnorm @ v
        else:
            return v 
    
    # calculate ensemble singular vector
    def ensv(self, fmat, vnorm=None):
        #gmat = xg - np.mean(xg, axis=1).reshape(-1,1)
        #fmat = xf - np.mean(xf, axis=1).reshape(-1,1)
        #ginv = la.inv(gmat.transpose()@gmat) @ gmat.transpose()
        if vnorm is not None:
            zmat = vnorm
        else:
            zmat = np.eye(self.nx)
        zmat = zmat @ fmat
        u, s, vt = la.svd(zmat)
        v = vt[0]
        #if gnorm is not None:
        #    v = gnorm @ v
        #return ginv @ v  
        return v

    # initialize control 
    def init_ctl(self, xt0):
        sigma = self.obs.get_sig()
        return np.random.normal(0.0, scale=sigma, size=xt0.size) + xt0
        
    # initialize ensemble member
    def init_ens(self, xt0, opt):
        maxiter = np.max(np.array(self.t0f))+1
        if(opt==0): # random
            logger.info("spin up max = {}".format(self.t0c))
            X0c = self.init_ctl(xt0)
            logger.debug("X0c={}".format(X0c))
            np.random.seed(514)
            X0 = np.zeros((self.nx, len(self.t0f)))
            if self.pt == "rep-mb":
                X0[:, :] = np.random.normal(0.0,0.01,size=(self.nx,len(self.t0f))) + X0c[:, None]
            else:
                X0[:, :] = np.random.normal(0.0,1.0,size=(self.nx,len(self.t0f))) + X0c[:, None]
        else: # lagged forecast
            logger.info("spin up max = {}".format(maxiter))
            X0 = np.zeros((self.nx,len(self.t0f)))
            tmp = np.ones(self.nx)*self.F
            tmp[self.nx//2 - 1] += 0.001*self.F
            for j in range(maxiter):
                tmp = self.step(tmp)
                if j in self.t0f:
                    X0[:,self.t0f.index(j)] = tmp
        return X0

    # initialize variables
    def initialize(self, opt=0):
        xt, yobs = self.get_true_and_obs()
        logger.info("truth length = {}".format(xt.shape[0]))
        xa = np.zeros((self.na, self.nx))
        xf = np.zeros_like(xa)
        if self.ft == "deterministic":
            u = self.init_ctl(xt[0])
            xf[0] = u
        else:
            u = self.init_ens(xt[0], opt)
            if self.pt == "mlef" or self.pt == "grad" or self.pt == "rep-mb":
                xf[0] = u[:, 0]
            else:
                xf[0] = np.mean(u, axis=1)
        pa  = np.zeros((self.nx, self.nx))
        #if self.pt == "mlef" or self.pt == "grad":
        #    sqrtpa = np.zeros((self.na, self.nx, len(self.t0f)-1))
        #else:
        sqrtpa = np.zeros((self.na, self.nx, self.nx))
        return xt, yobs, u, xa, xf, pa, sqrtpa

    # forecast
    def forecast(self, u):
        if self.ft == "ensemble":
            uf = np.zeros((self.a_window, u.shape[0], u.shape[1]))
        else:
            uf = np.zeros((self.a_window, u.size))
        for l in range(self.a_window):
            for k in range(self.nt):
                u = self.step(u)
            uf[l] = u
        
        if self.a_window > 1:
            return uf
        else:
            return u

    # (not used) plot initial state
    def plot_initial(self, uc, u, ut, lag, model):
        fig, ax = plt.subplots()
        x = np.arange(ut.size) + 1
        ax.plot(x, ut, label="true")
        ax.plot(x, uc, label="control")
        for i in range(u.shape[1]):
            ax.plot(x, u[:,i], linestyle="--", label="mem{}".format(i+1))
        ax.set(xlabel="points", ylabel="X", title="initial lag={}".format(lag))
        ax.set_xticks(x[::5])
        ax.set_xticks(x, minor=True)
        ax.legend()
        fig.savefig("{}_initial_lag{}.png".format(model, lag))
