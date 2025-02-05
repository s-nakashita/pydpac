import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('param')

class Exp_func():

    def __init__(self, model, step, obs, params, save_data=True):
        self.model = model
        logger.info(f"Experimental setup: {self.model}")
        self.step = step
        model_params = self.step.get_params()
        if self.model == "l96":
            self.nx, self.dt, self.F = model_params
            logger.info("nx={} F={} dt={:7.3e}".format(self.nx, self.F, self.dt))
        elif len(model_params) < 5:
            self.nx, self.nk, self.dt, self.F = model_params
            logger.info("nx={} nk={} F={} dt={:7.3e}".format(self.nx, self.nk, self.F, self.dt))
        else:
            self.nx, self.nk, self.ni, self.b, self.c, self.dt, self.F = model_params
            logger.info("nx={} nk={} ni={}".format(self.nx, self.nk, self.ni))
            logger.info("b={:.1f} c={:.1f} F={} dt={:7.3e}".format(self.b, self.c, self.F, self.dt))
        self.obs = obs
        self.save_data = save_data
        self.nobs = params["nobs"]
        self.nmem = params["nmem"]
        self.t0c = params["t0c"]
        self.t0off = params["t0off"]
        self.nt = params["nt"]
        self.na = params["na"]
        self.namax = params["namax"]
        self.a_window = params["a_window"]
        self.op = params["op"]
        self.pt = params["pt"]
        self.ft = params["ft"]
        self.rseed = params["rseed"]
        if self.rseed is not None:
            self.rng = np.random.default_rng(seed=self.rseed)
        else:
            self.rng = np.random.default_rng()
        if params["model_error"]:
            self.nx_true = params["nx_true"]
            logger.info("nx_true={}".format(self.nx_true))
        else:
            self.nx_true = self.nx
        logger.info("nobs={}".format(self.nobs))
        logger.info("nt={} na={}".format(self.nt, self.na))
        logger.info("operator={} perturbation={} sig_obs={} ftype={}".format\
        (self.op, self.pt, self.obs.get_sig(), self.ft))
        logger.info("Assimilation window size = {}".format(self.a_window))
    
    # generate truth
    def gen_true(self):
        xt = np.zeros((self.na, self.nx))
        x = np.random.randn(self.nx_true)
        # spin up
        logger.debug(self.namax*self.nt)
        for k in range(self.namax*self.nt):
            tmp = self.step(x)
            x[:] = tmp[:]
        xt[0, :] = x
        for i in range(self.na-1):
            for k in range(self.nt):
                tmp = self.step(x)
                x[:] = tmp[:]
            xt[i+1, :] = x
        return xt

    # get truth and make observation
    def get_true_and_obs(self,obsloctype="random"):
        truefile = "truth.npy"
        if not os.path.isfile(truefile):
            logger.info("create truth")
            xt = self.gen_true()
            if self.save_data: np.save("truth.npy",xt)
        else:
            logger.info("read truth")
            xtfull = np.load(truefile)
            if xtfull.shape[0] < self.na:
                logger.info("recreate truth")
                xt = self.gen_true()
                if self.save_data: np.save(truefile,xt)
            else:
                xt = xtfull[:self.na]
        
        xloc = np.arange(self.nx_true)
        obs_s = self.obs.get_sig()
        oberr = int(obs_s*1e4)
        obsfile="obs_{}_{}.npy".format(self.op, oberr)
        if not os.path.isfile(obsfile):
            logger.info("create obs")
            yobs = np.zeros((self.na,self.nobs,2)) # location and value
            if self.nobs == self.nx_true:
                logger.info("entire observation")
                obsloc = xloc.copy()
                for k in range(self.na):
                    yobs[k,:,0] = obsloc[:]
                    yobs[k,:,1] = self.obs.h_operator(obsloc, xt[k])
                yobs[:,:,1] = self.obs.add_noise(yobs[:,:,1])
            elif obsloctype=="regular":
                logger.info("regular observation: nobs={}".format(self.nobs))
                intobs = self.nx_true // self.nobs
                obsloc = xloc[::intobs]
                for k in range(self.na):
                    yobs[k,:,0] = obsloc[:]
                    yobs[k,:,1] = self.obs.h_operator(obsloc, xt[k])
            else:
                logger.info("random observation: nobs={}".format(self.nobs))
                obsloc = np.random.choice(xloc, size=self.nobs, replace=False)
                for k in range(self.na):
                    #obsloc = np.random.choice(xloc, size=self.nobs, replace=False)
                    #obsloc = xloc[:self.nobs]
                    #obsloc = np.random.uniform(low=0.0, high=self.nx, size=self.nobs)
                    yobs[k,:,0] = obsloc[:]
                    yobs[k,:,1] = self.obs.h_operator(obsloc, xt[k])
                yobs[:,:,1] = self.obs.add_noise(yobs[:,:,1])
            if self.save_data: np.save(obsfile, yobs)
        else:
            logger.info("read obs")
            yobs = np.load(obsfile)
        
        return xt, yobs

    # initialize control 
    def init_ctl(self):
        X0c = self.rng.normal(0, scale=1.0, size=self.nx)
        #ix = np.arange(self.nx)/self.nx
        #nk = 2.0
        #X0c = np.cos(2.0*np.pi*ix*nk)*self.F
        tmp = X0c.copy()
        for j in range(self.t0c):
            tmp = self.step(X0c)
            X0c = tmp
        return X0c

    # initialize ensemble member
    def init_ens(self,opt):
        # t0 for ensemble members
        if self.nmem%2 == 0: # even
            t0m = [self.t0c + self.t0off//2 + self.t0off * i for i in range(self.nmem//2)]
            t0f = t0m + [self.t0c + self.t0off//2 + self.t0off * i for i in range(-self.nmem//2, 0)]
        else: # odd
            t0m = [self.t0c + self.t0off//2 + self.t0off * i for i in range(-(self.nmem-1)//2, (self.nmem-1)//2)]
            t0f = [self.t0c] + t0m
        maxiter = np.max(np.array(t0f))+1
        if(opt==0): # random
            logger.info("spin up max = {}".format(self.t0c))
            X0c = self.init_ctl()
            logger.debug("X0c={}".format(X0c))
            X0 = np.zeros((self.nx, len(t0f)))
            X0[:, :] = self.rng.normal(0.0,1.0,size=(self.nx,len(t0f))) + X0c[:, None]
        else: # lagged forecast
            logger.info("t0f={}".format(t0f))
            logger.info("spin up max = {}".format(maxiter))
            X0 = np.zeros((self.nx,len(t0f)))
            tmp = self.rng.normal(0.0,1.0,size=self.nx)
            #tmp = np.ones(self.nx)*self.F
            #tmp[self.nx//2 - 1] += 0.001*self.F
            #ix = np.arange(self.nx)/self.nx
            #nk = 2.0
            #tmp = np.cos(2.0*np.pi*ix*nk)*self.F
            for j in range(maxiter):
                tmp = self.step(tmp)
                if j in t0f:
                    X0[:,t0f.index(j)] = tmp
        return X0

    # initialize variables
    def initialize(self, opt=0):
        xa = np.zeros((self.na, self.nx))
        xf = np.zeros_like(xa)
        xsa = np.zeros_like(xa)
        if self.ft == "deterministic":
            u = self.init_ctl()
            xf[0] = u
        else:
            u = self.init_ens(opt)
            if self.pt == "mlef":
                uc = u[:, 0]
                xf[0] = uc
                u[:,1:] = (u[:,1:] - uc[:,None])/np.sqrt(u.shape[1]-1) # first scaling
            else:
                xf[0] = np.mean(u, axis=1)
        pa  = np.zeros((self.nx, self.nx))
        return u, xa, xf, pa, xsa

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
            if self.ft == "ensemble":
                if self.pt=="mlef" or self.pt == "4dmlef":
                    um = uf[:,:,0]
                else:
                    um = uf.mean(axis=2)
            else:
                um = uf.copy()
            return um, uf
        else:
            if self.ft == "ensemble":
                if self.pt=="mlef" or self.pt=="4dmlef":
                    um = u[:,0]
                else:
                    um = u.mean(axis=1)
            else:
                um = u.copy()
            return um, u

    # plot initial state
    def plot_initial(self, uc, ut, lag):
        fig, ax = plt.subplots()
        x = np.arange(ut.size) + 1
        ax.plot(x, ut, label="true")
        if uc.ndim==1:
            ax.plot(x, uc, label="control")
        else:
            ax.plot(x, uc[:,0], label="control")
            for i in range(1,min(uc.shape[1], 6)):
                ax.plot(x, uc[:,i], linestyle="--", label="mem{}".format(i+1))
        ax.set(xlabel="points", ylabel="X", title="initial lag={}".format(lag))
        ax.set_xticks(x[::5])
        ax.set_xticks(x, minor=True)
        ax.legend()
        fig.savefig("{}_initial_lag{}.png".format(self.model, lag))
