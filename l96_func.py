import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('param')

class L96_func():

    def __init__(self, params):
        self.step = params["step"]
        self.nx, self.dt, self.F = self.step.get_params()
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
        self.linf = params["linf"]
        self.lloc = params["lloc"]
        self.ltlm = params["ltlm"]
        self.infl_parm = params["infl_parm"]
        self.lsig = params["lsig"]
        logger.info("nx={} F={} dt={:7.3e}".format(self.nx, self.F, self.dt))
        logger.info("nobs={}".format(self.nobs))
        logger.info("t0f={}".format(self.t0f))
        logger.info("nt={} na={}".format(self.nt, self.na))
        logger.info("operator={} perturbation={} sig_obs={} ftype={}".format\
        (self.op, self.pt, self.obs.get_sig(), self.ft))
        logger.info("inflation={} localization={} TLM={}".format(self.linf,self.lloc,self.ltlm))
        logger.info("infl_parm={} loc_parm={}".format(self.infl_parm, self.lsig))
        logger.info("Assimilation window size = {}".format(self.a_window))
    
    # generate truth
    def gen_true(self):
        xt = np.zeros((self.na, self.nx))
        x = np.ones(self.nx)*self.F
        x[self.nx//2 - 1] += 0.001*self.F
        tmp = x.copy()
        # spin up for 1 years
        logger.debug(self.namax*self.nt)
        for k in range(self.namax*self.nt):
            tmp = self.step(x)
            x = tmp
        xt[0, :] = x
        for i in range(self.na-1):
            for k in range(self.nt):
                tmp = self.step(x)
                x = tmp
            xt[i+1, :] = x
        return xt

    # get truth and make observation
    def get_true_and_obs(self):
        f = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
            "data/data.csv")
        truth = pd.read_csv(f)
        xt = truth.values.reshape(self.namax,self.nx)
        #truefile = "truth.npy"
        #if not os.path.isfile(truefile):
        #    logger.info("create truth")
        #    xt = self.gen_true()
        #    np.save("truth.npy",xt)
        #else:
        #    logger.info("read truth")
         #   xt = np.load(truefile)
        logger.debug("xt={}".format(xt))

        xloc = np.arange(self.nx)
        obs_s = self.obs.get_sig()
        oberr = int(obs_s*1e4)
        obsfile="obs_{}_{}.npy".format(self.op, oberr)
        if not os.path.isfile(obsfile):
            logger.info("create obs")
            yobs = np.zeros((self.na,self.nobs,2)) # location and value
            if self.nobs == self.nx:
                logger.info("regular observation")
                obsloc = xloc.copy()
                for k in range(self.na):
                    yobs[k,:,0] = obsloc[:]
                    yobs[k,:,1] = self.obs.h_operator(obsloc, xt[k])
                yobs[:,:,1] = self.obs.add_noise(yobs[:,:,1])
            else:
                logger.info("random observation")
                for k in range(self.na):
                    obsloc = np.random.choice(xloc, size=self.nobs, replace=False)
                    #obsloc = np.random.uniform(low=0.0, high=self.nx, size=self.nobs)
                    yobs[k,:,0] = obsloc[:]
                    yobs[k,:,1] = self.obs.h_operator(obsloc, xt[k])
                yobs[:,:,1] = self.obs.add_noise(yobs[:,:,1])
            np.save(obsfile, yobs)
        else:
            logger.info("read obs")
            yobs = np.load(obsfile)
        
        return xt, yobs

    # initialize control 
    def init_ctl(self):
        X0c = np.ones(self.nx)*self.F
        X0c[self.nx//2 - 1] += 0.001*self.F
        tmp = X0c.copy()
        for j in range(self.t0c):
            tmp = self.step(X0c)
            X0c = tmp
        return X0c

    # initialize ensemble member
    def init_ens(self,opt):
        maxiter = np.max(np.array(self.t0f))+1
        if(opt==0): # random
            logger.info("spin up max = {}".format(self.t0c))
            X0c = self.init_ctl()
            logger.debug("X0c={}".format(X0c))
            np.random.seed(514)
            X0 = np.zeros((self.nx, len(self.t0f)))
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
        xa = np.zeros((self.na, self.nx))
        xf = np.zeros_like(xa)
        if self.ft == "deterministic":
            u = self.init_ctl()
            xf[0] = u
        else:
            u = self.init_ens(opt)
            if self.pt == "mlef" or self.pt == "grad":
                xf[0] = u[:, 0]
            else:
                xf[0] = np.mean(u, axis=1)
        pa  = np.zeros((self.nx, self.nx))
        if self.pt == "mlef" or self.pt == "grad":
            sqrtpa = np.zeros((self.na, self.nx, len(self.t0f)-1))
        else:
            sqrtpa = np.zeros((self.na, self.nx, self.nx))
        return u, xa, xf, pa, sqrtpa

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
