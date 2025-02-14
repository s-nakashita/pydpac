import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('param')

class L05j_func():

    def __init__(self, naturestep, step, obs, params):
        self.naturestep = naturestep
        self.nx_t, self.dt_t, self.params_t = self.naturestep.get_params()
        self.step = step
        self.nx, self.dt, self.params_m = self.step.get_params()
        self.obs = obs
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
        self.linf = params["linf"]
        self.lloc = params["lloc"]
        self.ltlm = params["ltlm"]
        self.infl_parm = params["infl_parm"]
        self.lsig = params["lsig"]
        self.seed = params["seed"]
        logger.info("nature: nx={} dt={:7.3e} params={}".format(self.nx_t, self.dt_t, self.params_t))
        logger.info("model: nx={} dt={:7.3e} params={}".format(self.nx, self.dt, self.params_m))
        logger.info("nobs={}".format(self.nobs))
        logger.info("nt={} na={}".format(self.nt, self.na))
        logger.info("operator={} perturbation={} sig_obs={} ftype={}".format\
        (self.op, self.pt, self.obs.get_sig(), self.ft))
        logger.info("inflation={} localization={} TLM={}".format(self.linf,self.lloc,self.ltlm))
        logger.info("infl_parm={} loc_parm={}".format(self.infl_parm, self.lsig))
        logger.info("Assimilation window size = {}".format(self.a_window))
        self.key = jax.random.key(self.seed)
    
    # generate truth
    def gen_true(self):
        xt = np.zeros((self.na, self.nx))
        #x = np.ones(self.nx)*self.F
        #x[self.nx//2 - 1] += 0.001*self.F
        key, subkey = jax.random.split(self.key)
        x = jax.random.normal(subkey,(self.nx))
        self.key = key
        #tmp = x.copy()
        # spin up for 1 years
        logger.debug(self.namax*self.nt)
        for k in range(self.namax*self.nt):
            x = self.naturestep(x)
        xt[0, :] = jax.device_get(x)
        for i in range(self.na-1):
            for k in range(self.nt):
                x = self.naturestep(x)
            xt[i+1, :] = jax.device_get(x)
        return xt

    # get truth and make observation
    def get_true_and_obs(self):
        #f = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
        #    "data/data.csv")
        #truth = pd.read_csv(f)
        #xt = truth.values.reshape(self.namax,self.nx)
        truefile = "truth.npy"
        if not os.path.isfile(truefile):
            logger.info("create truth")
            xt = self.gen_true()
            np.save("truth.npy",xt)
        else:
            logger.info("read truth")
            xt = np.load(truefile)
        #logger.debug("xt={}".format(xt))

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
                obsloc = np.random.choice(xloc, size=self.nobs, replace=False)
                for k in range(self.na):
                    #obsloc = np.random.choice(xloc, size=self.nobs, replace=False)
                    #obsloc = xloc[:self.nobs]
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
        #X0c = np.ones(self.nx)*self.F
        #X0c[self.nx//2 - 1] += 0.001*self.F
        key, subkey = jax.random.split(self.key)
        X0c = jax.random.normal(subkey,(self.nx))
        self.key = key
        ix = np.arange(self.nx)/self.nx
        #nk = 2.0
        #X0c = np.cos(2.0*np.pi*ix*nk)*self.F
        for j in range(self.t0c):
            X0c = self.step(X0c)
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
            key, subkey = jax.random.split(self.key)
            X0 = jax.random.normal(subkey,shape=(self.nx,len(t0f))) + X0c[:, None]
            self.key = key
        else: # lagged forecast
            logger.info("t0f={}".format(t0f))
            logger.info("spin up max = {}".format(maxiter))
            #X0 = jnp.zeros((self.nx,len(t0f)))
            #tmp = np.ones(self.nx)*self.F
            #tmp[self.nx//2 - 1] += 0.001*self.F
            ix = jnp.arange(self.nx)/self.nx
            nk = 2.0
            tmp = jnp.cos(2.0*jnp.pi*ix*nk)*self.F
            x0list = []
            for j in range(maxiter):
                tmp = self.step(tmp)
                if j in t0f:
                    x0list.append(tmp)
            #        X0[:,t0f.index(j)] = tmp
            X0 = jnp.asarray(x0list)
        return X0

    # initialize variables
    def initialize(self, opt=0):
        xa = np.zeros((self.na, self.nx))
        xf = np.zeros_like(xa)
        if self.ft == "deterministic":
            u = self.init_ctl()
            xf[0] = jax.device_get(u)
        else:
            u = self.init_ens(opt)
            if self.pt == "mlef":
                uc = u[:, 0]
                xf[0] = jax.device_get(uc)
                u[:,1:] = (u[:,1:] - uc[:,None])/jnp.sqrt(u.shape[1]-1) # first scaling
            else:
                xf[0] = jax.device_get(jnp.mean(u, axis=1))
        pa  = np.zeros((self.nx, self.nx))
        #if self.pt == "mlef" or self.pt == "grad":
        #    savepa = np.zeros((self.na, self.nx, self.nmem-1))
        #else:
        #savepa = np.zeros((self.na, self.nx, self.nx))
        return u, xa, xf, pa#, savepa

    # forecast
    def forecast(self, u):
        #if self.ft == "ensemble":
        #    uf = np.zeros((self.a_window, u.shape[0], u.shape[1]))
        #else:
        #    uf = np.zeros((self.a_window, u.size))
        uflist = []
        for l in range(self.a_window):
            for k in range(self.nt):
                u = self.step(u)
            #uf[l] = u
            uflist.append(u)
        if self.a_window > 1:
            uf = jnp.asarray(uflist)
            return uf
        else:
            return u

    # (not used) plot initial state
    def plot_initial(self, uc, u, ut, lag, model):
        fig, ax = plt.subplots()
        x = np.arange(ut.size) + 1
        ax.plot(x, ut, label="true")
        ax.plot(x, uc, label="control")
        for i in range(min(u.shape[1], 5)):
            ax.plot(x, u[:,i], linestyle="--", label="mem{}".format(i+1))
        ax.set(xlabel="points", ylabel="X", title="initial lag={}".format(lag))
        ax.set_xticks(x[::5])
        ax.set_xticks(x, minor=True)
        ax.legend()
        fig.savefig("{}_initial_lag{}.png".format(model, lag))
