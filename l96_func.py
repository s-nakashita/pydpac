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
        self.nmem = params["nmem"]
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
        logger.info("nmem={} t0f={}".format(self.nmem, self.t0f))
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
        # spin up for 1 years
        for k in range(self.namax*self.nt):
            x = self.step(x)
        xt[0, :] = x
        for i in range(self.na-1):
            for k in range(self.nt):
                x = self.step(x)
            xt[i+1, :] = x
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

        xloc = np.arange(self.nx)
        obs_s = self.obs.get_sig()
        oberr = int(obs_s*1e4)
        obsfile="obs_{}_{}.npy".format(self.op, oberr)
        if not os.path.isfile(obsfile):
            logger.info("create obs")
            yobs = np.zeros((self.na,self.nobs,2)) # location and value
            if self.nobs == self.nx:
                obsloc = xloc.copy()
                for k in range(self.na):
                    yobs[k,:,0] = obsloc[:]
                    yobs[k,:,1] = self.obs.add_noise(self.obs.h_operator(obsloc, xt[k]))
            else:
                for k in range(self.na):
                    obsloc = np.random.choice(xloc, size=self.nobs, replace=False)
                    #obsloc = np.random.uniform(low=0.0, high=self.nx, size=self.nobs)
                    yobs[k,:,0] = obsloc[:]
                    yobs[k,:,1] = self.obs.add_noise(self.obs.h_operator(obsloc, xt[k]))
            #yobs = self.obs.add_noise(self.obs.h_operator(xt))
            np.save(obsfile, yobs)
        else:
            logger.info("read obs")
            yobs = np.load(obsfile)
        
        return xt, yobs

    # initialize control 
    def init_ctl(self):
        X0c = np.ones(self.nx)*self.F
        X0c[self.nx//2 - 1] += 0.001*self.F
        for j in range(self.t0c):
            X0c = self.step(X0c)
        return X0c

    # initialize ensemble member
    def init_ens(self,opt):
        #X0c = self.init_ctl()
        maxiter = np.max(np.array(self.t0f))+1
        if(opt==0): # random
            logger.info("spin up max = {}".format(self.t0c))
            X0c = self.init_ctl()
            #X0c = np.ones(self.nx)*self.F
            #X0c[self.nx//2 - 1] += 0.001*self.F
            #np.random.seed(514)
            X0 = np.zeros((self.nx, self.nmem))
            if self.pt == "mlef" or self.pt == "grad":
                X0[:, 0] = X0c
                X0[:, 1:] = np.random.normal(0.0,3.0,size=(self.nx,self.nmem-1)) / np.sqrt(self.nmem-1) + X0c[:, None]
            else:
                X0[:, :] = np.random.normal(0.0,3.0,size=(self.nx,self.nmem)) + X0c[:, None]
            #for j in range(self.t0c):
            #    X0c = self.step(X0c)
            #    X0 = self.step(X0)
        else: # lagged forecast
            logger.info("spin up max = {}".format(maxiter))
            X0 = np.zeros((self.nx,self.nmem))
            tmp = np.ones(self.nx)*self.F
            tmp[self.nx//2 - 1] += 0.001*self.F
            for j in range(maxiter):
                tmp = self.step(tmp)
                if j in self.t0f:
                    X0[:,self.t0f.index(j)] = tmp
        if self.pt == "mlef" or self.pt == "grad":
            pf = (X0[:, 1:] - X0c[:, None]) @ (X0[:, 1:] - X0c[:, None]).T
        else:
            pf = (X0 - np.mean(X0, axis=1)[:, None]) @ (X0 - np.mean(X0, axis=1)[:, None]).T / (self.nmem-1)
        logger.info("p0 max {}, min {}".format(np.max(pf),np.min(pf)))
        return X0, pf

    # initialize variables
    def initialize(self, opt=0):
        xa = np.zeros((self.na, self.nx))
        xf = np.zeros_like(xa)
        if self.ft == "deterministic":
            u = self.init_ctl()
            xf[0] = u
            if self.pt == "kf":
                pf = np.eye(self.nx)*25.0
            else:
                pf = np.eye(self.nx)*0.2
        else:
            u = np.zeros((self.nx, self.nmem))
            u, pf = self.init_ens(opt)
            if self.pt == "mlef" or self.pt == "grad":
                xf[0] = u[:, 0]
            else:
                xf[0] = np.mean(u, axis=1)
        if self.pt == "mlef" or self.pt == "grad":
            sqrtpa = np.zeros((self.na, self.nx, self.nmem-1))
        else:
            sqrtpa = np.zeros((self.na, self.nx, self.nx))
        return u, xa, xf, pf, sqrtpa

    # forecast
    def forecast(self, u, pa, tlm=True):
        if self.ft == "ensemble":
            uf = np.zeros((self.a_window, u.shape[0], u.shape[1]))
        else:
            uf = np.zeros((self.a_window, u.size))
        pf = np.zeros((self.a_window, pa.shape[0], pa.shape[0]))
        for l in range(self.a_window):
            for k in range(self.nt):
                u = self.step(u)
            uf[l] = u
        
            if self.pt == "etkf" or self.pt == "po" or self.pt == "letkf" or self.pt == "srf":
                #u[:, 0] = np.mean(u[:, 1:], axis=1)
                dxf = u - np.mean(u, axis=1).reshape(-1,1)
                p = dxf @ dxf.T / (self.nmem-1)
            elif self.pt == "mlef" or self.pt == "grad":
                spf = u[:, 1:] - u[:, 0].reshape(-1,1)
                p = spf @ spf.T
            elif self.pt == "kf":
                M = np.eye(u.shape[0])
                MT = np.eye(u.shape[0])
                if tlm:
                    E = np.eye(u.shape[0])
                    uk = u
                    for k in range(self.nt):
                        Mk = self.step.step_t(uk[:,None], E)
                        M = Mk @ M
                        MkT = self.step.step_adj(uk[:,None], E)
                        MT = MT @ MkT
                        uk = self.step(uk)
                else:
                    for k in range(self.nt):
                        M = self.analysis.get_linear(u, M)
                    MT = M.T
                p = M @ pa @ MT
            elif self.pt == "var" or self.pt == "var4d":
                p = pa
            pf[l] = p
        if self.a_window > 1:
            return uf, pf
        else:
            return u, p

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
