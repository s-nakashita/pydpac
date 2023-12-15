import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('param')
#np.random.seed(514)

class L05nest_func():

    def __init__(self, step, obs, params_gm, params_lam):
        self.step = step
        self.nx_true = step.nx_true
        (self.nx_gm, self.nk_gm, self.dt_gm, self.F_gm),\
        (self.nx_lam, self.nk_lam, self.ni_lam, self.b, self.c, self.dt_lam, self.F_lam), self.lamstep, self.nsp \
            = self.step.get_params()
        logger.info("GM: nx={} nk={} F={} dt={:7.3e}".format(\
            self.nx_gm, self.nk_gm, self.F_gm, self.dt_gm))
        logger.info("LAM: nx={} nk={} ni={} b={} c={} F={} dt={:7.3e}".format(\
            self.nx_lam, self.nk_lam, self.ni_lam, self.b, self.c, self.F_lam, self.dt_lam))
        logger.info("GM region: {} to {} by {}".format(step.ix_gm[0],step.ix_gm[-1],(step.ix_gm[1]-step.ix_gm[0])))
        logger.info("LAM region: {} to {} by {}".format(step.ix_lam[0],step.ix_lam[-1],(step.ix_lam[1]-step.ix_lam[0])))
        logger.info("LAM step per 1 GM step ={}, LAM sponge width ={}".format(self.lamstep, self.nsp))
        self.obs = obs
        self.nobs = params_gm["nobs"]
        self.nmem = params_gm["nmem"]
        self.t0c = params_gm["t0c"]
        self.t0off = params_gm["t0off"]
        self.nt = params_gm["nt"]
        self.na = params_gm["na"]
        self.namax = params_gm["namax"]
        self.a_window = params_gm["a_window"]
        self.op = params_gm["op"]
        self.pt = params_gm["pt"]
        self.ft = params_gm["ft"]
        logger.info("nobs={}".format(self.nobs))
        logger.info("nmem={}".format(self.nmem))
        logger.info("nt={} na={}".format(self.nt, self.na))
        logger.info("operator={} perturbation={} sig_obs={} ftype={}".format\
        (self.op, self.pt, self.obs.get_sig(), self.ft))
        self.linf_gm = params_gm["linf"]
        self.lloc_gm = params_gm["lloc"]
        self.ltlm_gm = params_gm["ltlm"]
        self.infl_parm_gm = params_gm["infl_parm"]
        self.lsig_gm = params_gm["lsig"]
        logger.info("GM")
        logger.info("inflation={} localization={} TLM={}".format(self.linf_gm,self.lloc_gm,self.ltlm_gm))
        logger.info("infl_parm={} loc_parm={}".format(self.infl_parm_gm, self.lsig_gm))
        self.linf_lam = params_lam["linf"]
        self.lloc_lam = params_lam["lloc"]
        self.ltlm_lam = params_lam["ltlm"]
        self.infl_parm_lam = params_lam["infl_parm"]
        self.lsig_lam = params_lam["lsig"]
        logger.info("LAM")
        logger.info("inflation={} localization={} TLM={}".format(self.linf_lam,self.lloc_lam,self.ltlm_lam))
        logger.info("infl_parm={} loc_parm={}".format(self.infl_parm_lam, self.lsig_lam))
        logger.info("Assimilation window size = {}".format(self.a_window))
        self.lamstart = params_lam["lamstart"]
        self.anlsp = params_lam["anlsp"]
    
    # generate truth
    def gen_true(self):
        from model.lorenz3 import L05III
        # true model
        self.step_true = L05III(self.nx_true,self.nk_lam, self.ni_lam, \
            self.b, self.c, self.dt_lam, self.F_lam)
        xt = np.zeros((self.na, self.nx_true))
        #x = np.ones(self.nx)*self.F
        #x[self.nx//2 - 1] += 0.001*self.F
        x = np.random.randn(self.nx_true)
        #tmp = x.copy()
        # spin up
        logger.debug(self.namax*self.nt)
        for k in range(self.namax*self.nt):
            for j in range(self.lamstep):
                tmp = self.step_true(x)
                x[:] = tmp[:]
        xt[0, :] = x
        for i in range(self.na-1):
            for k in range(self.nt):
                for j in range(self.lamstep):
                    tmp = self.step_true(x)
                    x[:] = tmp[:]
            xt[i+1, :] = x
        return xt

    # get truth and make observation
    def get_true_and_obs(self,obsloctype="random"):
        #f = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
        #    "data/data.csv")
        #truth = pd.read_csv(f)
        #xt = truth.values.reshape(self.namax,self.nx)
        #truedir = os.path.join(os.path.abspath(os.path.dirname(__file__)),"data/l05III")
        truefile = "truth.npy"
        if not os.path.isfile(truefile):
            logger.info("create truth")
            xt = self.gen_true()
            np.save(truefile,xt)
        else:
            logger.info("read truth")
            xtfull = np.load(truefile)
            if xtfull.shape[0] < self.na:
                logger.info("recreate truth")
                xt = self.gen_true()
                np.save(truefile,xt)
            else:
                xt = xtfull[:self.na]
        #logger.debug("xt={}".format(xt))

        xloc = self.step.ix_true
        obs_s = self.obs.get_sig()
        oberr = int(obs_s*1e4)
        obsfile="obs_{}_{}.npy".format(self.op, oberr)
        obslamfile="obslam_{}_{}.npy".format(self.op, oberr)
        if not os.path.isfile(obsfile):
            logger.info("create obs")
            yobs = np.zeros((self.na,self.nobs,2)) # location and value
            iobs_lam = np.zeros((self.na,self.nobs)) # whether obs in LAM region (1) or not (0)
            if self.nobs == self.nx_true:
                logger.info("entire observation")
                obsloc = xloc.copy()
                if self.anlsp:
                    obs_in_lam = np.where((obsloc >= self.step.ix_lam[0])&(obsloc<=self.step.ix_lam[-1]), 1, 0)
                else:
                    obs_in_lam = np.where((obsloc >= self.step.ix_lam[self.nsp])&(obsloc<=self.step.ix_lam[-self.nsp]), 1, 0)
                for k in range(self.na):
                    yobs[k,:,0] = obsloc[:]
                    yobs[k,:,1] = self.obs.h_operator(obsloc, xt[k])
                    iobs_lam[k,:] = obs_in_lam
            elif obsloctype=="regular":
                logger.info("regular observation: nobs={}".format(self.nobs))
                intobs = self.nx_true // self.nobs
                obsloc = xloc[::intobs]
                if self.anlsp:
                    obs_in_lam = np.where((obsloc >= self.step.ix_lam[0])&(obsloc<=self.step.ix_lam[-1]), 1, 0)
                else:
                    obs_in_lam = np.where((obsloc >= self.step.ix_lam[self.nsp])&(obsloc<=self.step.ix_lam[-self.nsp]), 1, 0)
                for k in range(self.na):
                    yobs[k,:,0] = obsloc[:]
                    yobs[k,:,1] = self.obs.h_operator(obsloc, xt[k])
                    iobs_lam[k,:] = obs_in_lam
            else:
                logger.info("random observation: nobs={}".format(self.nobs))
                for k in range(self.na):
                    obsloc = np.random.choice(xloc, size=self.nobs, replace=False)
                    #obsloc = xloc[:self.nobs]
                    #obsloc = np.random.uniform(low=0.0, high=self.nx, size=self.nobs)
                    if self.anlsp:
                        obs_in_lam = np.where((obsloc >= self.step.ix_lam[0])&(obsloc<=self.step.ix_lam[-1]), 1, 0)
                    else:
                        obs_in_lam = np.where((obsloc >= self.step.ix_lam[self.nsp])&(obsloc<=self.step.ix_lam[-self.nsp]), 1, 0)
                    yobs[k,:,0] = obsloc[:]
                    yobs[k,:,1] = self.obs.h_operator(obsloc, xt[k])
                    iobs_lam[k,:] = obs_in_lam
            yobs[:,:,1] = self.obs.add_noise(yobs[:,:,1])
            np.save(obsfile, yobs)
            np.save(obslamfile, iobs_lam)
        else:
            logger.info("read obs")
            yobs = np.load(obsfile)
            iobs_lam = np.load(obslamfile)
        
        return xt, yobs, iobs_lam

    # initialize control 
    def init_ctl(self):
        #X0c = np.ones(self.nx)*self.F
        #X0c[self.nx//2 - 1] += 0.001*self.F
        X0c_gm = np.random.randn(self.nx_gm)
        for j in range(self.t0c):
            X0c_gm = self.step.gm(X0c_gm)
        gm2lam = interp1d(self.step.ix_gm,X0c_gm,axis=0)
        X0c_lam = gm2lam(self.step.ix_lam)
        return X0c_gm, X0c_lam

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
            X0c_gm, X0c_lam = self.init_ctl()
            logger.debug("X0c_gm={}".format(X0c_gm))
            logger.debug("X0c_lam={}".format(X0c_lam))
            X0_gm = np.zeros((self.nx_gm, len(t0f)))
            X0_gm[:, :] = np.random.normal(0.0,1.0,size=(self.nx_gm,len(t0f))) + X0c_gm[:, None]
            for j in range(self.t0c):
                X0_gm = self.step.gm(X0_gm)
        else: # lagged forecast
            logger.info("t0f={}".format(t0f))
            logger.info("spin up max = {}".format(maxiter))
            X0_gm = np.zeros((self.nx_gm,len(t0f)))
            tmp = np.ones(self.nx_gm)*self.F_gm
            tmp[self.nx_gm//2 - 1] += 0.001*self.F_gm
            #ix = np.arange(self.nx_gm)/self.nx_gm
            #nk = 2.0
            #tmp = np.cos(2.0*np.pi*ix*nk)*self.F_gm
            for j in range(maxiter):
                tmp = self.step.gm(tmp)
                if j in t0f:
                    X0_gm[:,t0f.index(j)] = tmp
        gm2lam = interp1d(self.step.ix_gm, X0_gm, axis=0)
        X0_lam = gm2lam(self.step.ix_lam)
        return X0_gm, X0_lam

    # initialize variables
    def initialize(self, opt=0):
        xa_gm  = np.zeros((self.na, self.nx_gm))
        xa_lam = np.zeros((self.na, self.nx_lam))
        xf_gm = np.zeros_like(xa_gm)
        xf_lam = np.zeros_like(xa_lam)
        xsa_gm = np.zeros_like(xa_gm)
        xsa_lam = np.zeros_like(xa_lam)
        if self.ft == "deterministic":
            u_gm, u_lam = self.init_ctl()
            xf_gm[0] = u_gm
            xf_lam[0] = u_lam
        else:
            u_gm, u_lam = self.init_ens(opt)
            if self.pt == "mlef":
                uc_gm = u_gm[:, 0]
                xf_gm[0] = uc_gm
                u_gm[:,1:] = (u_gm[:,1:] - uc_gm[:,None])/np.sqrt(u_gm.shape[1]-1) # first scaling
                uc_lam = u_lam[:, 0]
                xf_lam[0] = uc_lam
                u_lam[:,1:] = (u_lam[:,1:] - uc_lam[:,None])/np.sqrt(u_lam.shape[1]-1) # first scaling
            else:
                xf_gm[0] = np.mean(u_gm, axis=1)
                xf_lam[0] = np.mean(u_lam, axis=1)
        pa_gm  = np.zeros((self.nx_gm, self.nx_gm))
        pa_lam  = np.zeros((self.nx_lam, self.nx_lam))
        #if self.pt == "mlef" or self.pt == "grad":
        #    savepa = np.zeros((self.na, self.nx, self.nmem-1))
        #else:
        #savepa = np.zeros((self.na, self.nx, self.nx))
        return u_gm, xa_gm, xf_gm, pa_gm, xsa_gm, u_lam, xa_lam, xf_lam, pa_lam, xsa_lam #, savepa

    # forecast
    def forecast(self, u_gm, u_lam):
        if self.ft == "ensemble":
            uf_gm = np.zeros((self.a_window, u_gm.shape[0], u_gm.shape[1]))
            uf_lam = np.zeros((self.a_window, u_lam.shape[0], u_lam.shape[1]))
        else:
            uf_gm = np.zeros((self.a_window, u_gm.size))
            uf_lam = np.zeros((self.a_window, u_lam.size))
        for l in range(self.a_window):
            for k in range(self.nt):
                u_gm,u_lam = self.step(u_gm,u_lam)
            uf_gm[l] = u_gm
            uf_lam[l] = u_lam
        
        if self.a_window > 1:
            return uf_gm, uf_lam
        else:
            return u_gm, u_lam

    # (not used) plot initial state
    def plot_initial(self, uc_gm, uc_lam, ut, uens_gm=None, uens_lam=None, method=""):
        fig, ax = plt.subplots(figsize=[8,6],constrained_layout=True)
        ax.plot(self.step.ix_true, ut, label="true")
        ax.plot(self.step.ix_gm, uc_gm, label="GM,control")
        ax.plot(self.step.ix_lam, uc_lam, label="LAM,control")
        if uens_gm is not None and uens_lam is not None:
            for i in range(0,uens_gm.shape[1],uens_gm.shape[1]//5):
                ax.plot(self.step.ix_gm, uens_gm[:,i]+uc_gm, ls="--", c='gray',label="GM,mem{}".format(i+1))
                ax.plot(self.step.ix_lam, uens_lam[:,i]+uc_lam, ls="dotted", c='gray',label="LAM,mem{}".format(i+1))
        ax.set(xlabel="points", ylabel="X", title="initial state")
        ax.set_xticks(self.step.ix_true[::60])
        ax.set_xticks(self.step.ix_true[::20], minor=True)
        ax.legend(loc='upper left',bbox_to_anchor=(1.0,0.9))
        fig.savefig("initial_{}.png".format(method))
