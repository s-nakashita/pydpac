import sys 
import os
import logging
from logging.config import fileConfig
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('param')

class Z05_func():
    def __init__(self,params):
        self.step = params["step"]
        self.nx, self.dt, self.dx, self.nu = self.step.get_params()
        self.obs = params["obs"]
        self.analysis = params["analysis"]
        self.nobs = params["nobs"]
        self.obsnet = params["obsnet"]
        self.t0c = params["t0c"]
        self.t0f = params["t0f"]
        self.nt = params["nt"]
        self.na = params["na"]
        self.a_window = params["a_window"]
        self.op = params["op"]
        self.pt = params["pt"]
        self.ft = params["ft"]
        #self.linf = params["linf"]
        #self.lloc = params["lloc"]
        self.ltlm = params["ltlm"]
        #self.infl_parm = params["infl_parm"]
        #self.lsig = params["lsig"]
        logger.info("nx={} dx={:7.3e} dt={:7.3e} nu={:.2f}".format
        (self.nx, self.dx, self.dt, self.nu))
        logger.info("nobs={}".format(self.nobs))
        logger.info("t0f={}".format(self.t0f))
        logger.info("nt={} na={}".format(self.nt, self.na))
        logger.info("operator={} perturbation={} sig_obs={} ftype={}".format\
        (self.op, self.pt, self.obs.get_sig(), self.ft))
        logger.info("TLM={}".format(self.ltlm))
        #logger.info("inflation={} localization={}".format(self.linf,self.lloc))
        #logger.info("infl_parm={} loc_parm={}".format(self.infl_parm, self.lsig))
        logger.info("Assimilation window size = {}".format(self.a_window))

    # generate truth
    def gen_true(self):
        tfile = "truth.npy"
        if not os.path.isfile(tfile):
            logger.info("create truth")
            xt = np.zeros((self.na, self.nx))
            b1 = 0.5
            b2 = 1.0
            #xmin, xmax = -25, 25
            t0 = -5.0
            for i in range(self.na):
                if i == 0:
                    xt[i] = self.step.soliton2(t0,np.sqrt(0.5*b1),np.sqrt(0.5*b2))
                else:
                    xtmp = xt[i-1]
                    for k in range(self.nt):
                        xtmp = self.step(xtmp)
                    xt[i] = xtmp
            np.save(tfile,xt)
        else:
            logger.info("read truth")
            xt = np.load(tfile)
        return xt

    # prepare observation
    def gen_obs(self, xt):
        obs_s = self.obs.get_sig()
        obsfile = f"obs{self.obsnet}_{self.op}_{int(obs_s*1e4)}.npy"
        if not os.path.isfile(obsfile):
            logger.info("create obs")
            yobs = self.create_obs(xt)
            np.save(obsfile,yobs)
        else:
            logger.info("read obs")
            yobs = np.load(obsfile)
            if yobs.shape[0] < self.na:
                logger.warning("observations are insufficient, recreate")
                yobs = self.create_obs(xt)
                np.save(obsfile,yobs)
        return yobs
    # create observations from truth
    def create_obs(self,xt):
        xloc = np.arange(self.nx)
        yobs = np.zeros((self.na, self.nobs, 2))
        logger.info(f"observation network={self.obsnet}")
        if self.obsnet == "all":
            for k in range(self.na):
                yobs[k,:,0] = xloc[:]
        elif self.obsnet == "fixed":
            ## random choice
            #obsloc = np.random.choice(xloc, size=self.nobs, replace=False)
            # search local max
            #ux = np.abs(np.roll(xt[0],-1) - np.roll(xt[0],1))
            #obsloc = np.sort(ux.argsort()[-self.nobs:])
            obsloc = np.sort(np.argsort(np.abs(xt[0]))[-self.nobs:])
            for k in range(self.na):
                yobs[k,:,0] = obsloc[:]
        elif self.obsnet == "targeted":
            # search local max
            for k in range(self.na):
                #ux = np.abs(np.roll(xt[k],-1) - np.roll(xt[k],1))
                #obsloc = np.sort(ux.argsort()[-self.nobs:])
                obsloc = np.sort(np.argsort(np.abs(xt[k]))[-self.nobs:])
                yobs[k,:,0] = obsloc[:]
        for k in range(self.na):
            yobs[k,:,1] = self.obs.h_operator(yobs[k,:,0],xt[k])
        yobs[:,:,1] = self.obs.add_noise(yobs[:,:,1])
        return yobs
    
    # initialize control
    def init_ctl(self):
        x0c = np.zeros(self.nx)
        b1 = 0.4
        b2 = 0.9
        x0c = self.step.soliton2(self.t0c, np.sqrt(0.5*b1), np.sqrt(0.5*b2))
        return x0c
    
    # initialize ensemble
    def init_ens(self):
        x0 = np.zeros((self.nx,len(self.t0f)+1))
        x0c = self.init_ctl()
        xc = x0c.copy()
        spf = np.zeros((self.nx,len(self.t0f)))
        b1 = 0.4
        b2 = 0.9
        b1e = b1 + 0.1*b1*np.random.randn(len(self.t0f))
        b2e = b2 + 0.1*b2*np.random.randn(len(self.t0f))
        for m in range(len(self.t0f)):
            spf[:,m] = self.step.soliton2(self.t0f[m], np.sqrt(0.5*b1e[m]), np.sqrt(0.5*b2e[m]))
        # spinup
        for i in range(400):
            xc = self.step(xc)
            spf = self.step(spf)
        spf = spf - xc[:, None]
        ## scale
        #spf /= np.sqrt(spf.shape[1]-1)
        x0[:,0] = x0c
        x0[:,1:] = spf + x0c[:,None]
        return x0
    
    # initialization
    def initialize(self):
        xa = np.zeros((self.na,self.nx))
        xf = np.zeros_like(xa)
        if self.ft == "deterministic":
            u = self.init_ctl()
            xf[0] = u
        else:
            u = self.init_ens()
            if self.pt == "mlef" or self.pt == "grad":
                xf[0] = u[:, 0]
            else:
                xf[0] = np.mean(u, axis=1)
        pa  = np.zeros((self.nx, self.nx))
        #savepa = np.zeros((self.na, self.nx, self.nx))
        return u, xa, xf, pa#, savepa

    # forecast
    def forecast(self,u):
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

    # plot initial state
    def plot_initial(self,uc,u,ut,obs,model='z05'):
        plt.rcParams["font.size"] = 18
        fig, ax = plt.subplots(figsize=(10,5))
        x = self.step.get_x()
        ax.plot(x,ut,label='true')
        ax.plot(x,uc,label='control')
        ax.plot(x,u,color='gray')
        xobs = np.zeros(obs.shape[0])
        for i in range(obs.shape[0]):
            xobs[i] = x[int(obs[i,0])]
        ax.scatter(xobs, obs[:,1], label='obs')
        ax.legend()
        fig.savefig(f"{model}_initial_{self.op}.png", bbox_inches='tight', dpi=300)
