import sys
import logging
from logging.config import fileConfig
import os
import numpy as np
import matplotlib.pyplot as plt

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('param')

class Z08_func():

    def __init__(self, params):
        self.step = params["step"]
        self.nx, self.dx, self.dt, self.nu \
            = self.step.get_params()
        self.obs = params["obs"]
        self.analysis = params["analysis"]
        self.nobs = params["nobs"]
        self.t0off = params["t0off"]
        self.t0true = params["t0true"]
        self.t0f = params["t0f"]
        self.nt = params["nt"]
        self.na = params["na"]
        self.op = params["op"]
        self.pt = params["pt"]
        self.ft = params["ft"]
        self.linf = params["linf"]
        self.lloc = params["lloc"]
        self.ltlm = params["ltlm"]
        logger.info("nx={} nu={} dt={:7.3e} dx={:7.3e}"\
            .format(self.nx, self.nu, self.dt, self.dx))
        logger.info("t0true={} t0f={}".format(self.t0true, self.t0f))
        logger.info("nt={} na={} nobs={}".format(self.nt, self.na, self.nobs))
        logger.info("operator={} perturbation={} sigma={} ftype={}".format\
            (self.op, self.pt, self.obs.get_sig(), self.ft))
        logger.info("inflation={} localization={} TLM={}".format(self.linf,self.lloc,self.ltlm))

    # generate truth and initialize
    def gen_true(self, x):
        t0c = self.t0f[0]
        u = np.zeros_like(x)
        u[0] = 1
        ut = np.zeros((self.na, self.nx))
        if self.ft == "deterministic":
            u0 = np.zeros(self.nx)
        else:
            u0 = np.zeros((self.nx, len(self.t0f)))
        for k in range(self.t0true):
            u = self.step(u)
            if self.ft == "ensemble":
                if k+1 in self.t0f:
                    u0[:, self.t0f.index(k+1)] = u
        ut[0, :] = u
        for i in range(self.na-1):
            for k in range(self.nt):
                u = self.step(u)
                #j = (i + 1) * nt + k
                j = self.t0true + i * self.nt + k
                if self.ft == "deterministic":
                    if j == t0c:
                        u0 = u
                if self.ft == "ensemble":
                    if j in self.t0f:
                        u0[:, self.t0f.index(j)] = u
            ut[i+1, :] = u
        pa = np.eye(self.nx)
        return ut, u0, pa

    # initialize variables
    def init_hist(self, u0):
        if self.ft == "ensemble":
            ua = np.zeros((self.na, u0.shape[0], u0.shape[1]))
        else:
            ua = np.zeros((self.na, u0.size))
        uf = np.zeros_like(ua)
        uf[0] = u0
        if self.pt == "mlef" or self.pt == "grad":
            sqrtpa = np.zeros((self.na, u0.shape[0], u0.shape[1]-1))
        else:
            sqrtpa = np.zeros((self.na, self.nx, self.nx))
        return ua, uf, sqrtpa

    # make observation
    def gen_obs(self, u):
        xloc = np.arange(self.nx)
        obs_s = self.obs.get_sig()
        yobs = np.zeros((self.na, self.nobs, 2)) # location and value
        if self.nobs == self.nx:
            logger.info("regular observation")
            obsloc = xloc.copy()
            for k in range(self.na):
                yobs[k,:,0] = obsloc[:]
                yobs[k,:,1] = self.obs.h_operator(obsloc, u[k])
            yobs[:,:,1] = self.obs.add_noise(yobs[:,:,1])
        else:
            logger.info("random observation")
            for k in range(self.na):
                obsloc = np.random.choice(xloc, size=self.nobs, replace=False)
                #obsloc = np.random.uniform(low=0.0, high=self.nx, size=self.nobs) 
                yobs[k,:,0] = obsloc[:]
                yobs[k,:,1] = self.obs.h_operator(obsloc, u[k])
            yobs[:,:,1] = self.obs.add_noise(yobs[:,:,1])
        return yobs

    # (if observation file exist) get observation
    def get_obs(self, f):
        yobs = np.load(f)
        return yobs


    def forecast(self, u, pa, tlm=False):
        for k in range(self.nt):
            u = self.step(u)
        return u

    # plot initial state (for ensemble)
    def plot_initial(self, u, ut, model):
        fig, ax = plt.subplots()
        x = np.arange(ut.size) + 1
        ax.plot(x, ut, label="true")
        if self.pt != "mlef":
            ax.plot(x, np.mean(u, axis=1), label="mean")
            for i in range(u.shape[1]):
                ax.plot(x, u[:,i], linestyle="--", color="tab:green", label="mem{}".format(i+1))
            diff = np.mean(u, axis=1) - ut
            ax.plot(x, diff, linestyle="dotted",color="tab:red",label="cntl-mean")
        else:
            for i in range(u.shape[1]):
                if i==0:
                    ax.plot(x, u[:,i], label="control")
                else:
                    ax.plot(x, u[:,i], linestyle="--", color="tab:green", label="mem{}".format(i))
            diff = u[:,0] - ut
            ax.plot(x, diff, linestyle="dotted",color="tab:red",label="cntl-true")
        ax.set(xlabel="points", ylabel="u", title="initial lag={}".format(self.t0off))
        ax.set_xticks(x[::10])
        ax.set_xticks(x[::5], minor=True)
        ax.legend()
        fig.savefig("{}_initial_{}_lag{}.png".format(model, self.pt, self.t0off))

    # plot initial spread
    def plot_spread(self, dxf):
        fig, ax = plt.subplots()
        x = np.arange(dxf.shape[0]) + 1
        spr = np.sqrt(np.mean(dxf**2, axis=1))
        for i in range(dxf.shape[1]):
            ax.plot(x, dxf[:,i], label="mem{}".format(i+1))
        ax.plot(x, spr, label="spread")
        ax.set(xlabel="points", ylabel="spread", title="initial spread lag={}".format(self.t0off))
        ax.set_xticks(x[::10])
        ax.set_xticks(x[::5], minor=True)
        ax.legend()
        fig.savefig("z08_init_spread_lag{}.png".format(self.t0off))
