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
        self.nmem = params["nmem"]
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
        logger.info("nmem={} t0true={} t0f={}".format(self.nmem, self.t0true, self.t0f))
        logger.info("nt={} na={}".format(self.nt, self.na))
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
            u0 = np.zeros((self.nx, self.nmem+1))
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
        if self.pt == "etkf" or self.pt == "po" \
            or self.pt == "letkf" or self.pt == "srf":
            dxf = u0[:, 1:] - u0[:, 0].reshape(-1,1) / np.sqrt(self.nmem-1)
            pf = dxf @ dxf.T
        elif self.pt == "mlef" or self.pt == "grad":
            dxf = u0[:, 1:] - u0[:, 0].reshape(-1,1)
            pf = dxf @ dxf.T
        else:
            pf = np.eye(self.nx) * 0.02
        return ut, u0, pf

    def init_hist(self, u0):
        if self.ft == "ensemble":
            ua = np.zeros((self.na, self.nx, self.nmem+1))
        else:
            ua = np.zeros((self.na, self.nx))
        uf = np.zeros_like(ua)
        uf[0] = u0
        if self.pt == "mlef" or self.pt == "grad":
            sqrtpa = np.zeros((self.na, self.nx, self.nmem))
        else:
            sqrtpa = np.zeros((self.na, self.nx, self.nx))
        return ua, uf, sqrtpa

    def gen_obs(self, u):
        y = self.obs.h_operator(self.obs.add_noise(u))
        return y

    def get_obs(self, f):
        y = np.load(f)
        return y


    def forecast(self, u, pa, tlm=False):
        for k in range(self.nt):
            u = self.step(u)
        if self.pt == "etkf" or self.pt == "po" \
            or self.pt == "letkf" or self.pt == "srf":
            u[:, 0] = np.mean(u[:, 1:], axis=1)
            dxf = u[:, 1:] - u[:, 0].reshape(-1,1) / np.sqrt(self.nmem-1)
            pf = dxf @ dxf.T
        elif self.pt == "mlef" or self.pt == "grad":
            dxf = u[:, 1:] - u[:, 0].reshape(-1,1)
            pf = dxf @ dxf.T
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
            pf = M @ pa @ MT
        else:
            pf = pa
        return u, pf


    def plot_initial(self, u, ut, model):
        fig, ax = plt.subplots()
        x = np.arange(ut.size) + 1
        ax.plot(x, ut, label="true")
        for i in range(u.shape[1]):
            if i==0:
                ax.plot(x, u[:,i], label="control")
            else:
                ax.plot(x, u[:,i], linestyle="--", color="tab:green", label="mem{}".format(i))
        ax.set(xlabel="points", ylabel="u", title="initial lag={}".format(self.t0off))
        ax.set_xticks(x[::10])
        ax.set_xticks(x[::5], minor=True)
        ax.legend()
        fig.savefig("{}_initial_lag{}.pdf".format(model, self.t0off))
