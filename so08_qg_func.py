import sys
import os
import logging
from logging.config import fileConfig
import numpy as np
import matplotlib.pyplot as plt

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('param')

class SO08_qg_func():

    def __init__(self, params):
        self.step = params["step"]
        self.ni,self.nj,self.dt,_,_,_,_,_,_,_,_ = self.step.get_params()
        self.nx = self.ni*self.nj
        self.obs = params["obs"]
        self.analysis = params["analysis"]
        self.nobs = params["nobs"]
        self.t0c = params["t0c"]
        self.t0f = params["t0f"]
        self.t0true = params["t0true"]
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
        logger.info("nx={} dt={:4.2f}".format(self.nx, self.dt))
        logger.info("nobs={}".format(self.nobs))
        logger.info("t0f={}".format(self.t0f))
        logger.info("t0true={}".format(self.t0true))
        logger.info("nt={} na={}".format(self.nt, self.na))
        logger.info("operator={} perturbation={} sig_obs={} ftype={}".format\
        (self.op, self.pt, self.obs.get_sig(), self.ft))
        logger.info("inflation={} localization={} TLM={}".format(self.linf,self.lloc,self.ltlm))
        logger.info("infl_parm={} loc_parm={}".format(self.infl_parm, self.lsig))
        logger.info("Assimilation window size = {}".format(self.a_window))
        self.truedir = "../../data/qg"

    # get truth and make observation
    def get_true_and_obs(self):
        #f = os.path.join(os.path.abspath(os.path.dirname(__file__)), \
        #    "data/data.csv")
        #truth = pd.read_csv(f)
        #xt = truth.values.reshape(self.namax,self.nx)
        truefile = self.truedir+f"/q{self.t0true:06d}.npy"
        qt = np.zeros((self.na*self.nt+1,self.nx))
        if not os.path.isfile(truefile):
            logger.error(f"not exist {truefile}")
            exit
        else:
            logger.info("read truth q")
            qt[0,] = np.ravel(np.load(truefile))
        truefile = self.truedir+f"/p{self.t0true:06d}.npy"
        psit = np.zeros((self.na*self.nt+1,self.nx))
        if not os.path.isfile(truefile):
            logger.error(f"not exist {truefile}")
            exit
        else:
            logger.info("read truth p")
            psit[0,] = np.ravel(np.load(truefile))
        #logger.debug("xt={}".format(xt))
        #observation settings
        obs_offset = 10
        obs_interval = 27
        obsloc = np.zeros((self.nobs,2))
        iloc = obs_offset
        jloc = 2
        for i in range(self.nobs):
            obsloc[i,0] = iloc
            obsloc[i,1] = jloc
            iloc += obs_interval
            if iloc > self.ni-2:
                iloc = iloc - self.ni + 2
                jloc += 2
        logger.info("observation location {}".format(obsloc))
        obs_s = self.obs.get_sig()
        oberr = int(obs_s)
        obsfile="obs_{}_{}.npy".format(self.op, oberr)
        
        logger.info("start nature run and create obs")
        logger.info("create obs")
        yobs = np.zeros((self.na,self.nobs,3)) # location[i,j] and value
        i=0
        for icyc in range(self.na):
            for it in range(self.nt):
                q = qt[i,].reshape(self.ni,self.nj)
                psi = psit[i,].reshape(self.ni,self.nj)
                q = self.step(q,psi)
                qt[i+1,] = np.ravel(q)
                psit[i+1,] = np.ravel(psi)
                i+=1
            yobs[icyc,:,:2] = obsloc
            yobs[icyc,:,2] = self.obs.add_noise(self.obs.h_operator(obsloc,psit[i,]))
        np.save(obsfile, yobs)
        
        return qt, psit, yobs

    # initialize control 
    def init_ctl(self):
        # lagged forecast
        q0 = np.zeros(self.nx)
        psi0 = np.zeros(self.nx)
        inf = self.truedir+f"/q{self.t0c:06d}.npy"
        q0 = np.ravel(np.load(inf))
        inf = self.truedir+f"/p{self.t0c:06d}.npy"
        psi0 = np.ravel(np.load(inf))
        return q0, psi0

    # initialize ensemble member
    def init_ens(self):
        # lagged forecast
        q0 = np.zeros((self.nx,len(self.t0f)))
        psi0 = np.zeros((self.nx,len(self.t0f)))
        for j in range(len(self.t0f)):
            inf = self.truedir+f"/q{self.t0f[j]:06d}.npy"
            q0[:,j] = np.ravel(np.load(inf))
            inf = self.truedir+f"/p{self.t0f[j]:06d}.npy"
            psi0[:,j] = np.ravel(np.load(inf))
        return q0, psi0

    # initialize variables
    def initialize(self):
        qa = np.zeros((self.na, self.nx))
        qf = np.zeros_like(qa)
        psia = np.zeros((self.na, self.nx))
        psif = np.zeros_like(psia)
        if self.ft == "deterministic":
            q,psi = self.init_ctl()
            qf[0,] = q
            psif[0,] = psi
        else:
            q,psi = self.init_ens()
            if self.pt == "mlef":
                qf[0] = q[:, 0]
                psif[0] = psi[:, 0]
            else:
                qf[0] = np.mean(q, axis=1)
                psif[0] = np.mean(psi, axis=1)
        pa  = np.zeros((self.nx*2, self.nx*2))
        u = np.concatenate([q,psi],0)
        #if self.pt == "mlef" or self.pt == "grad":
        #    savepa = np.zeros((self.na, self.nx, len(self.t0f)-1))
        #else:
        #savepa = np.zeros((self.na, self.nx, self.nx))
        return u, qa, psia, qf, psif, pa#, savepa

    # forecast
    def forecast(self,u):
        if self.ft == "ensemble":
            q = u[:self.nx,:].reshape(self.ni,self.nj,-1)
            psi = u[self.nx:,:].reshape(self.ni,self.nj,-1)
            qf = np.zeros((self.a_window, self.nx, q.shape[2]))
            psif = np.zeros((self.a_window, self.nx, psi.shape[2]))
        else:
            q = u[:self.nx].reshape(self.ni,self.nj)
            psi = u[self.nx:].reshape(self.ni,self.nj)
            qf = np.zeros((self.a_window, self.nx))
            psif = np.zeros((self.a_window, self.nx))
        for l in range(self.a_window):
            for k in range(self.nt):
                q = self.step(q,psi)
            qf[l] = np.ravel(q)
            psif[l] = np.ravel(psi)
        
        if self.a_window > 1:
            uf = np.concatenate([qf,psif],1)
            return uf
        else:
            uf = np.concatenate([qf[0],psif[0]],0)
            return uf

    # plot initial state
    def plot_initial(self, psi, qt, psit, yobs):
        plt.rcParams["font.size"] = 16
        fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
        z = [psit.reshape(self.ni,self.nj),qt.reshape(self.ni,self.nj)]
        zmax = [6.0e1,1.5e5]
        x = np.linspace(0, 1, self.ni)
        y = np.linspace(0, 1, self.nj)
        title = [r"stream function $\psi$", r"potential vorticity $q$"]
        for i in range(len(axs)):
            ax = axs[i]
            c = ax.pcolormesh(x, y, z[i],
                   cmap="RdYlBu_r", vmin=-zmax[i], vmax=zmax[i])
            ax.set_title(title[i])
            ax.set_aspect("equal")
            fig.colorbar(c, ax=ax, shrink=0.8)
        axs[0].scatter(yobs[:,0]/self.ni,yobs[:,1]/self.nj,marker='o',
                    c=yobs[:,2],s=3.0,edgecolors='k',
                    cmap="RdYlBu_r", vmin=-zmax[0], vmax=zmax[0])
        fig.suptitle(r"$t=$"+f"{self.t0true}")
        fig.tight_layout()
        fig.savefig("qg_initial_truth.png")

        fig, axs = plt.subplots(nrows=5,ncols=5,figsize=(15,15),
        subplot_kw={'xticks': [], 'yticks': []})
        for i in range(len(axs.flatten())):
            ax = axs.flatten()[i]
            p = psi[:,i].reshape(self.ni,self.nj)
            c = ax.pcolormesh(x, y, p,
                   cmap="RdYlBu_r", vmin=-zmax[0], vmax=zmax[0])
            ax.set_title(r"$t=$"+f"{self.t0f[i]}")
            ax.set_aspect("equal")
        #fig.colorbar(c, shrink=0.8)
        #fig.suptitle(r"$t=$"+f"{self.t0true}")
        fig.tight_layout()
        fig.savefig("qg_initial_ens.png")