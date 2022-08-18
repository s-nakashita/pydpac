import sys
import os
import logging
from logging.config import fileConfig
import random
import numpy as np
import matplotlib.pyplot as plt

logging.config.fileConfig("logging_config.ini")
logger = logging.getLogger('param')

class SO08_qg_func():

    def __init__(self, params):
        self.step = params["step"]
        self.ni,self.nj,self.dt,self.d,_,_,_,_,_,_,_ = self.step.get_params()
        self.nx = self.ni*self.nj
        self.step_t = params["step_t"]
        _,_,self.dt_t,_,_,_,_,_,_,_,_ = self.step_t.get_params()
        self.obs = params["obs"]
        self.analysis = params["analysis"]
        self.nobs = params["nobs"]
        self.t0c = params["t0c"]
        self.t0f = params["t0f"]
        self.t0true = params["t0true"]
        self.t_intobs = params["t_intobs"]
        #self.nt = params["nt"]
        self.nt = int(self.t_intobs / self.dt)
        self.nt_t = int(self.t_intobs / self.dt_t)
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
        logger.info("nx={} dt={:4.2f} dx={:6.4f}".format(self.nx, self.dt, self.d))
        logger.info("nobs={}".format(self.nobs))
        logger.info("t0f={}".format(self.t0f))
        logger.info("t0true={}".format(self.t0true))
        logger.info("nt={} nt_t={} na={}".format(self.nt, self.nt_t, self.na))
        logger.info("operator={} perturbation={} sig_obs={} ftype={}".format\
        (self.op, self.pt, self.obs.get_sig(), self.ft))
        logger.info("inflation={} localization={} TLM={}".format(self.linf,self.lloc,self.ltlm))
        logger.info("infl_parm={} loc_parm={}".format(self.infl_parm, self.lsig))
        logger.info("Assimilation window size = {}".format(self.a_window))
        self.truedir = "../../data/qg"

    # get truth and make observation
    def get_true_and_obs(self):
        #observation settings
        nvar = 1 #psi
        obs_offset = 10
        obs_interval = 18
        obsloc = np.zeros((self.na,self.nobs,3)) #ivar,iloc,jloc
        for icyc in range(self.na):
            obs_offset = random.randrange(1,11)
            iloc = obs_offset
            jloc = 1
            for i in range(self.nobs):
                obsloc[icyc,i,0] = nvar
                obsloc[icyc,i,1] = iloc
                obsloc[icyc,i,2] = jloc
                iloc += obs_interval
                if iloc > self.ni-2:
                    iloc = iloc - self.ni + 2
                    jloc += 3
            logger.debug("observation location {}".format(obsloc[icyc,]))
        obs_s = self.obs.get_sig()
        oberr = int(obs_s)
        obsfile="obs_{}_{}.npy".format(self.op, oberr)
        yobs = np.zeros((self.na,self.nobs,4)) # var,location[i,j] and value
        
        #fq = self.truedir+"/qtrue.npy"
        #fp = self.truedir+"/ptrue.npy"
        fq = "qtrue.npy"
        fp = "ptrue.npy"
        if not os.path.isfile(fq) or not os.path.isfile(fp):
            logger.info("create truth")
            truefile = self.truedir+f"/q{self.t0true:06d}.npy"
            qt = np.zeros((self.na*self.nt+1,self.nx))
            if not os.path.isfile(truefile):
                logger.error(f"not exist {truefile}")
                exit()
            else:
                logger.info("read initial q")
                qt[0,] = np.ravel(np.load(truefile))
            truefile = self.truedir+f"/p{self.t0true:06d}.npy"
            psit = np.zeros((self.na*self.nt+1,self.nx))
            if not os.path.isfile(truefile):
                logger.error(f"not exist {truefile}")
                exit()
            else:
                logger.info("read initial p")
                psit[0,] = np.ravel(np.load(truefile))
        #logger.debug("xt={}".format(xt))
        
            logger.info("start nature run and create obs")
            i=0
            for icyc in range(self.na):
                for it in range(self.nt_t):
                    q = qt[i,].reshape(self.ni,self.nj)
                    psi = psit[i,].reshape(self.ni,self.nj)
                    q = self.step_t(q,psi)
                    qt[i+1,] = np.ravel(q)
                    psit[i+1,] = np.ravel(psi)
                    i+=1
                u = np.concatenate([qt[i],psit[i]],0)
                yobs[icyc,:,:3] = obsloc[icyc,]
                yobs[icyc,:,3] = self.obs.add_noise(self.obs.h_operator(obsloc[icyc,],u))
            np.save(fq,qt)
            np.save(fp,psit)
            np.save(obsfile, yobs)
        else:
            logger.info("read truth")
            qt = np.load(fq)
            psit = np.load(fp)
            if not os.path.isfile(obsfile):
                logger.info("create obs")
                for icyc in range(self.na):
                    u = np.concatenate([qt[(icyc+1)*self.nt],psit[(icyc+1)*self.nt]],0)
                    yobs[icyc,:,:3] = obsloc[icyc,]
                    yobs[icyc,:,3] = self.obs.add_noise(self.obs.h_operator(obsloc[icyc,],u))
                np.save(obsfile, yobs)
            else:
                logger.info("read obs")
                yobs = np.load(obsfile)
            
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
            logger.debug(self.t0f[j])
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
            nmem = u.shape[1]
            qf = np.zeros((self.a_window, self.nx, nmem))
            psif = np.zeros((self.a_window, self.nx, nmem))
            for m in range(nmem):
                q = u[:self.nx,m].reshape(self.ni,self.nj)
                psi = u[self.nx:,m].reshape(self.ni,self.nj)
                for l in range(self.a_window):
                    for k in range(self.nt):
                        q = self.step(q,psi)
                    qf[l,:,m] = np.ravel(q)
                    psif[l,:,m] = np.ravel(psi)
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

    # plot true and analysis state
    def plot_state(self, t, q, psi, qt, psit, yobs):
        plt.rcParams["font.size"] = 16
        logger.info(f"plot true and analysis states, t={t}")
        logger.debug(t)
        logger.debug(q.shape)
        logger.debug(psi.shape)
        logger.debug(qt.shape)
        logger.debug(psit.shape)
        logger.debug(yobs.shape)
        zt = [psit.reshape(self.ni,self.nj).transpose(),qt.reshape(self.ni,self.nj).transpose()]
        if self.pt == "mlef":
            z = [psi[:,0].reshape(self.ni,self.nj).transpose(),q[:,0].reshape(self.ni,self.nj).transpose()]
        else:
            z = [psi.mean(axis=1).reshape(self.ni,self.nj).transpose(),q.mean(axis=1).reshape(self.ni,self.nj).transpose()]
        zmax = [6.0e1,1.5e5]
        x = np.linspace(0, 1, self.ni)
        y = np.linspace(0, 1, self.nj)
        title = [r"stream function $\psi$", r"potential vorticity $q$"]
        fname = ["p","q"]
        for i in range(len(z)):
            fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
            ax = axs[0]
            c = ax.pcolormesh(x, y, zt[i],
               cmap="RdYlBu_r", vmin=-zmax[i], vmax=zmax[i])
            ax.set_title("truth")
            ax.set_aspect("equal")
            fig.colorbar(c, ax=ax, shrink=0.8)
            if i==0:
                ax.scatter(yobs[:,1]/self.ni,yobs[:,2]/self.nj,marker='o',
                    c=yobs[:,3],s=7.0,edgecolors='k',linewidths=0.3,
                    cmap="RdYlBu_r", vmin=-zmax[0], vmax=zmax[0])
            ax = axs[1]
            c = ax.pcolormesh(x, y, z[i],
               cmap="RdYlBu_r", vmin=-zmax[i], vmax=zmax[i])
            ax.set_title("analysis")
            ax.set_aspect("equal")
            fig.colorbar(c, ax=ax, shrink=0.8)
            if t >= 0:
                fig.suptitle(r"$t=$"+f"{t*self.nt*self.dt}, "+title[i])
            else:
                fig.suptitle("initial, "+title[i])
            fig.tight_layout()
            if t >= 0:
                fig.savefig(f"{fname[i]}{t:06d}.png")
            else:
                fig.savefig(f"{fname[i]}initial.png")
            plt.close()

        fig, axs = plt.subplots(nrows=5,ncols=5,figsize=(15,15),
        subplot_kw={'xticks': [], 'yticks': []})
        for i in range(len(axs.flatten())):
            ax = axs.flatten()[i]
            p = psi[:,i].reshape(self.ni,self.nj).transpose()
            c = ax.pcolormesh(x, y, p,
                   cmap="RdYlBu_r", vmin=-zmax[0], vmax=zmax[0])
            if self.pt == "mlef":
                ax.set_title("mem "+f"{i}")
            else:
                ax.set_title("mem "+f"{i+1}")
            ax.set_aspect("equal")
        #fig.colorbar(c, shrink=0.8)
        #fig.suptitle(r"$t=$"+f"{self.t0true}")
        fig.tight_layout()
        if t>=0:
            fig.savefig(f"pens{t:06d}.png")
        else:
            fig.savefig(f"pensinitial.png")
        plt.close()
        fig, axs = plt.subplots(nrows=5,ncols=5,figsize=(15,15),
        subplot_kw={'xticks': [], 'yticks': []})
        for i in range(len(axs.flatten())):
            ax = axs.flatten()[i]
            p = q[:,i].reshape(self.ni,self.nj).transpose()
            c = ax.pcolormesh(x, y, p,
                   cmap="RdYlBu_r", vmin=-zmax[1], vmax=zmax[1])
            if self.pt == "mlef":
                ax.set_title("mem "+f"{i}")
            else:
                ax.set_title("mem "+f"{i+1}")
            ax.set_aspect("equal")
        #fig.colorbar(c, shrink=0.8)
        #fig.suptitle(r"$t=$"+f"{self.t0true}")
        fig.tight_layout()
        if t>=0:
            fig.savefig(f"qens{t:06d}.png")
        else:
            fig.savefig(f"qensinitial.png")
        plt.close()

    # plot true state
    def plot_truth(self, t_int, qt, psit, yobs):
        plt.rcParams["font.size"] = 16
        logger.info("plot truth")
        logger.debug(qt.shape)
        logger.debug(psit.shape)
        logger.debug(yobs.shape)
        ncyc = yobs.shape[0]
        zmax = [6.0e1,1.5e5]
        x = np.linspace(0, 1, self.ni)
        y = np.linspace(0, 1, self.nj)
        title = [r"stream function $\psi$", r"potential vorticity $q$"]
        for icyc in range(0,ncyc,t_int):
            zt = [psit[(icyc+1)*self.nt,].reshape(self.ni,self.nj).transpose(),\
                qt[(icyc+1)*self.nt,].reshape(self.ni,self.nj).transpose()]
            fig, axs = plt.subplots(nrows=1,ncols=2,figsize=(12,5))
            for i in range(len(zt)):
                ax = axs[i]
                c = ax.pcolormesh(x, y, zt[i],
                cmap="RdYlBu_r", vmin=-zmax[i], vmax=zmax[i])
                ax.set_title(title[i])
                ax.set_aspect("equal")
                fig.colorbar(c, ax=ax, shrink=0.8)
                if i==0:
                    ax.scatter(yobs[icyc,:,1]/self.ni,yobs[icyc,:,2]/self.nj,marker='o',
                    c=yobs[icyc,:,3],s=7.0,edgecolors='k',linewidths=0.3,
                    cmap="RdYlBu_r", vmin=-zmax[i], vmax=zmax[i])
            fig.suptitle(f"truth, t={icyc*self.dt}")
            fig.tight_layout()
            fig.savefig(f"truth{icyc:06d}.png")
            plt.close()
            #observation check
            ut = np.concatenate([qt[(icyc+1)*self.nt,],psit[(icyc+1)*self.nt,]],0)
            nobs = yobs.shape[1]
            hx = self.obs.h_operator(yobs[icyc,:,:3],ut)
            fig, ax = plt.subplots(figsize=(8,6))
            ax.plot(np.arange(1,nobs+1),hx,linewidth=0.0,marker='o',label=r'$H(x^t)$')
            ax.plot(np.arange(1,nobs+1),yobs[icyc,:,3],linewidth=0.0,marker='x',label=r'$y$')
            ax.legend()
            ax.set_xlabel("observation number")
            fig.savefig(f"obs{icyc:06d}.png")
            plt.close()
