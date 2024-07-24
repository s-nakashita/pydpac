import numpy as np
from numpy import linalg as la
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
import matplotlib.pyplot as plt
import logging

class EnASA():
    def __init__(self,vt,X0,Je,solver='minnorm',logfile='minnorm'):
        self.solver = solver
        self.vt = vt
        self.X0 = X0 #centered
        self.Je = Je #centered
        self.X = X0.T # (n_samples, n_features)
        self.nx = self.X.shape[1]
        self.nens = self.X.shape[0]
        self.logger = logging.getLogger(__name__)
        self.logfile = logfile
        logging.basicConfig(filename=f'{self.logfile}.log', encoding='utf-8', level=logging.INFO)
        
    def __call__(self,n_components=None,cthres=None,mu=0.01):
        if self.solver=='minnorm':
            dJedx0_s = self.enasa_minnorm(nrank=n_components,cthres=cthres)
        elif self.solver=='minvar':
            dJedx0_s = self.enasa_minvar()
        elif self.solver=='diag':
            dJedx0_s = self.enasa_diag()
        elif self.solver=='psd':
            dJedx0_s = self.enasa_psd()
        elif self.solver=='ridge':
            dJedx0_s = self.enasa_ridge(mu=mu)
        elif self.solver=='pcr':
            dJedx0_s = self.enasa_pcr(n_components=n_components,cthres=cthres)
        elif self.solver=='pls':
            dJedx0_s = self.enasa_pls(n_components=n_components,cthres=cthres)
        #print(f"dJedx0_s.shape={dJedx0_s.shape}")
        self.rescaling(dJedx0_s)
        
        return self.dJedx0

    def rescaling(self,dJedx0_s):
        if self.solver == 'pcr':
            #self.dJedx0 = dJedx0_s.copy()
            self.dJedx0 = dJedx0_s / self.std.scale_
            self.err = self.reg.intercept_ - np.dot(self.std.mean_, dJedx0_s)
        elif self.solver == 'pls':
            #self.dJedx0 = dJedx0_s.copy()
            self.dJedx0 = dJedx0_s / self.pls._x_std
            self.err = self.pls.intercept_ - np.dot(self.pls._x_mean, dJedx0_s)
        else:
            self.dJedx0 = dJedx0_s.copy()
            self.err = 0.0

    def estimate(self):
        #if self.solver == 'pcr':
        #    Je_est1 = self.pcr.predict(self.X)
        #elif self.solver == 'pls':
        #    Je_est = self.pls.predict(self.X)
        #else:
        Je_est = np.dot(self.X,self.dJedx0) + self.err
        return Je_est.ravel()

    def score(self):
        #if self.solver == 'pls':
        #    print(self.pls.score(self.X,self.Je))
        #elif self.solver == 'pcr':
        #    print(self.pcr.score(self.X,self.Je))
        #else:
        u = np.sum((self.Je - self.estimate())**2)
        v = np.sum((self.Je - self.Je.mean())**2)
        return 1.0 - u/v

    def enasa_minnorm(self,nrank=None,cthres=None):
        try:
            u, s, vt = la.svd(self.X0)
        except la.LinAlgError:
            dJedx0_s = np.full(self.nx,np.nan)
        else:
            if nrank is None:
                self.nrank = np.sum(s>s[0]*1.0e-10)
            else:
                self.nrank = nrank
            lam = s*s
            contrib = np.cumsum(lam)/np.sum(lam)
            if nrank is None and cthres is not None:
                nrank = 0
                while (nrank<self.nrank):
                    if contrib[nrank]>cthres: break
                    nrank += 1
                self.nrank = nrank
            self.logger.info(f"nrank {self.nrank} contrib {contrib[self.nrank]*1e2:.2f}%")
            v = vt.transpose()
            pinv = v[:,:self.nrank] @ np.diag(1.0/s[:self.nrank]/s[:self.nrank]) @ vt[:self.nrank,:]
            dJedx0_s = np.dot(np.dot(self.X0,pinv),self.Je)
        return dJedx0_s

    def enasa_minvar(self):
        dJedx0_s = np.dot(np.dot(np.linalg.inv(np.dot(self.X0,self.X0.T)),self.X0),self.Je)
        return dJedx0_s

    def enasa_diag(self):
        dJedx0_s = np.dot(np.dot(np.eye(self.X0.shape[0])/np.diag(np.dot(self.X0,self.X0.T)),self.X0),self.Je)
        return dJedx0_s

    def enasa_psd(self):
        try:
            u, s, vt = la.svd(self.X0)
        except la.LinAlgError:
            dJedx0_s = np.full(self.nx,np.nan)
        else:
            nrank = np.sum(s>s[0]*1.0e-10)
            ut = u.transpose()
            pinv = u[:,:nrank] @ np.diag(1.0/s[:nrank]/s[:nrank]) @ ut[:nrank,:]
            dJedx0_s = np.dot(np.dot(pinv,self.X0),self.Je)
        return dJedx0_s

    def enasa_pcr(self,n_components=None,cthres=None):
        # n_components: Number of components to keep. if n_components is not set all components are kept:
        # n_components == min(n_samples, n_features)
        # If n_components == 'mle' and svd_solver == 'full', Minka’s MLE is used to guess the dimension. Use of n_components == 'mle' will interpret svd_solver == 'auto' as svd_solver == 'full'.
        # If 0 < n_components < 1 and svd_solver == 'full', select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by n_components.
        # If svd_solver == 'arpack', the number of components must be strictly less than the minimum of n_features and n_samples.
        # Hence, the None case results in:
        # n_components == min(n_samples, n_features) - 1
        #
        if n_components is None and cthres is not None:
            nc = cthres
            svd_solver = 'full'
        else:
            nc = n_components
            svd_solver = 'auto'
        self.pcr = make_pipeline(StandardScaler(),PCA(n_components=nc,svd_solver=svd_solver), LinearRegression()).fit(self.X,self.Je)
        self.std = self.pcr.named_steps["standardscaler"]
        self.pca = self.pcr.named_steps["pca"]
        self.reg = self.pcr.named_steps["linearregression"]
        dJedx0_s = self.pca.inverse_transform(self.reg.coef_[None,:])[0,]
        contrib = self.pca.explained_variance_ratio_
        self.logger.info(f"nrank {self.pca.n_components_} contrib {np.sum(contrib)*1e2:.2f}%")
        return dJedx0_s

    def enasa_ridge(self,mu=0.01):
        dJedx0_s = np.dot(np.dot(np.linalg.inv(np.dot(self.X0,self.X0.T)+mu*np.eye(self.X0.shape[0])),self.X0),self.Je)
        return dJedx0_s

    def enasa_pls(self,n_components=None,cthres=None):
        # n_components: number of iteration
        if n_components is None:
            n_components = 2
        self.pls = PLSRegression(n_components=n_components,copy=True).fit(self.X,self.Je)
        #self.pls = PLSCanonical(n_components=1).fit(self.X,self.Je)
        dJedx0_s = self.pls.coef_[0,:]
        return dJedx0_s