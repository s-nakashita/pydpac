import numpy as np
from numpy import linalg as la
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt

class EnASA():
    def __init__(self,vt,X0,Je,solver='minnorm'):
        self.solver = solver
        self.vt = vt
        self.X0 = X0 #centered
        self.Je = Je #centered
        self.X = X0.T # (n_samples, n_features)
        self.nx = self.X.shape[1]
        self.nens = self.X.shape[0]
        
    def __call__(self,n_components=None,mu=0.01):
        if self.solver=='minnorm':
            dJedx0_s = self.enasa_minnorm(nrank=n_components)
        elif self.solver=='minvar':
            dJedx0_s = self.enasa_minvar()
        elif self.solver=='diag':
            dJedx0_s = self.enasa_diag()
        elif self.solver=='psd':
            dJedx0_s = self.enasa_psd()
        elif self.solver=='ridge':
            dJedx0_s = self.enasa_ridge(mu=mu)
        elif self.solver=='pcr':
            dJedx0_s = self.enasa_pcr(n_components=n_components)
        elif self.solver=='pls':
            dJedx0_s = self.enasa_pls(n_components=n_components)
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
        #    Je_est1 = self.pls.predict(self.X)
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

    def enasa_minnorm(self,nrank=None):
        try:
            u, s, vt = la.svd(self.X0)
        except la.LinAlgError:
            dJedx0_s = np.full(self.nx,np.nan)
        else:
            if nrank is None:
                self.nrank = np.sum(s>s[0]*1.0e-10)
            else:
                self.nrank = nrank
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

    def enasa_pcr(self,n_components=None):
        # n_components: number of PCA modes
        self.pcr = make_pipeline(StandardScaler(),PCA(n_components=n_components), LinearRegression()).fit(self.X,self.Je)
        self.std = self.pcr.named_steps["standardscaler"]
        self.pca = self.pcr.named_steps["pca"]
        self.reg = self.pcr.named_steps["linearregression"]
        dJedx0_s = self.pca.inverse_transform(self.reg.coef_[None,:])[0,]
        return dJedx0_s

    def enasa_ridge(self,mu=0.01):
        dJedx0_s = np.dot(np.dot(np.linalg.inv(np.dot(self.X0,self.X0.T)+mu*np.eye(self.X0.shape[0])),self.X0),self.Je)
        return dJedx0_s

    def enasa_pls(self,n_components=None):
        # n_components: number of PCA modes
        if n_components is None:
            n_components = min(self.nx,self.nens-1)
        self.pls = PLSRegression(n_components=n_components,copy=True).fit(self.X,self.Je)
        dJedx0_s = self.pls.coef_[0,:]
        return dJedx0_s