import numpy as np
from numpy.random import default_rng
from numpy import linalg as la
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
import matplotlib.pyplot as plt
import logging

class EnASA():
    def __init__(self,vt,X0,Je,esatype='minnorm',logfile='minnorm',seed=None):
        self.esatype = esatype
        self.vt = vt
        self.X0 = X0 #centered
        self.Je = Je #centered
        self.X = X0.T # (n_samples, n_features)
        self.nx = self.X.shape[1]
        self.nens = self.X.shape[0]
        self.logger = logging.getLogger(__name__)
        self.logfile = logfile
        logging.basicConfig(filename=f'{self.logfile}.log', encoding='utf-8', level=logging.INFO)
        self.rng = default_rng(seed=seed)
        
    def __call__(self,nrank=None,cthres=None,mu=0.01,n_components=None,vip=False,threshold=1.0,alpha=1.0,l1_ratio=0.5,a=1.0,b=0.01,cv=False):
        if self.esatype=='minnorm':
            dJedx0_s = self.enasa_minnorm(nrank=nrank,cthres=cthres)
        elif self.esatype=='minvar':
            dJedx0_s = self.enasa_minvar()
        elif self.esatype=='diag':
            dJedx0_s = self.enasa_diag()
        elif self.esatype=='psd':
            dJedx0_s = self.enasa_psd()
        elif self.esatype=='ridge':
            dJedx0_s = self.enasa_ridge(mu=mu) #,cv=cv)
        elif self.esatype=='pcr':
            dJedx0_s = self.enasa_pcr(n_components=n_components,cthres=cthres)
        elif self.esatype=='pls':
            dJedx0_s = self.enasa_pls(n_components=n_components)
        elif self.esatype=='pls_vip':
            dJedx0_s = self.enasa_pls(n_components=n_components,vip=True,threshold=threshold)
        elif self.esatype=='lasso':
            dJedx0_s = self.enasa_lasso(alpha=alpha,cv=cv)
        elif self.esatype=='elnet':
            dJedx0_s = self.enasa_elnet(alpha=alpha,l1_ratio=l1_ratio,a=a,b=b,cv=cv)
        #print(f"dJedx0_s.shape={dJedx0_s.shape}")
        self.rescaling(dJedx0_s)
        
        return self.dJedx0

    def rescaling(self,dJedx0_s):
        if self.esatype == 'pcr':
            #self.dJedx0 = dJedx0_s.copy()
            self.dJedx0 = dJedx0_s / self.std.scale_
            self.err = self.reg.intercept_ - np.dot(self.std.mean_, dJedx0_s)
        elif self.esatype == 'pls':
            self.dJedx0 = dJedx0_s.copy()
            #self.dJedx0 = dJedx0_s / self.pls._x_std
            self.err = self.pls.intercept_ - np.dot(self.pls._x_mean, dJedx0_s)
        elif self.esatype == 'pls_vip':
            self.dJedx0 = np.zeros(self.nx)
            self.dJedx0[self.val_sel] = dJedx0_s[:]
            #self.dJedx0 = dJedx0_s / self.pls._x_std
            self.err = self.pls.intercept_ - np.dot(self.pls._x_mean, dJedx0_s)
        else:
            self.dJedx0 = dJedx0_s.copy()
            self.err = 0.0

    def estimate(self,X=None,beta=None):
        if X is None:
            X = self.X
        if beta is None:
            beta = self.dJedx0
        #if self.esatype == 'pcr':
        #    Je_est1 = self.pcr.predict(self.X)
        if self.esatype == 'pls':
            Je_est = self.pls.predict(X)
        elif self.esatype == 'pls_vip':
            Je_est = self.pls.predict(self.Xsel)
        else:
            Je_est = np.dot(X,beta) + self.err
        return Je_est.ravel()

    def score(self):
        if self.esatype == 'pls':
            return self.pls.score(self.X,self.Je)
        elif self.esatype == 'pls_vip':
            return self.pls.score(self.Xsel,self.Je)
        #elif self.esatype == 'pcr':
        #    print(self.pcr.score(self.X,self.Je))
        else:
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
        # If n_components == 'mle' and svd_esatype == 'full', Minka’s MLE is used to guess the dimension. Use of n_components == 'mle' will interpret svd_esatype == 'auto' as svd_esatype == 'full'.
        # If 0 < n_components < 1 and svd_esatype == 'full', select the number of components such that the amount of variance that needs to be explained is greater than the percentage specified by n_components.
        # If svd_esatype == 'arpack', the number of components must be strictly less than the minimum of n_features and n_samples.
        # Hence, the None case results in:
        # n_components == min(n_samples, n_features) - 1
        #
        if n_components is None and cthres is not None:
            nc = cthres
            svd_esatype = 'full'
        else:
            nc = n_components
            svd_esatype = 'auto'
        self.pcr = make_pipeline(StandardScaler(),PCA(n_components=nc,svd_esatype=svd_esatype), LinearRegression()).fit(self.X,self.Je)
        self.std = self.pcr.named_steps["standardscaler"]
        self.pca = self.pcr.named_steps["pca"]
        self.reg = self.pcr.named_steps["linearregression"]
        dJedx0_s = self.pca.inverse_transform(self.reg.coef_[None,:])[0,]
        contrib = self.pca.explained_variance_ratio_
        self.logger.info(f"nrank {self.pca.n_components_} contrib {np.sum(contrib)*1e2:.2f}%")
        return dJedx0_s

    def enasa_ridge(self,X=None,Y=None,mu=0.01,alphas=(0.001,0.01,0.1,1.0,10.0),cv=False):
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Je
        self.alpha = mu / X.shape[0]
        if cv:
            self.ridge = RidgeCV(alphas=alphas,store_cv_results=True).fit(X,Y)
            self.alphas = alphas
            self.alpha = self.ridge.alpha_
        else:
            self.ridge = Ridge(alpha=self.alpha,copy_X=True).fit(X,Y)
        dJedx0_s = self.ridge.coef_
        #dJedx0_s = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)+mu*np.eye(X.shape[1])),X.T),Y)
        return dJedx0_s

    def enasa_pls(self,X=None,Y=None,n_components=None,vip=False,threshold=1.0):
        # n_components: number of iteration
        if n_components is None:
            n_components = 2
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Je
        self.pls = PLSRegression(n_components=n_components,copy=True).fit(X,Y)
        #self.pls = PLSCanonical(n_components=1).fit(self.X,self.Je)
        if vip:
            # evaluate variable importance for projection score
            vip = np.zeros(self.nx)
            ssyr = (self.pls.y_loadings_[0,:]**2)*np.diag(self.pls.y_scores_.T@self.pls.y_scores_)
            ssytotal = np.sum(ssyr)
            wgts = self.pls.x_weights_/la.norm(self.pls.x_weights_,ord=2,axis=0)
            for i in range(self.nx):
                vip[i] = vip[i] + np.sum(ssyr*wgts[i,:]*wgts[i,:])/ssytotal
            self.vip = np.sqrt(vip*self.nx)
            self.val_sel = np.arange(self.nx)[self.vip > threshold]
            self.Xsel = X[:,self.val_sel]
            self.pls = PLSRegression(n_components=n_components,copy=True).fit(self.Xsel,Y)
        dJedx0_s = self.pls.coef_[0,:]
        return dJedx0_s

    def enasa_lasso(self,X=None,Y=None,alpha=1.0,cv=False):
        # alpha: regularization strength
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Je
        if cv:
            self.lasso = LassoCV(copy_X=True).fit(X,Y)
            self.alpha = self.lasso.alpha_
        else:
            self.alpha = alpha
            self.lasso = Lasso(alpha=self.alpha, copy_X=True).fit(X,Y)
        dJedx0_s = self.lasso.coef_
        return dJedx0_s
    
    def enasa_elnet(self,X=None,Y=None,alpha=None,l1_ratio=None,a=1.0,b=0.01,cv=False):
        # a: coefficient for L1 norm
        # b: coefficient for L2 norm
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Je
        if cv:
            self.l1_ratios = [.1,.5,.75,.9,.95,.99]
            self.elnet = ElasticNetCV(l1_ratio=self.l1_ratios,copy_X=True).fit(X,Y)
            self.alpha = self.elnet.alpha_
            self.l1_ratio = self.elnet.l1_ratio_
        else:
            if alpha is not None and l1_ratio is not None:
                self.alpha = alpha
                self.l1_ratio = l1_ratio
            else:
                self.alpha = a + 2.0*b
                self.l1_ratio = a / alpha
            self.elnet = ElasticNet(alpha=self.alpha,l1_ratio=self.l1_ratio,copy_X=True).fit(X,Y)
        dJedx0_s = self.elnet.coef_
        return dJedx0_s

    def calc_dxeopt(self):
        # optimal initial perturbations for maximizing \Delta J 
        # (Duc et al. 2023; Enomoto et al. 2015)
        # \lambda = -1 / stdJ for EYO15
        # \rho = \|cor(X0,Je)\|_{C^{-1}} for DHK23
        stdJ = np.std(self.Je)
        cov = np.dot(self.X0,self.Je)/(self.nens-1)
        return -cov / stdJ

    def cv(self,params,K=None):
        #cross-validation for determining hyperparameters
        # input:
        #   params: parameter list
        #   K:      number of data decomposition
        # output:
        #   popt:  optimal parameter
        #   press: prediction residual sum of squares

        if K is None:
            K = self.nens # leave-one-out
        
        all_index = [i for i in range(self.nens)]
        nparams = len(params)
        ntest = self.nens // K
        result = np.zeros((K,nparams))

        # random shuffle of data
        Y = self.Je.reshape(-1,1)
        Z = np.hstack((self.X,Y))
        self.rng.shuffle(Z, axis=0)
        X = Z[:,0:self.nx]
        Y = Z[:,self.nx:self.nx+1]

        for i, p in enumerate(params):
            for k in range(K):
                if k < K-1:
                    val_index = all_index[k*ntest:(k+1)*ntest]
                else:
                    val_index = all_index[k*ntest:]
                train_index = [j for j in all_index if not j in val_index]
                X_train = X[train_index,:]
                Y_train = Y[train_index,:]
                X_val   = X[val_index,:]
                Y_val   = Y[val_index,:]
                if i==0 and k==0:
                    print(f"train_index={train_index}, val_index={val_index}")
                    print(f"X_train={X_train.shape}, Y_train={Y_train.shape}, X_val={X_val.shape}, Y_val={Y_val.shape}")
                #if self.esatype=='ridge':
                #    beta_tmp = self.enasa_ridge(X=X_train,Y=Y_train,mu=p)
                if self.esatype=='pls':
                    beta_tmp = self.enasa_pls(X=X_train,Y=Y_train,n_components=p)
                beta = self.rescaling(beta_tmp)
                # validation
                Y_hat = self.estimate(X=X_val,beta=beta)
                result[k,i] = np.sum((Y_val - Y_hat)**2)
        press = np.sum(result,axis=0)
        popt = params[np.argmin(press)]

        return popt, press

    def check_cv(self,figdir='.'):
        if self.esatype=='ridge':
            fig, ax = plt.subplots()
            ax.plot(self.alphas,self.ridge.cv_results_.mean(axis=0))
            ax.vlines([self.alpha],0,1,colors='r',ls='dashed',transform=ax.get_xaxis_transform())
            ax.set_title(f'optimal alpha={self.alpha:.2f}')
            fig.savefig(figdir+f'/cv_ridge_vt{self.vt}ne{self.nens}.png')
        elif self.esatype=='lasso':
            fig, ax = plt.subplots()
            ax.plot(self.lasso.alphas_,self.lasso.mse_path_.mean(axis=1))
            ax.vlines([self.alpha],0,1,colors='r',ls='dashed',transform=ax.get_xaxis_transform())
            ax.set_title(f'optimal alpha={self.alpha:.2f}')
            fig.savefig(figdir+f'/cv_lasso_vt{self.vt}ne{self.nens}.png')
        elif self.esatype=='elnet':
            fig, ax = plt.subplots()
            n_l1_ratio, n_alpha, n_folds = self.elnet.mse_path_.shape
            cmap = plt.get_cmap('tab10')
            for i in range(n_l1_ratio):
                ax.plot(self.elnet.alphas_[i,:],self.elnet.mse_path_[i,:,:].mean(axis=1),c=cmap(i),label=f'l1_ratio={self.l1_ratios[i]:.2f}')
            ax.vlines(self.elnet.alpha_,0,1,colors=cmap(range(n_l1_ratio)),ls='dashed',transform=ax.get_xaxis_transform())
            ax.legend()
            ax.set_title(f'optimal parameters: l1_ratio={self.l1_ratio:.2f}, alpha={self.alpha:.2f}')
            fig.savefig(figdir+f'/cv_elnet_vt{self.vt}ne{self.nens}.png')
        else:
            print(f"invalid EnASA type: {self.esatype}")
