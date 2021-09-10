import numpy as np
from scipy.linalg import eigh, norm, inv
import logging
from logging.config import fileConfig
fileConfig('logging_config.ini')
logger = logging.getLogger('anl')
from analysis.enkf import EnKF
from analysis.mlef import Mlef
from analysis.mlef_rloc import Mlef_rloc

## True forecast covariance
N = 100
d1 = 1
d2 = 8
Pt = np.eye(N)
for i in range(N):
    for j in range(N):
        Pt[i,j] = np.sqrt(i*j/N/N)*np.exp(-0.5*((i-j)/d1)**2) + np.sqrt((1.0-float(i)/N)*(1.0-float(j)/N))*np.exp(-0.5*((i-j)/d2)**2) 

eigs, eigv = eigh(Pt)
sPt = eigv @ np.diag(np.sqrt(eigs))
logger.info(sPt.shape)
logger.info(norm(Pt - sPt@sPt.transpose()))

## localization matrix
d1 = 1*3.0
d2 = 8*3.0
Ftilde = np.eye(N)
for i in range(N):
    for j in range(N):
        Ftilde[i,j] = np.sqrt(i*j/N/N)*np.exp(-0.5*((i-j)/d1)**2) + np.sqrt((1.0-float(i)/N)*(1.0-float(j)/N))*np.exp(-0.5*((i-j)/d2)**2) 
eigs, eigv = eigh(Ftilde)
neig = 1
thres = 0.85
frac = 0.0
while frac < thres:
    frac = eigs[N-neig:N].sum() / eigs.sum()
    neig += 1
logger.info(neig)
Wtilde = eigv[:,N-neig:N] @ np.diag(np.sqrt(eigs[N-neig:N]))
Flow = Wtilde @ Wtilde.transpose()
W = Wtilde
diagF = np.diag(Flow)
for i in range(W.shape[0]):
    W[i, :] /= np.sqrt(diagF[i])
F = W @ W.transpose()

def calc_dist(i):
    d1 = 1
    d2 = 8
    dist = np.zeros(N)
    for j in range(N):
        dist[j] = np.sqrt(i*j/N/N)*(np.fabs(i-j)/d1) + np.sqrt((1.0-float(i)/N)*(1.0-float(j)/N))*(np.fabs(i-j)/d2) 
    return dist

def calc_dist1(i,j):
    d1 = 1
    d2 = 8
    dist = np.sqrt(i*j/N/N)*(np.fabs(i-j)/d1) + np.sqrt((1.0-float(i)/N)*(1.0-float(j)/N))*(np.fabs(i-j)/d2) 
    return dist

## Observation operator and error covariance
class Obs():
    def __init__(self, operator, sigma,Pt):
        self.operator=operator
        self.sigma=sigma
        self.Pt = Pt
        logger.info(f"operator={self.operator}, sigma={self.sigma}")
    
    def get_op(self):
        return self.operator
    
    def get_sig(self):
        return self.sigma
        
    def set_r(self, N):
        from scipy.linalg import inv
        oberrstdev = self.sigma
        oberrvar = oberrstdev**2
        H = self.dh_operator(np.arange(N), np.zeros(N))
        HPHt = H @ self.Pt @ H.transpose()
        R = oberrvar * np.diag(np.diag(HPHt))
        Rsqrt = oberrstdev * np.diag(np.sqrt(np.diag(HPHt)))
        Rsqrtinv = inv(Rsqrt)
        Rinv = Rsqrtinv.transpose() @ Rsqrtinv
        return R, Rsqrtinv, Rinv
    
    def dh_operator(self, obsloc, x):
        p = obsloc.size
        n = x.size
        H = np.zeros((p,n))
        logger.debug(f"H={H.shape}")
        smooth_len = 4.0
        for j in range(p):
            for i in range(n):
                rr = float(i)-obsloc[j]
                r = np.fabs(rr) / smooth_len
                H[j,i] = np.exp(-r**2)
            H[j,:] = H[j,:]/H[j,:].sum()
        return H
    
    def h_operator(self,obsloc,x):
        if x.ndim > 1:
            hx = self.dh_operator(obsloc,x[:,0]) @ x
        else:
            hx = self.dh_operator(obsloc,x) @ x
        logger.debug(f"hx={hx.shape}")
        return hx

oberrstdev = 1.0 / 8.0
obs = Obs('vint', oberrstdev,Pt)

p = N #observed all grid
H = obs.dh_operator(np.arange(N), np.zeros(N))
oberrvar = obs.get_sig()**2
HPHt = H @ Pt @ H.transpose()
R, Rsqrtinv, Rinv = obs.set_r(N)
logger.info(f"oberr={oberrvar}, R={R.shape}, Rsqrtinv={Rsqrtinv.shape}, Rinv={Rinv.shape}")
Rsqrt = inv(Rsqrtinv)
logger.info(f"R - Rsqrt**2 ={norm(R - Rsqrt@Rsqrt)}")
logger.info(f"Rinv - Rsqrtinv**2 ={norm(Rinv - Rsqrtinv@Rsqrtinv)}")

## Varidation
def mse(Pa_app):
    diff = np.zeros_like(Pa_app)
    for j in range(N):
        for i in range(N):
            diff[j,i] = (Pa_app[j,i] - Pat[j,i])**2
    diff = F * diff
    mse = diff.sum()
    mse /= N**2
    return mse

def corr(Pa_app):
    tmp = Pa_app**2
    var1 = tmp.sum()
    tmp = Pat**2
    var2 = tmp.sum()
    tmp = Pa_app * Pat
    cov = tmp.sum()
    corr = cov / np.sqrt(var1) / np.sqrt(var2)
    return corr

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
## Lists for storing results and counter
    xrmse_mean = []
    mse_mean = []
    corr_mean = []
    ntest = 0
    ntest_max = 50
    ## Random seed for obs and forecast
    rng = np.random.default_rng(517)
    seeds = rng.choice(np.arange(10000), size=ntest_max, replace=False)
    logger.info(f"Random seeds for obs = {seeds}")
### Test
    while ntest < ntest_max:
        logger.info(f"{ntest}th test")
    ## Random seed
        rs = np.random.RandomState(seeds[ntest]) #variable
        rstrue = np.random.RandomState(514) #fix

    ## True state and observation
        xt = sPt @ rstrue.standard_normal(size=N)
        y = obs.dh_operator(np.arange(p), xt) @ xt + Rsqrt @ rs.standard_normal(size=p)

    ## Forecast ensemble
        K = 50
        Xf = rs.standard_normal(size=(N,K))
        Xf = sPt @ Xf
        Xf = Xf - Xf.mean(axis=1)[:, None]
        Pe = Xf @ Xf.transpose() / (K-1)
        err = norm(Pt - Pe)
        logger.info(f"ensemble approximation of Pf : {err}")

        Ploc = Pe * F
        err = norm(Pt - Ploc)
        logger.info(f"localization of Pf : {err}")

    # modulated ensemble
        L = W.shape[1]
        M = K*L
        Xfm = np.empty((N, M), Xf.dtype)
        for l in range(L):
            for k in range(K):
                m = l*K + k
                Xfm[:, m] = W[:, l] * Xf[:, k]
        Xfm *= np.sqrt((M-1)/(K-1))
        Pmod = Xfm @ Xfm.transpose() / (M-1)
        err = norm(Pt - Pmod)
        logger.info(f"modulation ensemble approximation of Pf : {err}")

    # forecast ensemble
    ## mean
        xf_ = sPt @ rs.standard_normal(size=N)
    ## member
        xf = xf_[:, None] + Xf
        logger.info(f"xf.shape={xf.shape}")
        initial_mean_err = np.sqrt(((xf.mean(axis=1) - xt)**2).mean())
        logger.info(f"initial error (mean) ={initial_mean_err}")

    # compute true analysis error covariance for modulated ensemble
        Htilde = Rsqrtinv @ obs.dh_operator(np.arange(p), xf_)
        Z = Xfm / np.sqrt(M-1)
        HZ = Htilde @ Z
        A = HZ.transpose() @ HZ
        eigs, eigv = eigh(A)
        Dsqrt = np.sqrt(1.0/(1.0+eigs))
        T = eigv @ np.diag(Dsqrt) @ eigv.transpose()
        Za = Z @ T
        Kmat = Za @ Za.transpose() @ Htilde.transpose()
        KHP = Kmat @ Htilde @ Pt
        Pat = Pt - KHP - KHP.transpose() + Kmat@(Htilde@Pt@Htilde.transpose()+np.eye(p))@Kmat.transpose()
        Pam = Za @ Za.transpose()

    ## EnKF
        params = {'etkf':('etkf',None,False,False),'etkf-ss':('etkf',2,True,False),'etkf-rg':('etkf',2,False,True),'etkf-kloc':('etkf',0,False,False),
                'po':('po',None,False,False),'po-mod':('po',2,False,False),'po-kloc':('po',0,False,False),
                'srf':('srf',None,False,False),'srf-ss':('srf',2,True,False),'srf-rg':('srf',2,False,True),'srf-kloc':('srf',0,False,False),
                'letkf':('letkf',0,False,False)}
        names = ['mod','etkf','etkf-ss','etkf-rg','etkf-kloc','po','po-mod','po-kloc','srf','srf-ss','srf-rg','srf-kloc','letkf']
        xa_list = []
        Pa_app_list = [Pam]
        for ptype in names[1:]:
            pt, iloc, ss, getkf = params[ptype]
            analysis = EnKF(pt, N, K, obs, iloc=iloc, lsig=3.0, ss=ss, getkf=getkf, l_mat=F, l_sqrt=W, calc_dist=calc_dist, calc_dist1=calc_dist1)
            xb = xf
            pb = Pe
            xa, Pa, sPa, innv, chi2, ds = analysis(xb, pb, y[::-1], np.arange(p-1,-1,-1))
            xa_list.append(xa.mean(axis=1))
            Pa_app_list.append(Pa)

    ### MLEF
    # forecast ensemble
    ## control
        xfc = xf_
    ## member
        xfe = xfc[:, None] + Xf / np.sqrt(K-1)
        xf = np.zeros((N,K+1))
        xf[:,0] = xfc
        xf[:,1:] = xfe
        logger.info(f"xf.shape={xf.shape}")
        initial_ctrl_err = np.sqrt(((xfc - xt)**2).mean())
        logger.info(f"initial error (control) ={initial_ctrl_err}")

        params = {'mlef':('mlef',None,False,False),'mlef-ss':('mlef',2,True,False),'mlef-rg':('mlef',2,False,True),'mlef-rloc':('mlef',0,False,False)}
        names2 = ['mlef','mlef-ss','mlef-rg','mlef-rloc']
        for ptype in names2:
            pt, iloc, ss, gain = params[ptype]
            if ptype != 'mlef-rloc':
                analysis = Mlef(pt, N, K, obs, iloc=iloc, lsig=3.0, ss=ss, gain=gain, l_mat=F, l_sqrt=W, calc_dist=calc_dist, calc_dist1=calc_dist1)
            else:
                analysis = Mlef_rloc(pt, K, obs, lsig=3.0, calc_dist=calc_dist, calc_dist1=calc_dist1)
            xb = xf
            pb = Pe
            xa, Pa, sPa, innv, chi2, ds = analysis(xb, pb, y[::-1], np.arange(p-1,-1,-1))
            xa_list.append(xa[:,0])
            Pa_app_list.append(Pa)

        method = names + names2
        #logger.info(names)
        xrmse = [0.0]
        mse_list = []
        corr_list = []
        i = 1
        for xam in xa_list:
            #logger.info(xam)
            if i < len(names):
                #logger.info(f"method:{method[i]} mean")
                xrmse.append(np.sqrt(((xam - xt)**2).mean())/initial_mean_err)
            else:
                #logger.info(f"method:{method[i]} ctrl")
                xrmse.append(np.sqrt(((xam - xt)**2).mean())/initial_ctrl_err)
            i += 1
        logger.info(f"initial error : mean={initial_mean_err}, ctrl={initial_ctrl_err} obs error:{obs.get_sig()}")
        logger.info(method)
        logger.info(xrmse)
        for Pa_app in Pa_app_list:
            mse_list.append(mse(Pa_app))
            corr_list.append(corr(Pa_app))
        logger.info(mse_list)
        logger.info(corr_list)
        xrmse_mean.append(xrmse)
        mse_mean.append(mse_list)
        corr_mean.append(corr_list)
        ntest += 1

    ## Check results
    logger.info(f"Number of tests : {ntest}, obs error:{obs.get_sig()}")
    logger.info(np.array(mse_mean).shape)
    rmse_m = np.array(xrmse_mean).mean(axis=0)
    rmse_s = np.sqrt(((np.array(xrmse_mean) - rmse_m[None,:])**2).sum(axis=0)/(ntest-1))
    mse_m = np.array(mse_mean).mean(axis=0)
    mse_s = np.sqrt(((np.array(mse_mean) - mse_m[None,:])**2).sum(axis=0)/(ntest-1))
    corr_m = np.array(corr_mean).mean(axis=0)
    corr_s = np.sqrt(((np.array(corr_mean) - corr_m[None,:])**2).sum(axis=0)/(ntest-1))
    for i in range(len(method)):
        logger.info("{:9} {:5.3e}({:5.3e}) {:5.3e}({:5.3e}) {:5.3e}({:5.3e})"
          .format(method[i], rmse_m[i], rmse_s[i], mse_m[i], mse_s[i], corr_m[i], corr_s[i]))