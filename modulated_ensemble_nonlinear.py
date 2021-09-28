import sys
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
    def __init__(self, operator, sigma,Pt,sPt,nl=4):
        self.operator=operator
        self.sigma=sigma
        self.Pt = Pt
        self.sPt = sPt
        self.sb = 1. # like Stephan-Boltzman constant
        self.nl = nl # nonlinearity
        logger.info(f"operator={self.operator}, sigma={self.sigma}")
        logger.info(f"Stephan-Boltzman constant={self.sb}, Nonlinearity={self.nl}")
    
    def get_op(self):
        return self.operator
    
    def get_sig(self):
        return self.sigma
        
    def set_r(self, obsloc):
        from scipy.linalg import inv
        oberrstdev = self.sigma
        oberrvar = oberrstdev**2
        #HsP = self.h_operator(np.arange(p), self.sPt)
        #HPHt = HsP @ HsP.transpose()
        n = self.Pt.shape[0]
        if obsloc.size <= n:
            H = self.itpl_operator(obsloc, self.Pt.shape[0])
            HPHt = H @ self.Pt @ H.transpose()
            diagR = oberrvar * np.diag(HPHt)
        else:
            obsloc1 = obsloc[:n]
            p2 = obsloc.size - obsloc1.size
            H = self.itpl_operator(obsloc1, self.Pt.shape[0])
            HPHt = H @ self.Pt @ H.transpose()
            diagR1 = oberrvar * np.diag(HPHt)
            diagR2 = oberrvar * np.ones(p2)
            diagR = np.hstack((diagR1,diagR2))
        R = np.diag(diagR)
        Rsqrt = np.diag(np.sqrt(diagR))
        #R = oberrvar * np.diag(np.diag(HPHt))
        #Rsqrt = oberrstdev * np.diag(np.sqrt(np.diag(HPHt)))
        Rsqrtinv = inv(Rsqrt)
        Rinv = Rsqrtinv.transpose() @ Rsqrtinv
        return R, Rsqrtinv, Rinv
    # vertical integration
    def itpl_operator(self, obsloc, n):
        p = obsloc.size
        H = np.zeros((p,n))
        logger.debug(f"H1={H.shape}")
        smooth_len = 4.0
        for j in range(p):
            for i in range(n):
                rr = float(i)-obsloc[j]
                r = np.fabs(rr) / smooth_len
                H[j,i] = np.exp(-r**2)
            H[j,:] = H[j,:]/H[j,:].sum()
        return H
    # vertical interpolation
    def itpl_operator2(self, obsloc, n):
        p = obsloc.size
        H = np.zeros((p,n))
        logger.debug(f"H2={H.shape}")
        for k in range(p):
            ri = obsloc[k]
            i = math.floor(ri)
            ai = ri - float(i)
            if i == n-1:
                H[k,i] = 1.0
            else:
                H[k,i] = 1.0 - ai
                H[k,i+1] = ai
        return H
    
    def dh_operator(self, obsloc, x):
        p = obsloc.size
        n = x.size
        if p <= n:
            H = self.itpl_operator(obsloc, n) @ np.diag(self.sb*self.nl*(x**(self.nl-1)))
        else:
            obsloc1 = obsloc[:n]
            obsloc2 = obsloc[n:]
        #x_itpl = self.itpl_operator(obsloc, n) @ x
            H1 = self.itpl_operator(obsloc1, n) @ np.diag(self.sb*self.nl*(x**(self.nl-1)))
            H2 = self.itpl_operator2(obsloc2, n)
            H = np.vstack((H1,H2))
        logger.debug(f"H={H.shape}")
        return H
    
    def h_operator(self,obsloc,x):
        p = obsloc.size
        if x.ndim > 1:
            n = x.shape[0]
        else:
            n = x.size
        if p <= n:
            hx = self.itpl_operator(obsloc, n) @ (self.sb*(x**self.nl))
        else:
            obsloc1 = obsloc[:n]
            obsloc2 = obsloc[n:]
        #x_itpl = self.itpl_operator(obsloc, n) @ x
            hx1 = self.itpl_operator(obsloc1, n) @ (self.sb*(x**self.nl))
            hx2 = self.itpl_operator2(obsloc2, n) @ x
            if x.ndim > 1:
                hx = np.vstack((hx1,hx2))
            else:
                hx = np.hstack((hx1,hx2))
        logger.debug(f"hx={hx.shape}")
        return hx
ntest_max = 10
nl = 4
oberrstdev = 0.125
if len(sys.argv) > 1:
    ntest_max = int(sys.argv[1])
if len(sys.argv) > 2:
    nl = int(sys.argv[2])
if len(sys.argv) > 3:
    oberrstdev = float(sys.argv[3])
obs = Obs('vint', oberrstdev, Pt, sPt,nl=nl)
p = N #observed all grid
H = obs.itpl_operator(np.arange(p), N)
vindex = np.arange(1,p+1)

oberrvar = obs.get_sig()**2

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
#
## Lists for storing results and counter
    initial_mean = 0.0
    initial_ctrl = 0.0
    initial_mean_obs = 0.0
    initial_ctrl_obs = 0.0
    xrmse_mean = []
    hxrmse_mean = []
    ntest = 0
    ## Random seed for obs and forecast
    rng = np.random.default_rng(517)
    seeds = rng.choice(np.arange(10000), size=ntest_max, replace=False)
    logger.info(f"Random seeds for obs = {seeds}")
#
### Test
    while ntest < ntest_max:
        logger.info(f"{ntest}th test")
        ## Random seed
        rs = np.random.RandomState(seeds[ntest]) #variable
        rstrue = np.random.RandomState(514) #fix

        ## True state and observation
        xt = sPt @ rstrue.standard_normal(size=N)
        obsloc1 = np.arange(p) # upward
        obsloc1 = np.arange(p-1,-1,-1) # downward
        obsloc1 = rs.choice(p, size=p, replace=False) # random
        obsloc = obsloc1
        #obsloc2 = np.arange(0,p-1,10)
        #obsloc = np.hstack((obsloc1, obsloc2))
        print(obsloc)
        hxt = obs.h_operator(obsloc, xt)
        R, Rsqrtinv, Rinv = obs.set_r(obsloc)
        logger.info(f"oberr={oberrvar}, R={R.shape}, Rsqrtinv={Rsqrtinv.shape}, Rinv={Rinv.shape}")
        Rsqrt = inv(Rsqrtinv)
        logger.info(f"R - Rsqrt**2 ={norm(R - Rsqrt@Rsqrt)}")
        logger.info(f"Rinv - Rsqrtinv**2 ={norm(Rinv - Rsqrtinv@Rsqrtinv)}")
        y = hxt + Rsqrt @ rs.standard_normal(size=obsloc.size)

        ## Ensemble Pf 
        K = 50
        Xf = rs.standard_normal(size=(N,K))
        Xf = sPt @ Xf
        Xf = Xf - Xf.mean(axis=1)[:, None]
        Pe = Xf @ Xf.transpose() / (K-1)
        err = norm(Pt - Pe)
        logger.info(f"ensemble approximation of Pf : {err}")
#
        # direct localization
        Ploc = Pe * F
        err = norm(Pt - Ploc)
        logger.info(f"localization of Pf : {err}")
#
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
#
        ### EnKF
        # forecast ensemble
        ## mean
        xf_ = sPt @ rs.standard_normal(size=N)
        ## member
        xf = xf_[:, None] + Xf
        logger.info(f"xf.shape={xf.shape}")
        initial_mean_err = np.sqrt(((xf.mean(axis=1) - xt)**2).mean())
        logger.info(f"initial error (mean) ={initial_mean_err}")
        hxf = obs.h_operator(obsloc, xf)
        initial_mean_obserr = np.sqrt(((hxf.mean(axis=1) - hxt)**2).mean())
        logger.info(f"initial error in observation space (mean) ={initial_mean_obserr}")
#
        params = {
                'enkf':('etkf',None,False,False),'enkf-b':('etkf',2,False,True),'enkf-k':('etkf',0,False,False),'letkf':('letkf',0,False,False),
                #'po':('po',None,False,False),'po-b':('po',2,False,False),'po-k':('po',0,False,False),
                'serial enkf':('srf',None,False,False),'serial enkf-b':('srf',2,False,True),'serial enkf-k':('srf',0,False,False),
                }
        names = ['enkf','enkf-b','enkf-k','letkf','serial enkf','serial enkf-b','serial enkf-k']
        #names = ['serial enkf','serial enkf-b','serial enkf-k']
        xa_list = []
#        for ptype in names:
#            pt, iloc, ss, getkf = params[ptype]
#            analysis = EnKF(pt, N, K, obs, iloc=iloc, lsig=3.0, ss=ss, getkf=getkf, l_mat=F, l_sqrt=W, calc_dist=calc_dist, calc_dist1=calc_dist1)
#            xb = xf
#            pb = Pe
#            xa, Pa, sPa, innv, chi2, ds = analysis(xb, pb, y, obsloc)
#            xa_list.append(xa)
#
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
        hxf = obs.h_operator(obsloc, xfc)
        initial_ctrl_obserr = np.sqrt(((hxf - hxt)**2).mean())
        logger.info(f"initial error in obs space (control) ={initial_ctrl_obserr}")
#
        params = {'mlef':('mlef',None,False,False),'mlef-b':('mlef',2,False,True),'mlef-r':('mlef',0,False,False)}
        #names2 = ['mlef','mlef-b','mlef-r']
        names2 = ['mlef-r']
        for ptype in names2:
            pt, iloc, ss, gain = params[ptype]
            if ptype != 'mlef-r':
                analysis = Mlef(pt, N, K, obs, iloc=iloc, lsig=3.0, ss=ss, gain=gain, l_mat=F, l_sqrt=W, calc_dist=calc_dist, calc_dist1=calc_dist1)
#                       ,incremental=True)
            else:
                analysis = Mlef_rloc(pt, K, obs, lsig=3.0, calc_dist=calc_dist, calc_dist1=calc_dist1)
            xb = xf
            pb = Pe
            xa, Pa, sPa, innv, chi2, ds = analysis(xb, pb, y, obsloc, method='LBFGS') #, restart=True)
            xa_list.append(xa[:,0])

        #method = names + names2
        method = names2
        #logger.info(names)
        xrmse = []
        hxrmse = []
        i = 0
        for xa in xa_list:
            #logger.info(xam)
            if i < 0:
            #if i < len(names):
                #logger.info(f"method:{method[i]} mean")
                xrmse.append(np.sqrt(((xa.mean(axis=1) - xt)**2).mean())/initial_mean_err)
                hxa = obs.h_operator(obsloc, xa.mean(axis=1))
                hxrmse.append(np.sqrt(((hxa - hxt)**2).mean())/initial_mean_obserr)
            else:
                #logger.info(f"method:{method[i]} ctrl")
                xrmse.append(np.sqrt(((xa - xt)**2).mean())/initial_ctrl_err)
                hxa = obs.h_operator(obsloc, xa)
                hxrmse.append(np.sqrt(((hxa - hxt)**2).mean())/initial_ctrl_obserr)
            i += 1
        logger.info(f"initial error : mean={initial_mean_err}, ctrl={initial_ctrl_err}")
        logger.info(f"initial obs error : mean={initial_mean_obserr}, ctrl={initial_ctrl_obserr} obs error:{obs.get_sig()}")
        logger.info(method)
        logger.info(xrmse)
        logger.info(hxrmse)
        initial_mean += initial_mean_err
        initial_ctrl += initial_ctrl_err
        initial_mean_obs += initial_mean_obserr
        initial_ctrl_obs += initial_ctrl_obserr
        xrmse_mean.append(xrmse)
        hxrmse_mean.append(hxrmse)
        ntest += 1
#
    ## Check results
    logger.info(f"Number of tests : {ntest}, Nonlinearity : {obs.nl}, obs error:{obs.get_sig()}")
    ini_m = initial_mean / ntest
    ini_c = initial_ctrl / ntest
    ini_m_o = initial_mean_obs / ntest
    ini_c_o = initial_ctrl_obs / ntest
    logger.info("initial error average : mean{:5.3e} ctrl{:5.3e}".format(ini_m, ini_c))
    logger.info("initial error in obs space average : mean{:5.3e} ctrl{:5.3e}".format(ini_m_o, ini_c_o))
    rmse_m = np.array(xrmse_mean).mean(axis=0)
    rmse_s = np.sqrt(((np.array(xrmse_mean) - rmse_m[None,:])**2).sum(axis=0)/(ntest-1))
    hrmse_m = np.array(hxrmse_mean).mean(axis=0)
    hrmse_s = np.sqrt(((np.array(hxrmse_mean) - rmse_m[None,:])**2).sum(axis=0)/(ntest-1))
    for i in range(len(method)):
        logger.info("{:13} {:5.3e}({:5.3e}) {:5.3e}({:5.3e})"
          .format(method[i], rmse_m[i], rmse_s[i], hrmse_m[i], hrmse_s[i]))