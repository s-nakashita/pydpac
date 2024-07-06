import numpy as np 

# Adaptive determination of prior multiplicative inflation
# References: 
#   Wang and Bishop (2003, JAS)
#   Li et al. (2009, QJRMS)
class infl_adap():

    def __init__(self,sigb=0.04):
        self.sigb = sigb # 'observation' error
        self.asave = [] # save parameters

    def __call__(self,apre,do,Y):
        # do: normalized innovation = R^{-1/2}(y-H(xb))
        # Y : normalized ensemble perturbations in obs space = R^{-1/2}HXb
        parm = np.zeros(3)
        parm[0] = np.dot(do,do)
        parm[1] = np.sum(Y*Y)
        parm[2] = do.size
        aest = (parm[0]- parm[2]) / parm[1]
        sig2o = 2.0 / parm[2] * ((apre*parm[1] + parm[2])/parm[1])**2
        gain = self.sigb*self.sigb / (self.sigb*self.sigb + sig2o)
        anow = apre + gain*(aest - apre)
        # update
        self.asave.append(anow)
        return anow