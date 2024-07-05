import numpy as np 
# 1-dimensional correlation functions
class Corrfunc():
    def __init__(self,l,a=0.5):
        self.l = l #correlation length
        self.a = a #shape parameter (gc5)
    
    def __call__(self,d,ftype="gauss"):
        if ftype == "gauss":
            return self.gauss(d)
        elif ftype == "gc5":
            return self.gc5(d)
        elif ftype == "tri":
            return self.tri(d)

    def gauss(self,d):
        sigma = self.l * np.sqrt(0.3)
        return np.exp(-0.5*(d/sigma)**2)
    
    def gc5(self,d):
        c = self.l
        r2 = np.zeros(2*d.size-1)
        r2[:d.size] = d[::-1]*-1.0
        r2[d.size-1:] = d[:]
        b2 = self._b0(r2,self.a,c)
        b1 = np.hstack((b2,b2))
        #plt.plot(r2,b2);plt.show();plt.close()
        c2 = np.convolve(b1,b2,mode='full')
        #plt.plot(c2);plt.show();plt.close()
        c = c2[r2.size-1:r2.size+d.size-1]
        return c/c[0]
    
    def _b0(self,r,a,c):
        return np.where(abs(r)<0.5*c,2*(a-1)*abs(r)/c+1,np.where(abs(r)<c,2*a*(1-abs(r)/c),0))

    def tri(self,d):
        nj = 1.0 / self.l * 2.0 * np.pi
        return np.where(d==0.0,1.0,np.sin(nj*d/2.0)/np.tan(d/2.0)/nj)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    l = np.deg2rad(30.0)
    corrfunc = Corrfunc(l)

    d = np.linspace(0,180,181)
    r = np.deg2rad(d)
    plt.plot(d,corrfunc(r,ftype="gauss"),label="gauss")
    plt.plot(d,corrfunc(r,ftype="gc5"),label="gc5,a=0.5")
    plt.plot(d,corrfunc(r,ftype="tri"),label="tri")
    plt.legend()
    plt.show()