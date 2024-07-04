import numpy as np 

class L63():
    def __init__(self,p=10.0,r=32.0,b=8.0/3.0,dt=0.01):
        self.p = p # prandtl number
        self.r = r # rayleigh number
        self.b = b # aspect ratio
        self.dt = dt # timestep

    def __call__(self,w):
        x, y, z = w 
        dwx = - self.p * x + self.p * y 
        dwy = - x * z      + self.r * x - y
        dwz =   x * y      - self.b * z 
        dw = np.array([dwx,dwy,dwz])
        return w + dw * self.dt 

    def step_t(self,wb,wt):
        xb, yb, zb = wb 
        xt, yt, zt = wt 
        dwx = - self.p * xt      + self.p * yt 
        dwy = (self.r - zb) * xt - yt          - xb * zt
        dwz = yb * xt            + xb * yt     - self.b * zt 
        dw = np.array([dwx,dwy,dwz])
        return wt + dw * self.dt

    def step_adj(self,wb,wa):
        xb, yb, zb = wb 
        xa, ya, za = wa
        dwx = - self.p * xa + (self.r - zb) * ya + yb * za
        dwy =   self.p * xa - ya                 + xb * za
        dwz =               - xb * ya            - self.b * za
        dw = np.array([dwx,dwy,dwz])
        return wa + dw * self.dt

if __name__ == "__main__":
    import matplotlib.pyplot as plt 
    
    model = L63()
    nstep = 500

    w1 = np.zeros((nstep+1,3))
    w2 = np.zeros((nstep+1,3))
    w1[0,:] = 1.0,3.0,5.0
    w2[0,:] = 1.1,3.3,5.5
    for i in range(nstep):
        w1[i+1,] = model(w1[i])
        w2[i+1,] = model(w2[i])
    
    fig = plt.figure(figsize=[8,8])
    ax = fig.add_subplot(projection="3d")
    ax.plot(*w1.transpose())
    ax.plot(*w2.transpose())
    plt.show()

    # TLM & ADJ check
    w0 = np.random.randn(3)
    dw0 = np.random.randn(3)*0.1
    alp = 1.0e-5
    wp = model(w0+alp*dw0)
    wb = model(w0)
    dw = model.step_t(w0,dw0)
    num = np.dot((wp-wb),(wp-wb))
    den = np.dot(dw,dw)
    ratio = np.sqrt(num)/alp/np.sqrt(den)
    print(f"TLM check: |M(x+a*dx)-M(x)| / a|TLM*dx| - 1 = {ratio - 1.0}")

    MtMw = model.step_adj(w0,dw)
    print(f"ADJ check: |(Mx)^T(Mx)| - |x^TM^TMx| = {den - np.dot(dw0,MtMw)}")