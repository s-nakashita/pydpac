import numpy as np

def l96(x, F):
    l = np.zeros_like(x)
    l = (np.roll(x, -1, axis=0) - np.roll(x, 2, axis=0)) * np.roll(x, 1, axis=0) - x + F
    return l

def step(xa, h, F):
    k1 = h * l96(xa, F)
    
    k2 = h * l96(xa+k1/2, F)
    
    k3 = h * l96(xa+k2/2, F)
    
    k4 = h * l96(xa+k3, F)
    
    return xa + (0.5*k1 + k2 + k3 + 0.5*k4)/3.0

def l96_t(x, dx):
    l = np.zeros_like(x)
    l = (np.roll(x, -1, axis=0) - np.roll(x, 2, axis=0)) * np.roll(dx, 1, axis=0) + \
        (np.roll(dx, -1, axis=0) - np.roll(dx, 2, axis=0)) * np.roll(x, 1, axis=0) - dx
    return l

def step_t(x, dx, h, F):
    k1 = h * l96(x, F)
    dk1 = h * l96_t(x, dx)
    
    k2 = h * l96(x+k1/2, F)
    dk2 = h * l96_t(x+k1/2, dx+dk1/2)
    
    k3 = h * l96(x+k2/2, F)
    dk3 = h * l96_t(x+k2/2, dx+dk2/2)
    
    k4 = h * l96(x+k3, F)
    dk4 = h * l96_t(x+k3, dx+dk3)
    
    return dx + (0.5*dk1 + dk2 + dk3 + 0.5*dk4)/3.0

def l96_adj(x, dx):
    l = np.zeros_like(x)
    l = np.roll(x, 2, axis=0) * np.roll(dx, 1, axis=0) + \
        (np.roll(x, -2, axis=0) - np.roll(x, 1, axis=0)) * np.roll(dx, -1, axis=0) - \
        np.roll(x, -1, axis=0) * np.roll(dx, -2, axis=0) - dx
    return l
# np.roll(X0,2,axis=0)*np.roll(dXb,1,axis=0) + (np.roll(X0,-2,axis=0) - np.roll(X0,1,axis=0))*np.roll(dXb,-1,axis=0) - 
# np.roll(X0,-1,axis=0)*np.roll(dXb,-2,axis=0) - dXb
def step_adj(x, dx, h, F):
    k1 = h * l96(x, F)
    x2 = x + 0.5*k1
    k2 = h * l96(x2, F)
    x3 = x + 0.5*k2
    k3 = h * l96(x3, F)
    x4 = x + k3
    k4 = h * l96(x4, F)

    dxa = dx
    dk1 = dx / 6
    dk2 = dx / 3
    dk3 = dx / 3
    dk4 = dx / 6

    dxa = dxa + h * l96_adj(x4, dk4)
    dk3 = dk3 + h * l96_adj(x4, dk4)

    dxa = dxa + h * l96_adj(x3, dk3)
    dk2 = dk2 + 0.5 * h * l96_adj(x3, dk3)

    dxa = dxa + h * l96_adj(x2, dk2)
    dk1 = dk1 + 0.5 * h * l96_adj(x2, dk2)

    dxa = dxa + h * l96_adj(x, dk1)

    return dxa

if __name__ == "__main__":
    n = 40
    F = 8.0
    h = 0.05

    x0 = np.ones(n)*F
    x0[19] += 0.001*F
    tmax = 2.0
    nt = int(tmax/h) + 1

    for k in range(nt):
        x0 = step(x0, h, F)
        #x0 = x
    print(x0)

    a = 1e-5
    x0 = np.ones(n)
    dx = np.random.randn(x0.size)
    adx = step(x0+a*dx, h, F)
    ax = step(x0, h, F)
    jax = step_t(x0, dx, h, F)
    d = np.sqrt(np.sum((adx-ax)**2)) / a / (np.sqrt(np.sum(jax**2)))
    print("TLM check diff.={}".format(d-1))

    ax = step_t(x0, dx, h, F)
    atax = step_adj(x0, ax, h, F)
    d = (ax.T @ ax) - (dx.T @ atax)
    print("ADJ check diff.={}".format(d))