import numpy as np 
import scipy.fft as fft

# correlation length-scale (Daley 1991; Belo Pereira and Berre 2006)
def corrscale(ix,bmat,cyclic=True):
    dx = ix[1] - ix[0]
    nx = ix.size
    delx = np.eye(nx)
    if cyclic:
        delx[0,-1] = -0.5 / dx
        delx[0,1] = 0.5 / dx
        delx[nx-1,nx-2] = -0.5 / dx
        delx[nx-1,0] = 0.5 / dx
    else:
        delx[0,0] = -1.0 / dx
        delx[0,1] = 1.0 / dx
        delx[nx-1,nx-2] = -1.0 / dx
        delx[nx-1,nx-1] = 1.0 / dx
    for i in range(1,nx-1):
        delx[i,i-1] = -0.5 / dx
        delx[i,i+1] = 0.5 / dx
    
    var = np.diag(bmat)
    var_del = np.diag(delx @ bmat @ delx.transpose())
    del_var = (delx @ np.sqrt(var))*(delx @ np.sqrt(var))
    l2 = var / (var_del - del_var)
    return np.where(l2>=0.0, np.sqrt(l2), 0.)

# grid variance => power spectral density
def psd(x,ix,axis=0,cyclic=True,nghost=None,average=True):
    nx = x.shape[axis]
    dx = ix[1] - ix[0]
    Lx = ix[-1] - ix[0]
    if cyclic:
        Lx+=dx
        xtmp = x.copy()
    else:
        if nghost is None:
            nghost = nx//10
        Lx += 2*nghost*dx
        if nghost>0:
            dwindow = (1.0 + np.cos(np.pi*np.arange(1,nghost+1)/nghost))*0.5
            if x.ndim==2:
                if axis==0:
                    xtmp = np.zeros((nx+2*nghost-1,x.shape[1]))
                    xtmp[nghost:nghost+nx,:] = x[:,:]
                    xtmp[0:nghost,:] = x[0,:].reshape(1,-1) * dwindow[::-1,None]
                    xtmp[nghost+nx:,:] = x[-1,:].reshape(1,-1) * dwindow[:-1,None]
                else:
                    xtmp = np.zeros((x.shape[0],nx+2*nghost-1))
                    xtmp[:,nghost:nghost+nx] = x[:,:]
                    xtmp[:,0:nghost] = x[:,0].reshape(-1,1) * dwindow[None,::-1]
                    xtmp[:,nghost+nx:] = x[:,-1].reshape(-1,1) * dwindow[None,:-1]
            else:
                xtmp = np.zeros(nx+2*nghost-1)
                xtmp[nghost:nghost+nx] = x[:]
                x[0:nghost] = x[0] * dwindow[::-1]
                x[nghost+nx:] = x[-1] * dwindow[:-1]
        else:
            xtmp = x.copy()
    sp = fft.rfft(xtmp,axis=axis)
    wnum = fft.rfftfreq(xtmp.shape[axis],dx)*2.0*np.pi
    if average and x.ndim==2:
        if axis==0:
            psd = 2.0*np.mean(np.abs(sp)**2,axis=1)*dx*dx/Lx
        else:
            psd = 2.0*np.mean(np.abs(sp)**2,axis=0)*dx*dx/Lx
    else:
        psd = 2.0*np.abs(sp)**2*dx*dx/Lx
    return wnum, psd

# cross power spectral density
def cpsd(x,y,ix,iy,axis=0,cyclic=True,nghost=None):
    nx = x.shape[axis]
    dx = ix[1] - ix[0]
    Lx = ix[-1] - ix[0]
    ny = y.shape[axis]
    dy = iy[1] - iy[0]
    Ly = iy[-1] - iy[0]
    if nx!=ny or dx!=dy or Lx!=Ly:
        print('cannot compute cross spectra')
        exit()
    if cyclic:
        Lx+=dx
        xtmp = x.copy()
        Ly+=dy
        ytmp = y.copy()
    else:
        if nghost is None:
            nghost_x = nx//10
            nghost_y = ny//10
        else:
            nghost_x = nghost
            nghost_y = nghost
        Lx += 2*nghost_x*dx
        Ly += 2*nghost_y*dy
        if nghost_x > 0 and nghost_y > 0:
            dwindow_x = (1.0 + np.cos(np.pi*np.arange(1,nghost_x+1)/nghost_x))*0.5
            dwindow_y = (1.0 + np.cos(np.pi*np.arange(1,nghost_y+1)/nghost_y))*0.5
            if x.ndim==2:
                if axis==0:
                    xtmp = np.zeros((nx+2*nghost_x-1,x.shape[1]))
                    xtmp[nghost_x:nghost_x+nx,:] = x[:,:]
                    xtmp[0:nghost_x,:] = x[0,:].reshape(1,-1) * dwindow_x[::-1,None]
                    xtmp[nghost_x+nx:,:] = x[-1,:].reshape(1,-1) * dwindow_x[:-1,None]
                    ytmp = np.zeros((ny+2*nghost_y-1,y.shape[1]))
                    ytmp[nghost_y:nghost_y+ny,:] = y[:,:]
                    ytmp[0:nghost_y,:] = y[0,:].reshape(1,-1) * dwindow_y[::-1,None]
                    ytmp[nghost_y+ny:,:] = y[-1,:].reshape(1,-1) * dwindow_y[:-1,None]
                else:
                    xtmp = np.zeros((x.shape[0],nx+2*nghost_x-1))
                    xtmp[:,nghost_x:nghost_x+nx] = x[:,:]
                    xtmp[:,0:nghost_x] = x[:,0].reshape(-1,1) * dwindow_x[None,::-1]
                    xtmp[:,nghost_x+nx:] = x[:,-1].reshape(-1,1) * dwindow_x[None,:-1]
                    ytmp = np.zeros((y.shape[0],ny+2*nghost_y-1))
                    ytmp[:,nghost_y:nghost_y+ny] = y[:,:]
                    ytmp[:,0:nghost_y] = y[:,0].reshape(-1,1) * dwindow_y[None,::-1]
                    ytmp[:,nghost_y+ny:] = y[:,-1].reshape(-1,1) * dwindow_y[None,:-1]
            else:
                xtmp = np.zeros(nx+2*nghost_x-1)
                xtmp[nghost_x:nghost_x+nx] = x[:]
                xtmp[0:nghost_x] = x[0] * dwindow_x[::-1]
                xtmp[nghost_x+nx:] = x[-1] * dwindow_x[:-1]
                ytmp = np.zeros(ny+2*nghost_y-1)
                ytmp[nghost:nghost+nx] = x[:]
                ytmp[0:nghost_y] = y[0] * dwindow_y[::-1]
                ytmp[nghost_y+ny:] = y[-1] * dwindow_y[:-1]
        else:
            xtmp = x.copy()
            ytmp = y.copy()
    sp_x = fft.rfft(xtmp,axis=axis)
    sp_y = fft.rfft(ytmp,axis=axis)
    wnum = fft.rfftfreq(xtmp.shape[axis],dx)*2.0*np.pi
    if x.ndim==2:
        if axis==0:
            spm_x = np.mean(sp_x,axis=1)
            spm_y = np.mean(sp_y,axis=1)
        else:
            spm_x = np.mean(sp_x,axis=0)
            spm_y = np.mean(sp_y,axis=0)
    else:
        spm_x = sp_x
        spm_y = sp_y
    cpsd = 2.0*((spm_x.real+spm_x.imag*1.j)*(spm_y.real-spm_y.imag*1.j))*dx*dx/Lx
    return wnum, np.abs(cpsd)

def wnum2wlen(wnum):
    #Vectorized 2\pi/x, treating x==0 manually
    wnum = np.array(wnum, float)
    near_zero = np.isclose(wnum, 0)
    wlen = np.zeros_like(wnum)
    wlen[near_zero] = np.inf
    wlen[~near_zero] = 2.0 * np.pi / wnum[~near_zero]
    return wlen

wlen2wnum = wnum2wlen

# truncation operator using FFT
def trunc_operator(x,ix=None,ftmax=None,first=False,cyclic=True,nghost=None):
    global E, F, T, Fi, Ei
    if first:
        nx = ix.size
        dx = ix[1] - ix[0]
        I = np.eye(nx)
        if cyclic:
            E = np.eye(ix.size)
        else:
            nghost = nx//10
            dwindow = (1.0 + np.cos(np.pi*np.arange(1,nghost+1)/nghost))*0.5
            E = np.zeros((nx+2*nghost-1,nx))
            E[0:nghost,0] = dwindow[::-1]
            E[nghost:nghost+nx,:] = np.eye(nx)[:,:]
            E[nghost+nx:,-1] = dwindow[1:]
            nx += 2*nghost-1
        F = fft.rfft(np.eye(nx),axis=0)
        f = fft.rfftfreq(nx,dx)
        ntrunc = np.argmin(np.abs(f-ftmax))
        print(f"ntrunc={ntrunc}")
        T = np.eye(F.shape[0])
        T[ntrunc+1:,:] = 0.0
        Fi = fft.irfft(np.eye(T.shape[0]),axis=0)
        if cyclic:
            Ei = np.eye(ix.size)
        else:
            Ei = np.zeros((ix.size,nx))
            Ei[:,nghost:nghost+ix.size] = np.eye(ix.size)[:,:]
    return np.dot(Ei,np.dot(Fi,np.dot(T,np.dot(F,np.dot(E,x))))).real