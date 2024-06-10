import numpy as np 
import scipy.fft as fft
import logging
from logging.config import fileConfig

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')

# 1-dimensional truncation module using FFT
class Trunc1d:
    def __init__(self,ix,ntrunc=None,ftrunc=None,cyclic=True,nghost=None,nglobal=None,ttype='f'):
        # cyclic=False: assuming that input data does not include boundary points
        nx = ix.size
        dx = ix[1] - ix[0]
        self.cyclic = cyclic
        self.ttype = ttype
        if not self.cyclic:
            nx -= 1 # DOF reduces due to detrending
        if self.ttype=='f':
            # Fourier transform
            logger.info("Trunc1d: Transform type = FT")
            if self.cyclic:
                self.E = np.eye(nx)
            else:
                self.E = np.zeros((nx,nx+1))
                self.E[:,:-1] = np.eye(nx)
            #    if nghost is None:
            #        if nglobal is None:
            #            nghost = nx//10
            #        else:
            #            nghost = (nglobal - nx)//2
            #    dwindow = (1.0 + np.cos(np.pi*np.arange(1,nghost+1)/nghost))*0.5
            #    dwindow = np.zeros(nghost)
            #    self.E = np.zeros((nx+2*nghost,nx))
            #    self.E[0:nghost,0] = dwindow[::-1]
            #    self.E[nghost:nghost+nx,:] = np.eye(nx)[:,:]
            #    self.E[nghost+nx:,-1] = dwindow[:]
            #    nx += 2*nghost
            self.F = fft.fft(np.eye(nx),axis=0)
            self.f = fft.fftfreq(nx,dx)
            logger.info(f"Trunc1d: f={self.f}")
            self.ftrunc = ftrunc
            self.ntrunc = ntrunc
            if self.ftrunc is None:
                if self.ntrunc is not None:
                    self.ftrunc = self.f[self.ntrunc]
                else:
                    self.ftrunc = self.f[self.f.size // 2]
                    self.ntrunc = self.f.size // 2 
            else:
                self.ntrunc = 0
                while(True):
                    if self.f[self.ntrunc] - self.ftrunc > 0: break
                    self.ntrunc += 1
                #self.ntrunc = np.argmin(np.abs(self.f - self.ftrunc))
            logger.info(f"Trunc1d: ntrunc={self.ntrunc} ftrunc={self.ftrunc:.4f}")
            self.nx_trunc = nx
            self.ix_trunc = ix.copy()
            self.dx_trunc = dx
            self.f_trunc = self.f.copy()
            self.T = np.eye(self.F.shape[0])
            if self.ntrunc<self.f.size-self.ntrunc:
                #self.T[self.ntrunc:self.f.size-self.ntrunc,:] = 0.0
                #self.T[self.f.size-self.ntrunc,:] *= 2
                self.nx_trunc = min(2*self.ntrunc,nx)
                if self.cyclic:
                    self.ix_trunc = np.linspace(ix[0],ix[-1]+dx,\
                        self.nx_trunc,endpoint=False)
                else:
                    self.ix_trunc = np.linspace(ix[0],ix[-1],\
                        self.nx_trunc+1,endpoint=True)
                self.dx_trunc = self.ix_trunc[1] - self.ix_trunc[0]
                self.f_trunc = fft.fftfreq(self.nx_trunc,self.dx_trunc)
                self.T = np.zeros((self.nx_trunc,self.F.shape[0]))
                i = 0
                for j in range(self.ntrunc):
                    self.T[i,j] = 1.0
                    i += 1
                for j in range(self.f.size-self.ntrunc,self.T.shape[1]):
                    self.T[i,j] = 1.0
                    #if j==self.f.size-self.ntrunc:
                    #    self.T[i,j] *= 2.0
                    i += 1
                self.T *= dx / self.dx_trunc
                logger.info(f"i:{i} nx_trunc:{self.nx_trunc}")
            self.Fi = fft.ifft(np.eye(self.T.shape[0]),axis=0)
            if self.cyclic:
                self.Ei = np.eye(self.ix_trunc.size)
            else:
                self.Ei = np.zeros((self.ix_trunc.size,self.Fi.shape[0]))
                self.Ei[:-1,:] = np.eye(self.Fi.shape[0])[:,:]
                self.Ei[-1, 0] = 1.0
        elif self.ttype == 's':
            # Discrete sine transform
            logger.info(f"Trunc1d: Transform type = DST")
            if self.cyclic:
                self.E = np.zeros((nx-1,nx)) #exclude a boundary point
            else:
                self.E = np.zeros((nx-1,nx+1)) #exclude boundary points
            for i in range(self.E.shape[0]):
                self.E[i,i+1] = 1.
            self.F = fft.dst(np.eye(self.E.shape[0]),type=1,axis=0)
            self.f = np.arange(1,self.F.shape[0]+1)/(2.*dx*nx)
            logger.info(f"Trunc1d: f={self.f}")
            self.ftrunc = ftrunc
            self.ntrunc = ntrunc
            if self.ftrunc is None:
                if self.ntrunc is not None:
                    self.ftrunc = self.f[self.ntrunc]
                else:
                    self.ftrunc = self.f[self.f.size]
                    self.ntrunc = self.f.size
            else:
                self.ntrunc = 0
                while(True):
                    if self.f[self.ntrunc] - self.ftrunc > 0: break
                    self.ntrunc += 1
                #self.ntrunc = np.argmin(np.abs(self.f - self.ftrunc))
            logger.info(f"Trunc1d: ntrunc={self.ntrunc} ftrunc={self.ftrunc:.4f}")
            self.nx_trunc = nx
            self.ix_trunc = ix.copy()
            self.dx_trunc = dx 
            self.f_trunc  = self.f.copy()
            self.T = np.eye(self.F.shape[0])
            if self.ntrunc < self.T.shape[0]:
                #self.T[self.ntrunc:] = 0.
                self.nx_trunc = self.ntrunc + 1
                if self.cyclic:
                    self.ix_trunc = np.linspace(ix[0],ix[-1]+dx,\
                        self.nx_trunc,endpoint=False)
                else:
                    self.ix_trunc = np.linspace(ix[0],ix[-1],\
                        self.nx_trunc+1,endpoint=True)
                self.dx_trunc = self.ix_trunc[1] - self.ix_trunc[0]
                self.f_trunc = np.arange(1,self.ntrunc+1)/(2.*self.dx_trunc*self.nx_trunc)
                self.T = np.zeros((self.ntrunc,self.F.shape[0]))
                self.T[:,:self.ntrunc] = np.eye(self.ntrunc)
                self.T *= dx / self.dx_trunc
            logger.info(f"Trunc1d: f_trunc={self.f_trunc}")
            self.Fi = fft.idst(np.eye(self.T.shape[0]),type=1,axis=0)
            #if self.cyclic:
            self.Ei = np.zeros((self.ix_trunc.size,self.Fi.shape[0]))
            #else:
            #    self.Ei = np.zeros((nx,nx-2))
            for i in range(self.Ei.shape[1]):
                self.Ei[i+1,i] = 1.
        logger.info(f"Trunc1d: T.shape={self.T.shape}")
        logger.info(f"Trunc1d: Fi.shape={self.Fi.shape}")
        #if cyclic:
        #    self.Ei = np.eye(ix.size)
        #else:
        #    self.Ei = np.zeros((ix.size,nx))
        #    self.Ei[:,nghost:nghost+ix.size] = np.eye(ix.size)[:,:]
    
    def __call__(self,x,axis=0,return_coef=False):
        if self.ttype == 's':
            xmean = np.mean(x,axis=axis)
            if x.ndim==2:
                if axis==1:
                    xtmp = x - xmean[:,None]
                else:
                    xtmp = x - xmean[None,:]
            else:
                xtmp = x - xmean
            logger.debug(f"xmean={xmean}")
        else:
            xtmp = x.copy()
        if not self.cyclic:
            # removing linear trend (Errico 1985, MWR)
            if xtmp.ndim==2:
                nx = xtmp.shape[axis]
                if axis==1:
                    trend = (xtmp[:,-1] - xtmp[:,0])/(nx - 1)
                    xtrend = 0.5 * trend[:,None] * (2*np.arange(nx)[None,:] - nx + 1)
                else:
                    trend = (xtmp[-1,] - xtmp[0,])/(nx - 1)
                    xtrend = 0.5 * trend[None,:] * (2*np.arange(nx)[:,None] - nx + 1)
            else:
                nx = xtmp.size
                trend = (xtmp[-1] - xtmp[0])/(nx - 1)
                xtrend = 0.5 * trend * (2*np.arange(nx) - nx + 1)
            xtmp = xtmp - xtrend
            logger.debug(f"xtrend={xtrend}")
        y = np.dot(self.F,np.dot(self.E,xtmp))
        ytrunc = np.dot(self.T,y)
        xtrunc = np.dot(self.Ei,np.dot(self.Fi,ytrunc)).real
        if not self.cyclic:
            if x.ndim==2:
                if axis==1:
                    xtrend = 0.5 * trend[:,None] * (2*nx*np.arange(self.ix_trunc.size)[None,:]/self.ix_trunc.size - nx + 1)
                else:
                    xtrend = 0.5 * trend[None,:] * (2*nx*np.arange(self.ix_trunc.size)[:,None]/self.ix_trunc.size - nx + 1)
            else:
                xtrend = 0.5 * trend * (2*x.size*np.arange(self.ix_trunc.size)/self.ix_trunc.size - x.size + 1)
            xtrunc = xtrunc + xtrend
        if self.ttype == 's':
            #xtrunc[0] = xtmp[0]
            #xtrunc[-1] = xtmp[-1]
            if x.ndim==2:
                if axis==1:
                    xtrunc = xtrunc + xmean[:,None]
                else:
                    xtrunc = xtrunc + xmean[None,:]
            else:
                xtrunc = xtrunc + xmean
        if return_coef:
            if self.ttype == 'f':
                return xtrunc, fft.fftshift(self.f), fft.fftshift(y), fft.fftshift(self.f_trunc), fft.fftshift(ytrunc)
            else:
                return xtrunc, self.f, y, self.f_trunc, ytrunc
        else:
            return xtrunc