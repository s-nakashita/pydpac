import numpy as np 
import scipy.fft as fft
import logging
from logging.config import fileConfig

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')

# 1-dimensional truncation module using FFT
class Trunc1d:
    def __init__(self,ix,ntrunc=None,cyclic=True,nghost=None,nglobal=None,ttype='f'):
        nx = ix.size
        dx = ix[1] - ix[0]
        self.cyclic = cyclic
        self.ttype = ttype
        if self.ttype=='f':
            # Fourier transform
            logger.info("Trunc1d: Transform type = FT")
            if self.cyclic:
                self.E = np.eye(ix.size)
            else:
                if nghost is None:
                    if nglobal is None:
                        nghost = nx//10
                    else:
                        nghost = (nglobal - nx)//2
                dwindow = (1.0 + np.cos(np.pi*np.arange(1,nghost+1)/nghost))*0.5
                dwindow = np.zeros(nghost)
                self.E = np.zeros((nx+2*nghost,nx))
                self.E[0:nghost,0] = dwindow[::-1]
                self.E[nghost:nghost+nx,:] = np.eye(nx)[:,:]
                self.E[nghost+nx:,-1] = dwindow[:]
                nx += 2*nghost
            self.F = fft.fft(np.eye(nx),axis=0)
            self.f = fft.fftfreq(nx,dx)
            if ntrunc is None:
                self.ntrunc = self.f.size // 2 
            else:
                self.ntrunc = ntrunc
            logger.info(f"Trunc1d: ntrunc={self.ntrunc} ftrunc={self.f[self.ntrunc]:.4f}")
            self.ix_trunc = ix.copy()
            self.dx_trunc = dx
            self.T = np.eye(self.F.shape[0])
            if self.ntrunc<self.f.size-self.ntrunc:
                self.T[self.ntrunc:self.f.size-self.ntrunc,:] = 0.0
                self.T[self.f.size-self.ntrunc,:] *= 2
            #self.ix_trunc = np.linspace(ix[0],ix[0]+nx*dx,\
            #    min(self.ntrunc*2+1,nx),endpoint=False)
            #self.dx_trunc = self.ix_trunc[1] - self.ix_trunc[0]
            #if self.ntrunc<self.f.size-self.ntrunc:
            #    self.T = np.zeros((self.ix_trunc.size,self.F.shape[0]))
            #    i = 0
            #    for j in range(self.ntrunc+1):
            #        self.T[i,j] = dx / self.dx_trunc
            #        i += 1
            #    for j in range(self.f.size-self.ntrunc,self.T.shape[1]):
            #        self.T[i,j] = dx / self.dx_trunc
            #        #if j==self.f.size-self.ntrunc:
            #        #    self.T[i,j] = 2 * dx / self.dx_trunc
            #        i += 1
            #else:
            #    self.T = np.eye(self.F.shape[0])
            self.Fi = fft.ifft(np.eye(self.T.shape[0]),axis=0)
            self.Ei = np.eye(self.ix_trunc.size)
        elif self.ttype == 's':
            # Discrete sine transform
            logger.info(f"Trunc1d: Transform type = DST")
            if self.cyclic:
                self.E = np.zeros((nx-1,nx)) #exclude boundary point
            else:
                self.E = np.zeros((nx-2,nx)) #exclude boundary points
            for i in range(self.E.shape[0]):
                self.E[i,i+1] = 1.
            self.F = fft.dst(np.eye(self.E.shape[0]),type=1,axis=0)
            self.f = np.arange(1,self.F.shape[0]+1)/(2.*dx*nx)
            if ntrunc is None:
                self.ntrunc = self.F.shape[0]
            else:
                self.ntrunc = ntrunc
            logger.info(f"Trunc1d: ntrunc={self.ntrunc} ftrunc={self.f[self.ntrunc-1]:.4f}")
            self.ix_trunc = ix.copy()
            self.dx_trunc = dx 
            self.T = np.eye(self.F.shape[0])
            if self.ntrunc < self.T.shape[0]:
                self.T[self.ntrunc:] = 0.
            self.Fi = fft.idst(np.eye(self.T.shape[0]),type=1,axis=0)
            if self.cyclic:
                self.Ei = np.zeros((nx,nx-1))
            else:
                self.Ei = np.zeros((nx,nx-2))
            for i in range(self.Ei.shape[1]):
                self.Ei[i+1,i] = 1.
        logger.info(f"Trunc1d: T.shape={self.T.shape}")
        logger.info(f"Trunc1d: Fi.shape={self.Fi.shape}")
        #if cyclic:
        #    self.Ei = np.eye(ix.size)
        #else:
        #    self.Ei = np.zeros((ix.size,nx))
        #    self.Ei[:,nghost:nghost+ix.size] = np.eye(ix.size)[:,:]
    
    def __call__(self,x,return_coef=False):
        y = np.dot(self.F,np.dot(self.E,x))
        ytrunc = np.dot(self.T,y)
        xtrunc = np.dot(self.Ei,np.dot(self.Fi,ytrunc)).real
        if return_coef:
            if self.ttype == 'f':
                return xtrunc, fft.fftshift(self.f), fft.fftshift(y)
            else:
                return xtrunc, self.f, y
        else:
            return xtrunc