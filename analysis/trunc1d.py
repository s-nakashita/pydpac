import numpy as np 
import scipy.fft as fft
import logging
from logging.config import fileConfig

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')

# 1-dimensional truncation module using FFT
class Trunc1d:
    def __init__(self,ix,ntrunc=None,cyclic=True,nghost=None,nglobal=None):
        nx = ix.size
        dx = ix[1] - ix[0]
        self.cyclic = cyclic
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
        self.T = np.eye(self.F.shape[0])
        if self.ntrunc+1<self.f.size-self.ntrunc:
            self.T[self.ntrunc+1:self.f.size-self.ntrunc,:] = 0.0
        self.Fi = fft.ifft(np.eye(self.T.shape[0]),axis=0)
        if cyclic:
            self.Ei = np.eye(ix.size)
        else:
            self.Ei = np.zeros((ix.size,nx))
            self.Ei[:,nghost:nghost+ix.size] = np.eye(ix.size)[:,:]
    
    def __call__(self,x,return_coef=False):
        y = np.dot(self.F,np.dot(self.E,x))
        ytrunc = np.dot(self.T,y)
        xtrunc = np.dot(self.Ei,np.dot(self.Fi,ytrunc)).real
        if return_coef:
            return xtrunc, fft.fftshift(self.f), fft.fftshift(y)
        else:
            return xtrunc