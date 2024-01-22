import numpy as np 
import scipy.fft as fft
import logging
from logging.config import fileConfig

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')

# 1-dimensional truncation module using FFT
class Trunc1d:
    def __init__(self,ix,ntrunc,cyclic=True):
        logger.info(f"Trunc1d: ntrunc={ntrunc}")
        nx = ix.size
        dx = ix[1] - ix[0]
        if cyclic:
            self.E = np.eye(ix.size)
        else:
            nghost = nx//10
            dwindow = (1.0 + np.cos(np.pi*np.arange(1,nghost+1)/nghost))*0.5
            self.E = np.zeros((nx+2*nghost-1,nx))
            self.E[0:nghost,0] = dwindow[::-1]
            self.E[nghost:nghost+nx,:] = np.eye(nx)[:,:]
            self.E[nghost+nx:,-1] = dwindow[1:]
            nx += 2*nghost-1
        self.F = fft.fft(np.eye(nx),axis=0)
        f = fft.fftfreq(nx,dx)
        self.T = np.eye(self.F.shape[0])
        self.T[ntrunc+1:f.size-ntrunc,:] = 0.0
        self.Fi = fft.ifft(np.eye(self.T.shape[0]),axis=0)
        if cyclic:
            self.Ei = np.eye(ix.size)
        else:
            self.Ei = np.zeros((ix.size,nx))
            self.Ei[:,nghost:nghost+ix.size] = np.eye(ix.size)[:,:]
    
    def __call__(self,x):
        return np.dot(self.Ei,np.dot(self.Fi,np.dot(self.T,np.dot(self.F,np.dot(self.E,x))))).real