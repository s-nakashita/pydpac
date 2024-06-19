import numpy as np 
import scipy.fft as fft
from scipy.interpolate import interp1d
import logging
from logging.config import fileConfig

logging.config.fileConfig("./logging_config.ini")
logger = logging.getLogger('anl')

# 1-dimensional truncation module using FFT
class Trunc1d:
    def __init__(self,ix,ntrunc=None,ftrunc=None,cyclic=True,resample=True,nghost=None,nglobal=None,ttype='f'):
        # cyclic=False: assuming that input data does not include boundary points
        self.ix = ix
        self.nx = ix.size
        self.dx = ix[1] - ix[0]
        self.cyclic = cyclic
        self.ttype = ttype
        if not self.cyclic and self.ttype == 'f': # or self.ttype == 's'):
            self.detrend = True
        else:
            self.detrend = False
        self.resample = resample
        self.tname = {'f':'DFT','s':'DST','c':'DCT'}
        if self.detrend:
            self.nx -= 1 # DOF reduces due to detrending
        logger.info(f"Trunc1d: Transform type = {self.tname[self.ttype]} cyclic={self.cyclic} detrend={self.detrend}")
        self.first = True
        self._setope(ntrunc=ntrunc,ftrunc=ftrunc)

    def _setope(self,ntrunc=None,ftrunc=None):
        self.ntrunc = ntrunc
        self.ftrunc = ftrunc
        if self.ttype=='f':
            # Discrete Fourier transform
            if self.cyclic:
                self.E = np.eye(self.nx)
            else:
                self.E = np.zeros((self.nx,self.nx+1))
                self.E[:,:-1] = np.eye(self.nx)
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
            self.F = fft.fft(np.eye(self.nx),axis=0)
            self.f = fft.fftfreq(self.nx,self.dx)*2.0*np.pi
            logger.info(f"Trunc1d: f={self.f}")
            if self.ftrunc is None:
                if self.ntrunc is not None:
                    self.ftrunc = self.f[self.ntrunc]
                else:
                    self.ftrunc = np.abs(self.f[self.f.size//2])
                    self.ntrunc = self.f.size // 2
            else:
                self.ntrunc = 0
                while(self.ntrunc<self.f.size):
                    if self.f[self.ntrunc] - self.ftrunc > 0: break
                    self.ntrunc += 1
                #self.ntrunc = np.argmin(np.abs(self.f - self.ftrunc))
            logger.info(f"Trunc1d: ntrunc={self.ntrunc} ftrunc={self.ftrunc:.4f}")
            self.nx_trunc = self.nx
            self.ix_trunc = self.ix.copy()
            self.dx_trunc = self.dx
            self.f_trunc = self.f.copy()
            self.T = np.eye(self.F.shape[0])
            truncation = False
            if self.ntrunc<self.f.size-self.ntrunc:
                truncation=True
                if self.resample:
                    self.nx_trunc = min(2*self.ntrunc,self.nx)
                    if self.cyclic:
                        self.ix_trunc = np.linspace(self.ix[0],self.ix[-1]+self.dx,\
                            self.nx_trunc,endpoint=False)
                        interp = interp1d(self.ix[:self.nx],np.eye(self.nx),axis=0)
                        trunc = interp(self.ix_trunc)
                    else:
                        self.ix_trunc = np.linspace(self.ix[0],self.ix[-1],\
                            self.nx_trunc+1,endpoint=True)
                        interp = interp1d(self.ix[:self.nx],np.eye(self.nx),axis=0)
                        trunc = interp(self.ix_trunc[:-1])
                    self.dx_trunc = self.ix_trunc[1] - self.ix_trunc[0]
                #    self.f_trunc = fft.fftfreq(self.nx_trunc,self.dx_trunc)*2.0*np.pi
                #    self.T = np.zeros((self.nx_trunc,self.F.shape[0]))
                #    #self.T = np.zeros((self.nx_trunc//2+1,self.F.shape[0]))
                #    i = 0
                #    for j in range(self.ntrunc):
                #        self.T[i,j] = 1.0
                #        i += 1
                #    for j in range(self.f.size-self.ntrunc,self.T.shape[1]):
                #        self.T[i,j] = 1.0
                #        #if j==self.f.size-self.ntrunc:
                #        #    self.T[i,j] *= 2.0
                #        i += 1
                #    self.T *= self.dx / self.dx_trunc
                #    logger.info(f"i:{i} nx_trunc:{self.nx_trunc}")
                #else:
                #    #self.T[self.ntrunc+1:,:] = 0.0
                self.T[self.ntrunc+1:self.f.size-self.ntrunc,:] = 0.0
                #self.T[self.f.size-self.ntrunc,:] *= 2
            self.Fi = fft.ifft(np.eye(self.T.shape[0]),axis=0)
            if truncation and self.resample:
                self.Fi = trunc @ self.Fi
            if self.cyclic:
                self.Ei = np.eye(self.ix_trunc.size)
            else:
                self.Ei = np.zeros((self.ix_trunc.size,self.Fi.shape[0]))
                self.Ei[:-1,:] = np.eye(self.Fi.shape[0])[:,:]
                self.Ei[-1, 0] = 1.0
        elif self.ttype == 's':
            # Discrete sine transform
            #if self.cyclic:
            #    self.E = np.zeros((nx-1,nx)) #exclude a boundary point
            #else:
            #    self.E = np.zeros((nx-1,nx+1)) #exclude boundary points
            #for i in range(self.E.shape[0]):
            #    self.E[i,i+1] = 1.
            self.E = np.eye(self.nx)
            self.F = fft.dst(np.eye(self.E.shape[0]),type=2,axis=0)
            self.f = np.arange(1,self.F.shape[0]+1)*np.pi/self.dx/self.nx
            logger.info(f"Trunc1d: f={self.f}")
            if self.ftrunc is None:
                if self.ntrunc is not None:
                    self.ftrunc = self.f[self.ntrunc]
                else:
                    self.ftrunc = self.f[self.f.size-1]
                    self.ntrunc = self.f.size
            else:
                self.ntrunc = 0
                while(self.ntrunc<self.f.size):
                    if self.f[self.ntrunc] - self.ftrunc > 0: break
                    self.ntrunc += 1
                #self.ntrunc = np.argmin(np.abs(self.f - self.ftrunc))
            logger.info(f"Trunc1d: ntrunc={self.ntrunc} ftrunc={self.ftrunc:.4f}")
            self.nx_trunc = self.nx
            self.ix_trunc = self.ix.copy()
            self.dx_trunc = self.dx 
            self.f_trunc  = self.f.copy()
            self.T = np.eye(self.F.shape[0])
            if self.ntrunc < self.T.shape[0]:
                if self.resample:
                    self.nx_trunc = self.ntrunc
                    if self.cyclic:
                        self.ix_trunc = np.linspace(self.ix[0],self.ix[-1]+self.dx,\
                            self.nx_trunc,endpoint=False)
                        self.dx_trunc = self.ix_trunc[1] - self.ix_trunc[0]
                    else:
                        self.dx_trunc = self.dx * self.nx / self.nx_trunc
                        self.ix_trunc = np.linspace(\
                            self.ix[0]-0.5*(self.dx-self.dx_trunc),self.ix[-1]+0.5*(self.dx-self.dx_trunc),\
                            self.nx_trunc,endpoint=True)
                    self.dx_trunc = self.ix_trunc[1] - self.ix_trunc[0]
                    self.f_trunc = np.arange(1,self.ntrunc+1)*np.pi/self.dx_trunc/self.nx_trunc
                    self.T = np.zeros((self.ntrunc,self.F.shape[0]))
                    self.T[:,:self.ntrunc] = np.eye(self.ntrunc)
                    self.T *= self.dx / self.dx_trunc
                else:
                    self.T[self.ntrunc:] = 0.
            logger.info(f"Trunc1d: f_trunc={self.f_trunc}")
            self.Fi = fft.idst(np.eye(self.T.shape[0]),type=2,axis=0)
            ##if self.cyclic:
            #self.Ei = np.zeros((self.ix_trunc.size,self.Fi.shape[0]))
            ##else:
            ##    self.Ei = np.zeros((nx,nx-2))
            #for i in range(self.Ei.shape[1]):
            #    self.Ei[i+1,i] = 1.
            self.Ei = np.eye(self.ix_trunc.size)
        elif self.ttype == 'c':
            # Discrete cosine transform
            self.E = np.eye(self.nx)
            self.F = fft.dct(np.eye(self.E.shape[0]),axis=0,norm='forward',type=2)
            self.f = np.arange(self.F.shape[0])*np.pi/self.dx/self.nx
            logger.info(f"Trunc1d: f={self.f}")
            if self.ftrunc is None:
                if self.ntrunc is not None:
                    self.ftrunc = self.f[self.ntrunc]
                else:
                    self.ftrunc = self.f[self.f.size-1]
                    self.ntrunc = self.f.size
            else:
                self.ntrunc = 0
                while(self.ntrunc<self.f.size):
                    if self.f[self.ntrunc] - self.ftrunc > 0: break
                    self.ntrunc += 1
                #self.ntrunc = np.argmin(np.abs(self.f - self.ftrunc))
            logger.info(f"Trunc1d: ntrunc={self.ntrunc} ftrunc={self.ftrunc:.4f}")
            self.nx_trunc = self.nx
            self.ix_trunc = self.ix.copy()
            self.dx_trunc = self.dx 
            self.f_trunc  = self.f.copy()
            self.T = np.eye(self.F.shape[0])
            if self.ntrunc < self.T.shape[0]:
                if self.resample:
                    self.nx_trunc = self.ntrunc
                    if self.cyclic:
                        self.ix_trunc = np.linspace(self.ix[0],self.ix[-1]+self.dx,\
                            self.nx_trunc,endpoint=False)
                        self.dx_trunc = self.ix_trunc[1] - self.ix_trunc[0]
                    else:
                        self.dx_trunc = self.dx * self.nx / self.nx_trunc
                        self.ix_trunc = np.linspace(\
                            self.ix[0]-0.5*(self.dx-self.dx_trunc),self.ix[-1]+0.5*(self.dx-self.dx_trunc),\
                            self.nx_trunc,endpoint=True)
                    self.f_trunc = np.arange(self.ntrunc)*np.pi/self.dx_trunc/self.nx_trunc
                    self.T = np.zeros((self.ntrunc,self.F.shape[0]))
                    self.T[:,:self.ntrunc] = np.eye(self.ntrunc)
                    #self.T *= dx / self.dx_trunc
                else:
                    self.T[self.ntrunc:] = 0.
            logger.info(f"Trunc1d: f_trunc={self.f_trunc}")
            self.Fi = fft.idct(np.eye(self.T.shape[0]),axis=0,norm='forward',type=2)
            self.Ei = np.eye(self.ix_trunc.size)
        logger.info(f"Trunc1d: T.shape={self.T.shape}")
        logger.info(f"Trunc1d: Fi.shape={self.Fi.shape}")
        if self.first and logger.isEnabledFor(logging.DEBUG):
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(ncols=3,nrows=2,constrained_layout=True)
            mp=[]
            axs[0,0].set_title('E')
            mp0=axs[0,0].matshow(self.E)
            mp.append(mp0)
            axs[0,1].set_title('F')
            mp1=axs[0,1].matshow(np.abs(self.F))
            mp.append(mp1)
            axs[0,2].set_title('T')
            mp2=axs[0,2].matshow(self.T)
            mp.append(mp2)
            axs[1,0].set_title('Fi')
            mp3=axs[1,0].matshow(np.abs(self.Fi))
            mp.append(mp3)
            axs[1,1].set_title('Ei')
            mp4=axs[1,1].matshow(self.Ei)
            mp.append(mp4)
            for m, ax in zip(mp,axs.flatten()):
                fig.colorbar(m,ax=ax,shrink=0.6,pad=0.01)
            axs[1,2].remove()
            plt.show()
            plt.close()
            self.first = False
        #if cyclic:
        #    self.Ei = np.eye(ix.size)
        #else:
        #    self.Ei = np.zeros((ix.size,nx))
        #    self.Ei[:,nghost:nghost+ix.size] = np.eye(ix.size)[:,:]
    
    def _preprocess(self,x,axis=0):
        global xmean, trend
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
        if self.detrend:
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
        return xtmp
    
    def _postprocess(self,x,xtrunc,axis=0):
        if self.detrend:
            nx = x.shape[axis]
            if xtrunc.ndim==2:
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

    def __call__(self,x,axis=0,return_coef=False,ntrunc=None,ftrunc=None):
        xtmp = self._preprocess(x,axis=axis)
        y = np.dot(self.F,np.dot(self.E,xtmp))
        #logger.debug(f"y={np.abs(y)}")
        ytrunc = np.dot(self.T,y)
        #logger.debug(f"ytrunc={np.abs(ytrunc)}")
        xtrunc = np.dot(self.Ei,np.dot(self.Fi,ytrunc).real)
        #logger.debug(f"xtrunc={xtrunc}")
        self._postprocess(x,xtrunc,axis=axis)
        if return_coef:
            if self.ttype == 'f':
                return xtrunc, fft.fftshift(self.f), fft.fftshift(y), fft.fftshift(self.f_trunc), fft.fftshift(ytrunc)
            else:
                return xtrunc, self.f, y, self.f_trunc, ytrunc
        else:
            return xtrunc
    
    # scale decomposition
    def scale_decomp(self,x,kthres=[],axis=0):
        if (type(kthres) == 'list' and len(kthres) == 0) or \
            (hasattr(kthres,'size') and kthres.size == 0):
            print("provide wavenumber thresholds for decomposition")
            return
        xtmp = self._preprocess(x,axis=axis)
        ixlist = []
        xdecomp = []
        for i,k in enumerate(kthres):
            self._setope(ftrunc=k)
            y = np.dot(self.F,np.dot(self.E,xtmp))
            #logger.debug(f"y={np.abs(y)}")
            ytrunc = np.dot(self.T,y)
            #logger.debug(f"ytrunc={np.abs(ytrunc)}")
            xtrunc = np.dot(self.Ei,np.dot(self.Fi,ytrunc).real)
            #logger.debug(f"xtrunc={xtrunc}")
            self._postprocess(x,xtrunc,axis=axis)
            ixlist.append(self.ix_trunc)
            if i>0:
                for j in range(i):
                    xtrunc = xtrunc - xdecomp[j]
            xdecomp.append(xtrunc)
        # residual
        ixlist.append(self.ix)
        xtrunc = x.copy()
        for j in range(len(kthres)):
            xtrunc = xtrunc - xdecomp[j]
        xdecomp.append(xtrunc)
        return ixlist, xdecomp