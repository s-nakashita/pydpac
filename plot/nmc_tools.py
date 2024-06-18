import numpy as np 
import scipy.fft as fft
import logging
logging.basicConfig(level=logging.INFO)

class NMC_tools:
    def __init__(self,ix,cyclic=True,ttype='f',detrend=None):
        self.ix = ix # dimensional gridpoints (ix[i] = i*dx)
        self.cyclic = cyclic # periodicity of fields
        self.nx = self.ix.size
        self.dx = self.ix[1] - self.ix[0]
        self.Lx = self.ix[-1] - self.ix[0]
        if self.cyclic:
            self.Lx+=self.dx
        self.ttype = ttype # transform type: f=DFT, s=DST, c=DCT
        self.tname = {'f':'DFT','s':'DST','c':'DCT'}
        self.detrend = detrend
        if self.detrend is None:
            if not self.cyclic and (self.ttype == 'f' or self.ttype == 's'):
                self.detrend = True
            else:
                self.detrend = False
        logging.info(f"NMC_tools: cyclic={self.cyclic} ttype={self.tname[self.ttype]} detrend={self.detrend}")

    # correlation length-scale (Daley 1991; Belo Pereira and Berre 2006)
    def corrscale(self,bmat):
        delx = np.eye(self.nx)
        if self.cyclic:
            delx[0,-1] = -0.5 / self.dx
            delx[0,1] = 0.5 / self.dx
            delx[self.nx-1,self.nx-2] = -0.5 / self.dx
            delx[self.nx-1,0] = 0.5 / self.dx
        else:
            delx[0,0] = -1.0 / self.dx
            delx[0,1] = 1.0 / self.dx
            delx[self.nx-1,self.nx-2] = -1.0 / self.dx
            delx[self.nx-1,self.nx-1] = 1.0 / self.dx
        for i in range(1,self.nx-1):
            delx[i,i-1] = -0.5 / self.dx
            delx[i,i+1] = 0.5 / self.dx
    
        var = np.diag(bmat)
        var_del = np.diag(delx @ bmat @ delx.transpose())
        del_var = (delx @ np.sqrt(var))*(delx @ np.sqrt(var))
        l2 = var / (var_del - del_var)
        return np.where(l2>=0.0, np.sqrt(l2), 0.)

    # grid variance => power spectral density
    def psd(self,x,axis=0,average=True):
        nx = x.shape[axis]
        if self.cyclic:
            xtmp = x.copy()
        else:
            #if nghost is None:
            #    nghost = nx//10
            #Lx += 2*nghost*dx
            #if nghost>0:
            #    dwindow = (1.0 + np.cos(np.pi*np.arange(1,nghost+1)/nghost))*0.5
            #    if x.ndim==2:
            #        if axis==0:
            #            xtmp = np.zeros((nx+2*nghost-1,x.shape[1]))
            #            xtmp[nghost:nghost+nx,:] = x[:,:]
            #            xtmp[0:nghost,:] = x[0,:].reshape(1,-1) * dwindow[::-1,None]
            #            xtmp[nghost+nx:,:] = x[-1,:].reshape(1,-1) * dwindow[:-1,None]
            #        else:
            #            xtmp = np.zeros((x.shape[0],nx+2*nghost-1))
            #            xtmp[:,nghost:nghost+nx] = x[:,:]
            #            xtmp[:,0:nghost] = x[:,0].reshape(-1,1) * dwindow[None,::-1]
            #            xtmp[:,nghost+nx:] = x[:,-1].reshape(-1,1) * dwindow[None,:-1]
            #    else:
            #        xtmp = np.zeros(nx+2*nghost-1)
            #        xtmp[nghost:nghost+nx] = x[:]
            #        x[0:nghost] = x[0] * dwindow[::-1]
            #        x[nghost+nx:] = x[-1] * dwindow[:-1]
            #else:
            if self.detrend:
                ## remove linear trend (Errico 1985, MWR)
                if x.ndim==2:
                    if axis==1:
                        trend = (x[:,-1] - x[:,0])/(self.nx - 1)
                        xtrend = 0.5 * (2*np.arange(self.nx)[None,:] - self.nx + 1)*trend[:,None]
                    else:
                        trend = (x[-1,] - x[0,])/(self.nx - 1)
                        xtrend = 0.5 * (2*np.arange(self.nx)[:,None] - self.nx + 1)*trend[None,:]
                else:
                    trend = (x[-1] - x[0])/(self.nx - 1)
                    xtrend = 0.5 * (2*np.arange(self.nx) - self.nx + 1)*trend
                xtmp = x - xtrend
            else:
                xtmp = x.copy()
        ## Durran et al. (2017, MWR)
        if self.ttype == 'f':
            sp = fft.rfft(xtmp,axis=axis,norm='forward')
            wnum = fft.rfftfreq(xtmp.shape[axis],self.dx)*2.0*np.pi
            wgt = np.ones(wnum.size) * self.Lx / 2.0 / np.pi
            wgt[0] *= 0.5
            wgt[-1] *= 0.5
        elif self.ttype == 'c':
            sp = fft.dct(xtmp,axis=axis,norm='forward',type=2)
            wnum = np.arange(xtmp.shape[axis])*np.pi/xtmp.shape[axis]/self.dx
            wgt = np.ones(wnum.size) * self.Lx / np.pi
            wgt[0] *= 0.5
        elif self.ttype == 's':
            sp = fft.dst(xtmp,axis=axis,norm='forward',type=2)
            wnum = np.arange(1,xtmp.shape[axis])*np.pi/xtmp.shape[axis]/self.dx
            wgt = np.ones(wnum.size) * self.Lx / np.pi
            wgt[-1] = 0.0
        if average and x.ndim==2:
            if axis==0:
                psd = np.mean(np.abs(sp)**2*wgt[:,None],axis=1)
            else:
                psd = np.mean(np.abs(sp)**2*wgt[None,:],axis=0)
        else:
            psd = np.abs(sp)**2*wgt
        if self.ttype == 'c':
            # gathering procedure (Denis et al. 2002)
            wnum = wnum[::2]
            psdtmp = psd.copy()
            if x.ndim==2 and not average:
                if axis==0:
                    psd = psd[::2,:]
                    psd[1:] = psd[1:] + 0.5 * psdtmp[1:-1:2]
                    psd[:-1] = psd[:-1] + 0.5 * psdtmp[1:-1:2]
                else:
                    psd = psd[:,::2]
                    psd[:,1:] = psd[:,1:] + 0.5 * psdtmp[:,1:-1:2]
                    psd[:,:-1] = psd[:,:-1] + 0.5 * psdtmp[:,1:-1:2]
            else:
                psd = psdtmp[::2]
                psd[1:] = psd[1:] + 0.5 * psdtmp[1:-1:2]
                psd[:-1] = psd[:-1] + 0.5 * psdtmp[1:-1:2]
        elif self.ttype == 's':
            # gathering procedure (Dennis et al. 2002)
            wnum = wnum[1::2]
            psdtmp = psd.copy()
            if x.ndim==2 and not average:
                if axis==0:
                    psd = psd[1::2,:]
                    psd[:] = psd[:] + 0.5 * psdtmp[:-1:2]
                    psd[:-1] = psd[:-1] + 0.5 * psdtmp[2::2]
                else:
                    psd = psd[:,1::2]
                    psd[:,:] = psd[:,:] + 0.5 * psdtmp[:,:-1:2]
                    psd[:,:-1] = psd[:,:-1] + 0.5 * psdtmp[:,2::2]
            else:
                psd = psdtmp[1::2]
                psd[:] = psd[:] + 0.5 * psdtmp[:-1:2]
                psd[:-1] = psd[:-1] + 0.5 * psdtmp[2::2]
        if not self.cyclic and self.detrend:
            return wnum, psd, xtmp
        else:
            return wnum, psd

    # cross power spectral density
    def cpsd(self,x,y,iy=None,axis=0):
        if iy is not None:
            ny = y.shape[axis]
            dy = iy[1] - iy[0]
            Ly = iy[-1] - iy[0]
            if self.cyclic:
                Ly += dy
        else:
            ny = self.nx
            dy = self.dx
            Ly = self.Lx
        if self.nx!=ny or self.dx!=dy or self.Lx!=Ly:
            print('cannot compute cross spectra')
            exit()
        if self.cyclic:
            xtmp = x.copy()
            ytmp = y.copy()
        else:
            #if nghost is None:
            #    nghost_x = nx//10
            #    nghost_y = ny//10
            #else:
            #    nghost_x = nghost
            #    nghost_y = nghost
            #Lx += 2*nghost_x*dx
            #Ly += 2*nghost_y*dy
            #if nghost_x > 0 and nghost_y > 0:
            #    dwindow_x = (1.0 + np.cos(np.pi*np.arange(1,nghost_x+1)/nghost_x))*0.5
            #    dwindow_y = (1.0 + np.cos(np.pi*np.arange(1,nghost_y+1)/nghost_y))*0.5
            #    if x.ndim==2:
            #        if axis==0:
            #            xtmp = np.zeros((nx+2*nghost_x-1,x.shape[1]))
            #            xtmp[nghost_x:nghost_x+nx,:] = x[:,:]
            #            xtmp[0:nghost_x,:] = x[0,:].reshape(1,-1) * dwindow_x[::-1,None]
            #            xtmp[nghost_x+nx:,:] = x[-1,:].reshape(1,-1) * dwindow_x[:-1,None]
            #            ytmp = np.zeros((ny+2*nghost_y-1,y.shape[1]))
            #            ytmp[nghost_y:nghost_y+ny,:] = y[:,:]
            #            ytmp[0:nghost_y,:] = y[0,:].reshape(1,-1) * dwindow_y[::-1,None]
            #            ytmp[nghost_y+ny:,:] = y[-1,:].reshape(1,-1) * dwindow_y[:-1,None]
            #        else:
            #            xtmp = np.zeros((x.shape[0],nx+2*nghost_x-1))
            #            xtmp[:,nghost_x:nghost_x+nx] = x[:,:]
            #            xtmp[:,0:nghost_x] = x[:,0].reshape(-1,1) * dwindow_x[None,::-1]
            #            xtmp[:,nghost_x+nx:] = x[:,-1].reshape(-1,1) * dwindow_x[None,:-1]
            #            ytmp = np.zeros((y.shape[0],ny+2*nghost_y-1))
            #            ytmp[:,nghost_y:nghost_y+ny] = y[:,:]
            #            ytmp[:,0:nghost_y] = y[:,0].reshape(-1,1) * dwindow_y[None,::-1]
            #            ytmp[:,nghost_y+ny:] = y[:,-1].reshape(-1,1) * dwindow_y[None,:-1]
            #    else:
            #        xtmp = np.zeros(nx+2*nghost_x-1)
            #        xtmp[nghost_x:nghost_x+nx] = x[:]
            #        xtmp[0:nghost_x] = x[0] * dwindow_x[::-1]
            #        xtmp[nghost_x+nx:] = x[-1] * dwindow_x[:-1]
            #        ytmp = np.zeros(ny+2*nghost_y-1)
            #        ytmp[nghost:nghost+nx] = x[:]
            #        ytmp[0:nghost_y] = y[0] * dwindow_y[::-1]
            #        ytmp[nghost_y+ny:] = y[-1] * dwindow_y[:-1]
            #else:
            if self.detrend:
                ## remove linear trend (Errico 1985, MWR)
                if x.ndim==2:
                    if axis==1:
                        trend = (x[:,-1] - x[:,0])/(self.nx - 1)
                        xtrend = 0.5 * (2*np.arange(self.nx) - self.nx)*trend[:,None]
                        trend = (y[:,-1] - y[:,0])/(self.nx - 1)
                        ytrend = 0.5 * (2*np.arange(self.nx) - self.nx)*trend[:,None]
                    else:
                        trend = (x[-1] - x[0])/(self.nx - 1)
                        xtrend = 0.5 * (2*np.arange(self.nx) - self.nx)*trend[None,:]
                        trend = (y[-1] - y[0])/(self.nx - 1)
                        ytrend = 0.5 * (2*np.arange(self.nx) - self.nx)*trend[None,:]
                    xtmp = x - xtrend
                    ytmp = y - ytrend
                else:
                    trend = (x[-1] - x[0])/(self.nx - 1)
                    xtrend = 0.5 * (2*np.arange(self.nx) - self.nx)*trend
                    xtmp = x - xtrend
                    trend = (y[-1] - y[0])/(self.nx - 1)
                    ytrend = 0.5 * (2*np.arange(self.nx) - self.nx)*trend
                    ytmp = y - ytrend
            else:
                xtmp = x.copy()
                ytmp = y.copy()
        if self.ttype=='f':
            sp_x = fft.rfft(xtmp,axis=axis)
            sp_y = fft.rfft(ytmp,axis=axis)
            wnum = fft.rfftfreq(xtmp.shape[axis],self.dx)*2.0*np.pi
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
        cpsd = 2.0*((spm_x.real+spm_x.imag*1.j)*(spm_y.real-spm_y.imag*1.j))*self.dx*self.dx/self.Lx
        return wnum, np.abs(cpsd)

    # scale decomposition using FFT
    def scale_decomp(self,x,kthres=[],axis=0):
        if (type(kthres) == 'list' and len(kthres) == 0) or \
            (hasattr(kthres,'size') and kthres.size == 0):
            print("provide wavenumber thresholds for decomposition")
            return
        xtmp = x.copy()
        if not self.cyclic:
            #if nghost>0:
            #    Lx += 2*nghost*dx
            #    dwindow = (1.0 + np.cos(np.pi*np.arange(1,nghost+1)/nghost))*0.5
            #    if x.ndim==2:
            #        if axis==0:
            #            xtmp = np.zeros((nx+2*nghost-1,x.shape[1]))
            #            xtmp[nghost:nghost+nx,:] = x[:,:]
            #            xtmp[0:nghost,:] = x[0,:].reshape(1,-1) * dwindow[::-1,None]
            #            xtmp[nghost+nx:,:] = x[-1,:].reshape(1,-1) * dwindow[:-1,None]
            #        else:
            #            xtmp = np.zeros((x.shape[0],nx+2*nghost-1))
            #            xtmp[:,nghost:nghost+nx] = x[:,:]
            #            xtmp[:,0:nghost] = x[:,0].reshape(-1,1) * dwindow[None,::-1]
            #            xtmp[:,nghost+nx:] = x[:,-1].reshape(-1,1) * dwindow[None,:-1]
            #    else:
            #        xtmp = np.zeros(nx+2*nghost-1)
            #        xtmp[nghost:nghost+nx] = x[:]
            #        x[0:nghost] = x[0] * dwindow[::-1]
            #        x[nghost+nx:] = x[-1] * dwindow[:-1]
            #else:
            if self.ttype=='s':
                ## remove mean
                xmean = np.mean(xtmp,axis=axis)
                if x.ndim==2:
                    if axis==1:
                        xtmp = xtmp - xmean[:,None]
                    else:
                        xtmp = xtmp - xmean[None,:]
                else:
                    xtmp = xtmp - xmean
            if self.detrend:
                ## remove linear trend (Errico 1985, MWR)
                if x.ndim==2:
                    if axis==1:
                        trend = (x[:,-1] - x[:,0])/(self.nx - 1)
                        xtrend = 0.5 * (2*np.arange(self.nx)[None,:] - self.nx + 1)*trend[:,None]
                    else:
                        trend = (x[-1,] - x[0,])/(self.nx - 1)
                        xtrend = 0.5 * (2*np.arange(self.nx)[:,None] - self.nx + 1)*trend[None,:]
                else:
                    trend = (x[-1] - x[0])/(self.nx - 1)
                    xtrend = 0.5 * (2*np.arange(self.nx) - self.nx + 1)*trend
                #xtmp = xtmp - xtrend
                if x.ndim==2 and axis==1:
                    xtmp = xtmp[:,:-1] - xtrend[:,:-1]
                else:
                    xtmp = xtmp[:-1] - xtrend[:-1]
        logging.debug(f'x.shape={x.shape}')
        logging.debug(f'xtmp.shape={xtmp.shape}')
        if self.ttype=='f':
            sp = fft.rfft(xtmp,axis=axis,norm='forward')
            wnum = fft.rfftfreq(xtmp.shape[axis],self.dx)*2.0*np.pi
        elif self.ttype == 'c':
            sp = fft.dct(xtmp,axis=axis,norm='forward',type=2)
            wnum = np.arange(xtmp.shape[axis])*np.pi/xtmp.shape[axis]/self.dx
        elif self.ttype == 's':
            sp = fft.dst(xtmp,axis=axis,norm='forward',type=2)
            wnum = np.arange(1,xtmp.shape[axis]+1)*np.pi/xtmp.shape[axis]/self.dx
        
        decomp = [sp]
        for k in kthres:
            ik = np.argmin(np.abs(wnum - k))
            sp0 = decomp[-1]
            sp1 = sp0.copy()
            if sp.ndim==2:
                if axis==1:
                    sp0[:,ik:] = 0.0
                    sp1[:,:ik] = 0.0
                else:
                    sp0[ik:,:] = 0.0
                    sp1[:ik,:] = 0.0
            else:
                sp0[ik:] = 0.0
                sp1[:ik] = 0.0
            decomp.append(sp1)
        
        if self.ttype=='f':
            xdecomp = [fft.irfft(stmp,axis=axis,norm='forward') for stmp in decomp]
        elif self.ttype=='c':
            xdecomp = [fft.idct(stmp,axis=axis,norm='forward',type=2) for stmp in decomp]
        elif self.ttype=='s':
            xdecomp = [fft.idst(stmp,axis=axis,norm='forward',type=2) for stmp in decomp]
        if not self.cyclic and self.detrend:
        #    xdecomp[0] = xdecomp[0] + xtrend
        #    if self.ttype=='s':
        #        if x.ndim==2:
        #            if axis==1:
        #                xdecomp[0] = xdecomp[0] + xmean[:,None]
        #            else:
        #                xdecomp[0] = xdecomp[0] + xmean[None,:]
        #        else:
        #            xdecomp[0] = xdecomp[0] + xmean
            xdecomp_new = []
            for i,xd in enumerate(xdecomp):
                xdnew = np.zeros_like(x)
                if x.ndim==2 and axis==1:
                    xdnew[:,:-1] = xd[:,:]
                    xdnew[:,-1] = xd[:,0]
                else:
                    xdnew[:-1,] = xd[:,]
                    xdnew[-1,] = xd[0,]
                if i==0:
                    xdnew = xdnew + xtrend
                    if self.ttype=='s':
                        if x.ndim==2:
                            if axis==1:
                                xdnew = xdnew + xmean[:,None]
                            else:
                                xdnew = xdnew + xmean[None,:]
                        else:
                            xdnew = xdnew + xmean
                xdecomp_new.append(xdnew)
            return xdecomp_new
        else:
            return xdecomp

def wnum2wlen(wnum):
    #Vectorized 2\pi/x, treating x==0 manually
    wnum = np.array(wnum, float)
    near_zero = np.isclose(wnum, 0)
    wlen = np.zeros_like(wnum)
    wlen[near_zero] = np.inf
    wlen[~near_zero] = 2.0 * np.pi / wnum[~near_zero]
    return wlen

def wlen2wnum(wlen):
    return wnum2wlen(wlen)
