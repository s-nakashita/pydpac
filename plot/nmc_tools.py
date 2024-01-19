import numpy as np 

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