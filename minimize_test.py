import numpy as np
import matplotlib.pyplot as plt
import copy
from analysis.minimize import Minimize
plt.rcParams['legend.fontsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
def callback(xk, alphak):
    global x_list, alpha_list
    x_list.append(copy.copy(xk))
    alpha_list.append(alphak)
    #print(x_list)
def callback2(xk):
    global x_list
    x_list.append(copy.copy(xk))
if __name__ == "__main__":
    from scipy.optimize import rosen, rosen_der, rosen_hess
    global x_list, alpha_list
    
    n = 2
    iprint = np.array([1,3], dtype=np.int32)

    args = None
    maxiter = 2000
    x0 = np.ones(n) * -1.0

    # plot
    fig, ax = plt.subplots(figsize=(10,10))
    # function
    delta = 0.01
    minXY = -2.0
    maxXY = 2.0
    xaxis = np.arange(minXY, maxXY, delta)
    yaxis = np.arange(minXY, maxXY, delta)
    X, Y = np.meshgrid(xaxis, yaxis)
    Z = [rosen(np.array([x,y])) for (x,y) in zip(X,Y)]
    print(np.min(np.array(Z)),np.max(np.array(Z)))
    levels = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 50000.0]
    cs = ax.contour(X, Y, Z, levels, colors='black')
    ax.scatter([1.0],[1.0],c='r',s=500,alpha=0.5,marker=(5,1))
    ax.set_xlim(minXY,maxXY)
    ax.set_ylim(minXY,maxXY)

    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['lines.markersize'] = 10.0
    # LBFGS
    x_list = []
    alpha_list = []
    x_list.append(copy.copy(x0))
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
    method="LBFGS", maxiter=maxiter)
    x, flg = minimize(x0, callback=callback)
    print(x_list)
    print(alpha_list)
    optXY = np.array(x_list)
    ax.plot(optXY[:,0], optXY[:,1], "-b", marker="+", label=f"LBFGS({len(x_list)-1})")
    ax.plot(x[0], x[1], '*b')
    ## BFGS
    #x_list = []
    #alpha_list = []
    #x_list.append(copy.copy(x0))
    #minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
    #method="BFGS", maxiter=maxiter)
    #x, flg = minimize(x0, callback=callback2)
    #print(len(x_list))
    #print(len(alpha_list))
    #optXY = np.array(x_list)
    #ax.plot(optXY[:,0], optXY[:,1], "-c", marker="+", label=f"BFGS({len(x_list)-1})")
    #ax.plot(x[0], x[1], '*c')
    # GD
    x_list = []
    alpha_list = []
    x_list.append(copy.copy(x0))
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
    method="GD", maxiter=maxiter)
    x, flg = minimize(x0, callback=callback)
    print(len(x_list))
    print(len(alpha_list))
    optXY = np.array(x_list)
    ax.plot(optXY[:,0], optXY[:,1], color="orange", marker="+", label=f"GD({len(x_list)-1})")
    ax.plot(x[0], x[1], color='orange', marker="*")
    # CGF
    x_list = []
    alpha_list = []
    minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
    method="CGF", maxiter=maxiter, cgtype=1)
    x, flg = minimize(x0, callback=callback)
    print(x_list)
    print(len(alpha_list))
    optXY = np.array(x_list)
    ax.plot(optXY[:,0], optXY[:,1], "-m", marker="+", 
    label=f"CG({len(x_list)-1})")
    #label=f"CG-FR({len(x_list)-1})")
    ax.plot(x[0], x[1], '*m')
    #x_list = []
    #alpha_list = []
    #minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
    #method="CGF", maxiter=maxiter, cgtype=2)
    #x, flg = minimize(x0, callback=callback)
    #print(x_list)
    #print(len(alpha_list))
    #optXY = np.array(x_list)
    #ax.plot(optXY[:,0], optXY[:,1], color="purple", linestyle="dashed", marker="+", label=f"CG-PR({len(x_list)-1})")
    #ax.plot(x[0], x[1], color="purple", marker='*')
    #x_list = []
    #alpha_list = []
    #minimize = Minimize(n, rosen, jac=rosen_der, args=args, iprint=iprint,
    #method="CGF", maxiter=maxiter, cgtype=3)
    #x, flg = minimize(x0, callback=callback)
    #print(x_list)
    #print(len(alpha_list))
    #optXY = np.array(x_list)
    #ax.plot(optXY[:,0], optXY[:,1], color="r", linestyle="dotted", marker="+", label=f"CG-PPR({len(x_list)-1})")
    #ax.plot(x[0], x[1], '*r')
    # EXN
    x_list = []
    alpha_list = []
    x_list.append(copy.copy(x0))
    minimize = Minimize(n, rosen, jac=rosen_der, hess=rosen_hess, args=args, iprint=iprint,
    method="EXN", maxiter=maxiter)
    x, flg = minimize(x0, callback=callback)
    print(len(x_list))
    print(len(alpha_list))
    optXY = np.array(x_list)
    ax.plot(optXY[:,0], optXY[:,1], "-y", marker="+", label=f"Fixed Newton({len(x_list)-1})")
    ax.plot(x[0], x[1], '*y')
    # NCG
    x_list = []
    alpha_list = []
    x_list.append(copy.copy(x0))
    minimize = Minimize(n, rosen, jac=rosen_der, hess=rosen_hess, args=args, iprint=iprint,
    method="NCG", maxiter=maxiter)
    x, flg = minimize(x0, callback=callback)
    print(len(x_list))
    print(len(alpha_list))
    optXY = np.array(x_list)
    ax.plot(optXY[:,0], optXY[:,1], "-g", marker="+", label=f"Newton({len(x_list)-1})")
    ax.plot(x[0], x[1], '*g')
    ## dogleg
    #x_list = []
    #alpha_list = []
    #x_list.append(copy.copy(x0))
    #minimize = Minimize(n, rosen, jac=rosen_der, hess=rosen_hess, args=args, iprint=iprint,
    #method="dogleg", maxiter=maxiter)
    #x, flg = minimize(x0, callback=callback2)
    #print(len(x_list))
    #print(len(alpha_list))
    #optXY = np.array(x_list)
    #ax.plot(optXY[:,0], optXY[:,1], linestyle="solid", color="lime", marker="+", label=f"dogleg({len(x_list)-1})")
    #ax.plot(x[0], x[1], marker='*', color="lime")

    ax.set_aspect('equal')
    ax.legend(ncol=2)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    func_text = r'$100(y-x^2)^2+(1-x)^2$'
    ax.set_title(func_text, fontsize=20)
    fig.savefig('rosen.png')