import numpy as np
from SimPEG import Utils
def SliceArray(mesh, var, imageType='CC', normal='z', index=0):
    assert normal in 'xyz', 'normal must be x, y, or z'
    axes = [p for p in 'xyz' if p not in normal.lower()]
    I = mesh.r(var,'CC','CC','M')
    if normal is 'x':
        I = I[index,:,:]
        ind1 = 1; ind2=2
    if normal is 'y':
        I = I[:,index,:]
        ind1 = 0; ind2=2
    if normal is 'z':
        I = I[:,:,index]
        ind1 = 0; ind2=1
    X1 = mesh.r(mesh.gridCC[:,ind1],'CC','CC','M')
    X2 = mesh.r(mesh.gridCC[:,ind2],'CC','CC','M')

    if normal is 'x':
        X1 = X1[index,:,:]
        X2 = X2[index,:,:]
    if normal is 'y':
        X1 = X1[:,index,:]
        X2 = X2[:,index,:]
    if normal is 'z':
        X1 = X1[:,:,index]
        X2 = X2[:,:,index]

    return X1.T, X2.T, I.T

def Rectangle2D(xc, xlen, ylen):
    x1 = xc[0]-xlen*0.5
    x2 = x1.copy()
    x3 = xc[0]+xlen*0.5
    x4 = x3.copy()
    x5 = x1.copy()
    y1 = xc[1]-ylen*0.5
    y2 = xc[1]+ylen*0.5
    y3 = y2.copy()
    y4 = xc[1]-ylen*0.5
    y5 = y1.copy()

    xy = np.c_[np.r_[x1, x2, x3, x4, x5], np.r_[y1, y2, y3, y4, y5]]

    return xy

def Circle2D(xc, r, n):
    theta = np.linspace(-np.pi, np.pi, n)
    x = r*np.cos(Utils.mkvc(theta))+xc[0]
    y = r*np.sin(Utils.mkvc(theta))+xc[1]

    return np.c_[x, y]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    test = Rectangle2D(np.r_[0, 0], 10., 10.)
    plt.plot(test[:,0],test[:,1], 'k--')
    plt.show()

