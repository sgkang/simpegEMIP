import numpy as np
import scipy.sparse as sp
from SimPEG import Utils
from scipy.constants import mu_0

def BiotSavartFun(mesh, r_pts, component = 'z'):
	"""
		Compute systematrix G using Biot-Savart Law


		G = np.vstack((G1,G2,G3..,Gnpts)

		.. math::

	"""
	if r_pts.ndim == 1:
		npts = 1
	else:
		npts = r_pts.shape[0]
	e = np.ones((mesh.nC, 1))
	o = np.zeros((mesh.nC, 1))
	const = mu_0/4/np.pi
	G = np.zeros((npts, mesh.nC*3))

	for i in range(npts):
		if npts == 1:
			r_rx = np.repeat(Utils.mkvc(r_pts).reshape([1,-1]), mesh.nC, axis = 0)
		else:
			r_rx = np.repeat(r_pts[i,:].reshape([1,-1]), mesh.nC, axis = 0)
		r_CC = mesh.gridCC
		r = r_rx-r_CC
		r_abs = np.sqrt((r**2).sum(axis = 1))
		rxind = r_abs==0.
		r_abs[rxind] = mesh.vol.min()**(1./3.)*0.5
		Sx = const*Utils.sdiag(mesh.vol*r[:,0]/r_abs**3)
		Sy = const*Utils.sdiag(mesh.vol*r[:,1]/r_abs**3)
		Sz = const*Utils.sdiag(mesh.vol*r[:,2]/r_abs**3)

		# G_temp = sp.vstack((sp.hstack(( o.T,     e.T*Sz, -e.T*Sy)), \
		# 	                    sp.hstack((-e.T*Sz,  o.T,     e.T*Sx)), \
		# 	                    sp.hstack((-e.T*Sy,  e.T*Sx,  o.T   ))))
		if component == 'x':
			G_temp = np.hstack(( o.T,     e.T*Sz, -e.T*Sy))
		elif component == 'y':
			G_temp = np.hstack((-e.T*Sz,  o.T,     e.T*Sx))
		elif component == 'z':
			G_temp = np.hstack(( e.T*Sy, -e.T*Sx,  o.T   ))
		G[i,:] = G_temp

	return G

def CondSphereAnalFunJ(x, y, z, R, x0, y0, z0, sig1, sig2, E0, flag):
    """
        test
        Analytic function for Electro-Static problem. The set up here is
        conductive sphere in whole-space.

        * (x0,y0,z0)
        * (x0, y0, z0 ): is the center location of sphere
        * r: is the radius of the sphere

    .. math::

        \mathbf{E}_0 = E_0\hat{x}


    """
    if (~np.size(x)==np.size(y)==np.size(z)):
        print ("Specify same size of x, y, z")
        return
    dim = x.shape
    x = Utils.mkvc(x-x0)
    y = Utils.mkvc(y-y0)
    z = Utils.mkvc(z-z0)

    ind = np.sqrt((x)**2+(y)**2+(z)**2 ) < R
    r = Utils.mkvc(np.sqrt((x)**2+(y)**2+(z)**2 ))

    Jx = np.zeros(x.size)
    Jy = np.zeros(x.size)
    Jz = np.zeros(x.size)

    # Inside of the sphere
    rf2 = 3*sig1/(sig2+2*sig1)
    if (flag == 'total'):
        Jx[ind] = sig2*E0*(rf2)
    elif (flag == 'secondary'):
        Jx[ind] = sig2*E0*(rf2)-sig1*E0
    Jy[ind] = 0.
    Jz[ind] = 0.
    # Outside of the sphere
    rf1 = (sig2-sig1)/(sig2+2*sig1)
    if (flag == 'total'):
        Jx[~ind] = sig1*(E0+E0/r[~ind]**5*(R**3)*rf1*(2*x[~ind]**2-y[~ind]**2-z[~ind]**2))
    elif (flag == 'secondary'):
        Jx[~ind] = sig1*(E0/r[~ind]**5*(R**3)*rf1*(2*x[~ind]**2-y[~ind]**2-z[~ind]**2))
    Jy[~ind] = sig1*(E0/r[~ind]**5*(R**3)*rf1*(3*x[~ind]*y[~ind]))
    Jz[~ind] = sig1*(E0/r[~ind]**5*(R**3)*rf1*(3*x[~ind]*z[~ind]))

    return np.reshape(Jx, x.shape, order='F'), np.reshape(Jy, x.shape, order='F'), np.reshape(Jz, x.shape, order='F')


def CondSphereAnalFunB(x, y, z, R, x0, y0, z0, sig1, sig2, E0, flag):
	"""
		test
		Analytic function for Electro-Static problem. The set up here is
		conductive sphere in whole-space.

		* (x0,y0,z0)
		* (x0, y0, z0 ): is the center location of sphere
		* r: is the radius of the sphere

		.. math::

			\mathbf{E}_0 = E_0\hat{x}


	"""

	if (~np.size(x)==np.size(y)==np.size(z)):
		print ("Specify same size of x, y, z")
		return
	dim = x.shape
	x = Utils.mkvc(x-x0)
	y = Utils.mkvc(y-y0)
	z = Utils.mkvc(z-z0)

	ind = np.sqrt((x)**2+(y)**2+(z)**2 ) < R
	r = Utils.mkvc(np.sqrt((x)**2+(y)**2+(z)**2 ))

	Hx = np.zeros(x.size)
	Hy = np.zeros(x.size)
	Hz = np.zeros(x.size)

	# Inside of the sphere
	rf2 = 3*sig1/(sig2+2*sig1)
	Hpy = -sig1*E0/2*z
	Hpz =  sig1*E0/2*y
	if (flag == 'total'):
		Hy[ind] = -3/2*sig2*E0*(rf2)*z[ind]
		Hz[ind] =  3/2*sig2*E0*(rf2)*y[ind]
	elif (flag == 'secondary'):
		Hy[ind] = -3/2*sig2*E0*(rf2)*z[ind] - Hpy[ind]
		Hz[ind] =  3/2*sig2*E0*(rf2)*y[ind] - Hpz[ind]

	# Outside of the sphere
	rf1 = (sig2-sig1)/(sig2+2*sig1)

	if (flag == 'total'):
		Hy[~ind] = sig1*(E0/r[~ind]**3*(R**3)*rf1*(-z[~ind]))+Hpy
		Hz[~ind] = sig1*(E0/r[~ind]**3*(R**3)*rf1*( y[~ind]))+Hpz
	elif (flag == 'secondary'):
		Hy[~ind] = sig1*(E0/r[~ind]**3*(R**3)*rf1*(-z[~ind]))
		Hz[~ind] = sig1*(E0/r[~ind]**3*(R**3)*rf1*( y[~ind]))

	return np.reshape(mu_0*Hx, x.shape, order='F'), np.reshape(mu_0*Hy, x.shape, order='F'), np.reshape(mu_0*Hz, x.shape, order='F')
