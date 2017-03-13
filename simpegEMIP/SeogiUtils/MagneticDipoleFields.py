import numpy as np
from scipy.constants import mu_0, pi

def MagneticDipoleFields(txLoc, obsLoc, component, dipoleMoment=1.):
    """
        Calculate the vector potential of a set of magnetic dipoles
        at given locations 'ref. <http://en.wikipedia.org/wiki/Dipole#Magnetic_vector_potential>'

        :param numpy.ndarray txLoc: Location of the transmitter(s) (x, y, z)
        :param numpy.ndarray obsLoc: Where the potentials will be calculated (x, y, z)
        :param str component: The component to calculate - 'x', 'y', or 'z'
        :param numpy.ndarray dipoleMoment: The vector dipole moment (vertical)
        :rtype: numpy.ndarray
        :return: The vector potential each dipole at each observation location
    """

    if component=='x':
        dimInd = 0
    elif component=='y':
        dimInd = 1
    elif component=='z':
        dimInd = 2
    else:
        raise ValueError('Invalid component')

    txLoc = np.atleast_2d(txLoc)
    obsLoc = np.atleast_2d(obsLoc)
    dipoleMoment = np.atleast_2d(dipoleMoment)

    nFaces = obsLoc.shape[0]
    nTx = txLoc.shape[0]

    m = np.array(dipoleMoment).repeat(nFaces, axis=0)
    B = np.empty((nFaces, nTx))
    for i in range(nTx):
        dR = obsLoc - txLoc[i, np.newaxis].repeat(nFaces, axis=0)
        r = np.sqrt((dR**2).sum(axis=1))
        if dimInd == 0:
            B[:, i] = +(mu_0/(4*pi)) /(r**3) * (3*dR[:,2]*dR[:,0]/r**2)
        elif dimInd == 1:
            B[:, i] = +(mu_0/(4*pi)) /(r**3) * (3*dR[:,2]*dR[:,1]/r**2)
        elif dimInd == 2:
            B[:, i] = +(mu_0/(4*pi)) /(r**3) * (3*dR[:,2]**2/r**2-1)
        else:
            raise Exception("Not Implemented")
    if nTx == 1:
        return B.flatten()
    return B
