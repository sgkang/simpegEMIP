from . import Linarg
import numpy as np
from SimPEG import Utils


def ComputeDepthWeight(mesh, active, alpha=1.5, topoflag=True, z0=0.):
    if topoflag:
        temp = np.zeros(mesh.nC)
        temp[active] = 1
        temp = temp.reshape((mesh.nCx*mesh.nCy, mesh.nCz), order = 'F')
        tempCCz = mesh.gridCC[:,2].reshape((mesh.nCx*mesh.nCy, mesh.nCz), order = 'F')
        zmax = np.zeros((mesh.nCx*mesh.nCy, 1))
        for i in range(mesh.nCx*mesh.nCy):
            act_temp = temp[i,:] == 1
            if act_temp.sum() == 0:
                val = 1000.
            else:
                val = np.max(tempCCz[i,act_temp])
                zmax[i] = val
        dzmin = mesh.hz.min()
        Zmax = Utils.mkvc(zmax.repeat(mesh.nCz, axis = 1))+dzmin
    elif topoflag == False:
        Zmax = ones(mesh.nC)*z0
    else:
        raise Exception("topoflag should be either True or False!!")

    # dweight_act = Linarg.linearmap(abs(1/(mesh.gridCC[active, 2]-Zmax[active]))**alpha, 1e-2, 1, flag='linear')

    dweight_act = abs(1/(mesh.gridCC[active, 2]-Zmax[active]))**alpha
    dweight_act /= dweight_act.max()
    dweight = np.zeros(mesh.nC) * np.nan
    dweight[active] = dweight_act

    return dweight
