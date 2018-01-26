from __future__ import division
import numpy as np
from SimPEG import Utils


def getJpol(
    timeSteps,
    tInd,
    E,
    MeK,
    MeCnk,
    MeDsigOff_0,
    MeDsigOff_n
    ):
    """
        Computation of polarization currents
    """

    dt = timeSteps[tInd]
    # Handling when jpol at t = 0
    if tInd < 0:
        jpol = MeDsigOff_0.flatten()*E[:, :, 0]
        return jpol
    print (E[:, :, tInd].shape)
    print (MeK.shape)
    jpol = MeK.flatten()*E[:, :, tInd]

    for k in range(1, tInd):
        dt = timeSteps[k]
        jpol += (dt/2)*MeCnk[:, k].flatten() * E[:, :, k]
        jpol += (dt/2)*MeCnk[:, k+1].flatten() * E[:, :, k+1]

    # Handling when jpol at t < 0
    jpol += MeDsigOff_n.flatten()*E[:, :, 0]
    return jpol
