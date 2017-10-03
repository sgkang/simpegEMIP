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
        jpol = Utils.sdiag(MeDsigOff_0)*E[:, :, 0]
        return jpol
    jpol = Utils.sdiag(MeK)*E[:, :, tInd]

    for k in range(1, tInd):
        dt = timeSteps[k]
        jpol += (dt/2)*Utils.sdiag(MeCnk[:, k]) * E[:, :, k]
        jpol += (dt/2)*Utils.sdiag(MeCnk[:, k+1]) * E[:, :, k+1]

    # Handling when jpol at t < 0
    jpol += Utils.sdiag(MeDsigOff_n)*E[:, :, 0]
    return jpol
