from __future__ import division
import numpy as np
from SimPEG import Utils
from scipy.sparse import coo_matrix


def sdiag(h):
    """Sparse diagonal matrix"""
    # return sp.spdiags(mkvc(h), 0, h.size, h.size, format="csr")
    N = h.size
    row = np.arange(N, dtype='intc')
    return coo_matrix((Utils.mkvc(h), (row, row)), shape=(N, N)).tocsr()


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
        jpol = sdiag(MeDsigOff_0)*E[:, :, 0]
        return jpol
    jpol = sdiag(MeK)*E[:, :, tInd]

    for k in range(1, tInd):
        dt = timeSteps[k]
        jpol += (dt/2)*sdiag(MeCnk[:, k]) * E[:, :, k]
        jpol += (dt/2)*sdiag(MeCnk[:, k+1]) * E[:, :, k+1]

    # Handling when jpol at t < 0
    jpol += sdiag(MeDsigOff_n)*E[:, :, 0]
    return jpol
