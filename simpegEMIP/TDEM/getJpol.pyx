from __future__ import division
import numpy as np

cimport numpy as np
cimport cython

DTYPE_int = np.int
ctypedef np.int_t DTYPE_int_t

DTYPE_float = np.float
ctypedef np.float_t DTYPE_float_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def getJpol(
    np.ndarray[DTYPE_float_t, ndim=1] timeSteps,
    int tInd,
    np.ndarray[DTYPE_float_t, ndim=3] E,
    np.ndarray[DTYPE_float_t, ndim=1] MeK,
    np.ndarray[DTYPE_float_t, ndim=2] MeCnk,
    np.ndarray[DTYPE_float_t, ndim=1] MeDsigOff_0,
    np.ndarray[DTYPE_float_t, ndim=1] MeDsigOff_n
):

    """
        Computation of polarization currents
    """
    cdef int k_from = 1
    cdef float dt
    cdef int i, k, j
    cdef int n = E.shape[0]
    cdef int nSrc = E.shape[1]
    cdef np.ndarray[DTYPE_float_t, ndim=2] jpol = np.zeros([n, nSrc], dtype=DTYPE_float)
    dt = timeSteps[tInd]
    cdef DTYPE_float_t value

    # Handling when jpol at t = 0
    if tInd < 0:
        for i in range(n):
            for j in range (nSrc):
                jpol[i, j] = MeDsigOff_0[i]*E[i, j, 0]
        return jpol

    for i in range(n):
        for j in range (nSrc):
            jpol[i, j] = MeK[i]*E[i, j, tInd]

    for i in range(n):
        for j in range (nSrc):
            value = 0.
            for k in range(k_from, tInd):
                dt = timeSteps[k]
                value += (dt/2)*MeCnk[i, k] * E[i, j, k]
                value += (dt/2)*MeCnk[i, k+1] * E[i, j, k+1]
            jpol[i, j] = value

    # Handling when jpol at t < 0
    for i in range(n):
        for j in range (nSrc):
            jpol[i, j] += MeDsigOff_n[i]*E[i, j, 0]
    return jpol
