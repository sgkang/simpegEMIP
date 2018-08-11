from __future__ import division, print_function
import scipy.sparse as sp
import numpy as np
from SimPEG import Problem, Utils, Solver as SimpegSolver
from SimPEG.EM.Base import BaseEMProblem
from simpegEMIP.TDEM.Survey import Survey
from SimPEG.EM.TDEM import BaseTDEMProblem as BaseTDEMProblemOrig
from SimPEG.EM.TDEM import Problem3D_e as Problem3D_e_orig
from SimPEG.EM.TDEM.FieldsTDEM import (
    FieldsTDEM, Fields3D_b, Fields3D_e, Fields3D_h, Fields3D_j,
    Fields_Derivs_eb, Fields_Derivs_hj
)
from scipy.constants import mu_0
import time


class BaseTDEMProblem(BaseTDEMProblemOrig):
    """
    We start with the first order form of Maxwell's equations, eliminate and
    solve the second order form. For the time discretization, we use backward
    Euler.
    """
    surveyPair = Survey  #: A simpegEMIP.TDEM.Survey Class
    dt_threshold = 1e-8

    def __init__(self, mesh, **kwargs):
        BaseEMProblem.__init__(self, mesh, **kwargs)

    def fields(self, m):
        """
        Solve the forward problem for the fields.

        :param numpy.array m: inversion model (nP,)
        :rtype: SimPEG.EM.TDEM.FieldsTDEM
        :return f: fields object
        """

        tic = time.time()
        self.model = m

        n_src = len(self.survey.srcList)
        n_time = self.times.size
        nE = self.mesh.nE
        e = np.zeros((nE, n_src, n_time), order='F', dtype=float)
        # set initial fields
        e[:, :, 0] = self.getInitialFields()

        if self.verbose:
            print('{}\nCalculating fields(m)\n{}'.format('*'*50, '*'*50))

        # timestep to solve forward
        Ainv = None
        for tInd, dt in enumerate(self.timeSteps):
            # keep factors if dt is the same as previous step b/c A will be the
            # same
            if Ainv is not None and (
                tInd > 0 and abs(dt-self.timeSteps[tInd - 1]) >
                self.dt_threshold
            ):
                Ainv.clean()
                Ainv = None

            if Ainv is None:
                A = self.getAdiag(tInd)
                if self.verbose:
                    print('Factoring...   (dt = {:e})'.format(dt))
                Ainv = self.Solver(A, **self.solverOpts)
                if self.verbose:
                    print('Done')

            rhs = self.getRHS(tInd+1)  # this is on the nodes of the time mesh
            Asubdiag = self.getAsubdiag(tInd)

            if self.verbose:
                print('    Solving...   (tInd = {:d})'.format(tInd+1))

            # taking a step
            sol = Ainv * (rhs - Asubdiag * e[:, :, tInd])

            if self.verbose:
                print('    Done...')

            if sol.ndim == 1:
                sol.shape = (sol.size, 1)
            e[:, :, tInd+1] = sol

        if self.verbose:
            print('{}\nDone calculating fields(m)\n{}'.format('*'*50, '*'*50))

        # clean factors and return
        Ainv.clean()
        return e


# ------------------------------- Problem3D_e ------------------------------- #
class Problem3D_e(BaseTDEMProblem, Problem3D_e_orig):
    """
        Solve the EB-formulation of Maxwell's equations for the electric field, e.

        Starting with

        .. math::

            \\nabla \\times \\mathbf{e} + \\frac{\\partial \\mathbf{b}}{\\partial t} = \\mathbf{s_m} \\
            \\nabla \\times \mu^{-1} \\mathbf{b} - \sigma \\mathbf{e} = \\mathbf{s_e}


        we eliminate :math:`\\frac{\\partial b}{\\partial t}` using

        .. math::

            \\frac{\\partial \\mathbf{b}}{\\partial t} = - \\nabla \\times \\mathbf{e} + \\mathbf{s_m}


        taking the time-derivative of Ampere's law, we see

        .. math::

            \\frac{\\partial}{\\partial t}\left( \\nabla \\times \mu^{-1} \\mathbf{b} - \\sigma \\mathbf{e} \\right) = \\frac{\\partial \\mathbf{s_e}}{\\partial t} \\
            \\nabla \\times \\mu^{-1} \\frac{\\partial \\mathbf{b}}{\\partial t} - \\sigma \\frac{\\partial\\mathbf{e}}{\\partial t} = \\frac{\\partial \\mathbf{s_e}}{\\partial t}


        which gives us

        .. math::

            \\nabla \\times \\mu^{-1} \\nabla \\times \\mathbf{e} + \\sigma \\frac{\\partial\\mathbf{e}}{\\partial t} = \\nabla \\times \\mu^{-1} \\mathbf{s_m} + \\frac{\\partial \\mathbf{s_e}}{\\partial t}


    """

    _fieldType = 'e'
    _formulation = 'EB'
    surveyPair = Survey

    def __init__(self, mesh, **kwargs):
        BaseTDEMProblem.__init__(self, mesh, **kwargs)
