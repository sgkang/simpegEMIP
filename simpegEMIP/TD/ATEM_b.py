from simpegAIP.TD import BaseATEMProblem
from SimPEG.Utils import mkvc, sdiag
import numpy as np
from SimPEG.EM.TDEM.BaseTDEM import FieldsTDEM
from SimPEG.EM.TDEM.SurveyTDEM import SurveyTDEM

class FieldsATEM_e_from_b(FieldsTDEM):
    """Fancy Field Storage for a TDEM survey."""
    knownFields = {'b': 'F'}
    aliasFields = {'e': ['b','E','e_from_b'], 'j': ['b','E','j_from_b']}

    def startup(self):
        self.MeI  = self.survey.prob.mesh.getEdgeInnerProduct(invMat=True)
        self.MeS = self.survey.prob.MeS
        self.MeSigmaI  = self.survey.prob.MeSigmaI
        self.edgeCurlT = self.survey.prob.mesh.edgeCurl.T
        self.MfMui     = self.survey.prob.MfMui

    def e_from_b(self, b, srcInd, timeInd):
        e = self.MeSigmaI*(self.edgeCurlT*(self.MfMui*b))
        if self.survey.prob.waveformType == "GENERAL":
            e -= self.MeSigmaI*self.MeS*self.survey.prob.current[timeInd+1]
        return  e

    def j_from_b(self, b, srcInd, timeInd):
        j = self.MeI*(self.edgeCurlT*(self.MfMui*b))
        if self.survey.prob.waveformType == "GENERAL":
            j -= self.MeI*self.MeS*self.survey.prob.current[timeInd+1]
        return  j
        

class ProblemATEM_b(BaseATEMProblem):
    """
        Time-Domain EM problem - B-formulation

        TDEM_b treats the following discretization of Maxwell's equations

        .. math::
            \dcurl \e^{(t+1)} + \\frac{\\b^{(t+1)} - \\b^{(t)}}{\delta t} = 0 \\\\
            \dcurl^\\top \MfMui \\b^{(t+1)} - \MeSig \e^{(t+1)} = \Me \j_s^{(t+1)}

        with \\\(\\b\\\) defined on cell faces and \\\(\e\\\) defined on edges.
    """

    def __init__(self, mesh, mapping=None, **kwargs):
        BaseATEMProblem.__init__(self, mesh, mapping=mapping, **kwargs)

    solType = 'b' #: Type of the solution, in this case the 'b' field
    surveyPair = SurveyTDEM
    _FieldsForward_pair = FieldsATEM_e_from_b     #: used for the forward calculation only

    ####################################################
    # Internal Methods
    ####################################################

    def getA(self, tInd):
        """
            :param int tInd: Time index
            :rtype: scipy.sparse.csr_matrix
            :return: A
        """
        dt = self.timeSteps[tInd]
        return self.MfMui*self.mesh.edgeCurl*self.MeSigmaI*self.mesh.edgeCurl.T*self.MfMui + (1.0/dt)*self.MfMui

    def getRHS(self, tInd, F):
        dt = self.timeSteps[tInd]
        B_n = np.zeros((self.mesh.nF,len(self.survey.srcList)))
        
        for isrc, src in enumerate(self.survey.srcList):       
            B_n[:,isrc] = F[src,'b',tInd].flatten()

        RHS = (1.0/dt)*self.MfMui*B_n      
        if self.waveformType =="GENERAL":            
            RHS+= self.MfMui*self.mesh.edgeCurl*self.MeSigmaI*self.MeS*self.current[tInd+1]
        return RHS

