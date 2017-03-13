from SimPEG import Mesh, Survey, Problem, Utils, Models, Maps, PropMaps, np, sp, Solver as SimpegSolver
from scipy.constants import mu_0
from SimPEG.EM.TDEM.BaseTDEM import FieldsTDEM
from simpegAIP.Base import BaseAEMProblem
from SimPEG.Problem import BaseTimeProblem
from SimPEG.Maps import IdentityMap
from simpegAIP.Base import ColeColePropMap


class BaseATEMIPProblem_b(BaseAEMProblem,BaseTimeProblem):
    """docstring for BaseTDEMProblem_b"""    

    PropMap = ColeColePropMap
    sigmaInf = None
    eta = None
    tau = None
    c = None        

    def __init__(self, mesh, mapping=None, **kwargs):
        BaseTimeProblem.__init__(self, mesh, mapping=mapping, **kwargs)

    _FieldsForward_pair = FieldsTDEM  #: used for the forward calculation only

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        # Later put somethings... 
        return toDelete 

    def fields(self, m):
        if self.verbose: print '%s\nCalculating fields(m)\n%s'%('*'*50,'*'*50)

        # This is the place we set Cole-Cole parameters!!!
        self.curModel = m        
        # Create a fields storage object
        F = self._FieldsForward_pair(self.mesh, self.survey)        
        for isrc, src in enumerate(self.survey.srcList):
            # Set the initial conditions
            F[src,'b',0] = src.getInitialFields(self.mesh)['b']
            F[src,'e',0] = 0.
            F[src,'j',0] = 0.

        F = self.forward(m, self.getRHS, F=F)
        if self.verbose: print '%s\nDone calculating fields(m)\n%s'%('*'*50,'*'*50)
        return F

    def forward(self, m, RHS, F=None):
        self.curModel = m
        F = F or FieldsTDEM(self.mesh, self.survey)
        dtFact = None
        Ainv   = None
        for tInd, dt in enumerate(self.timeSteps):
            if dt != dtFact:
                dtFact = dt
                if Ainv is not None:
                    Ainv.clean()
                A = self.getA(tInd)
                if self.verbose: print 'Factoring...   (dt = %e)'%dt
                Ainv = self.Solver(A, **self.solverOpts)
                if self.verbose: print 'Done'
            rhs = RHS(tInd, F)
            if self.verbose: print '    Solving...   (tInd = %d)'%tInd
            sol = Ainv * rhs            
            if self.verbose: print '    Done...'
            if sol.ndim == 1:
                sol.shape = (sol.size,1)
            en, jn = self.updateFields(sol, tInd, F)
            F[:,'b',tInd+1] = sol
            F[:,'e',tInd+1] = en
            F[:,'j',tInd+1] = jn
        Ainv.clean()
        return F

if __name__ == '__main__':
    cs, ncx, ncz, npad = 5., 25, 15, 15
    hx = [(cs,ncx), (cs,npad,1.3)]
    hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
    mesh = Mesh.CylMesh([hx,1,hz], '00C')    
    prob = BaseATEMIPProblem_b(mesh)
    sigmaInf = np.ones(mesh.nC)
    eta = np.ones(mesh.nC)*0.1
    tau = np.ones(mesh.nC)*0.01
    c = np.ones(mesh.nC)
    m = np.r_[sigmaInf, eta, tau, c]
    mapping = ColeColeTimeMap(mesh)
    out = mapping*m


















