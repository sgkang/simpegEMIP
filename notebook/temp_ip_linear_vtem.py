from SimPEG import *
from SimPEG import EM
from scipy.constants import mu_0
import numpy as np
import scipy.sparse as sp
from simpegEMIP.StretchedExponential import SEInvImpulseProblem, SESurvey
from simpegEMIP.TDEM import geteref, Problem3D_Inductive, Survey
import matplotlib.pyplot as plt
from pymatsolver import PardisoSolver
from simpegem1d import DigFilter
# %matplotlib inline
# import matplotlib 
# matplotlib.rcParams["font.size"] = 14
eta, tau, c = 0.1, 0.01, 0.5
cs, ncx, ncz, npad = 10., 25, 20, 18
hx = [(cs,ncx), (cs,npad,1.3)]
hz = [(cs,npad,-1.3), (cs,ncz), (cs,npad,1.3)]
mesh = Mesh.CylMesh([hx,1,hz], '00C')    
sigmaInf = np.ones(mesh.nC) * 0.001
airind = mesh.gridCC[:,2]>0.
actinds = ~airind
# layerind = np.logical_and(mesh.gridCC[:,2]<-50, mesh.gridCC[:,2]>-100.)
layerind = (np.logical_and(mesh.gridCC[:,2]<-50, mesh.gridCC[:,2]>-100.)) & (mesh.gridCC[:,0]<100.)
sigmaInf[airind] = 1e-8
sigmaInf[layerind] = 0.01
eta = np.zeros(mesh.nC)
eta[layerind] = 0.5
tau = np.ones(mesh.nC) * 0.005
c = np.ones(mesh.nC) * 0.5

actmapeta = Maps.InjectActiveCells(mesh, actinds, 0.)
actmaptau = Maps.InjectActiveCells(mesh, actinds, 0.005)
actmapc = Maps.InjectActiveCells(mesh, actinds, 0.5)

wires = Maps.Wires(('eta', actmapeta.nP), ('tau', actmapeta.nP), ('c', actmapeta.nP))

taumap = actmaptau*wires.tau
etamap = actmapeta*wires.eta
cmap = actmapc*wires.c

m = np.r_[eta[actinds], tau[actinds], c[actinds]]
rxloc = np.array([[0., 0., 30.]])
srcloc = np.array([[0., 0., 30.]])
dt = 1.47e-3

rxloc = np.array([[0., 0., 30.]])
srcloc = np.array([[0., 0., 30.]])
tpeak = 2.73e-3
# t0 = 4.2e-3
t0 = tpeak + dt
rx_vtem = EM.TDEM.Rx.Point_dbdt(rxloc, np.logspace(np.log10(2e-5), np.log10(0.009), 51)+t0, orientation='z')
src_vtem = EM.TDEM.Src.CircularLoop([rx_vtem], waveform=EM.TDEM.Src.VTEMWaveform(offTime=t0, peakTime=tpeak, a=3.), loc=srcloc)
survey_vtem = EM.TDEM.Survey([src_vtem])
prb_em_vtem = EM.TDEM.Problem3D_e(mesh, sigmaMap=Maps.IdentityMap(mesh))
# prb_em_vtem.verbose = True
prb_em_vtem.timeSteps = [(tpeak/10, 10), ((t0-tpeak)/10, 10), (1e-06, 5), (2.5e-06, 5), (5e-06, 5), (1e-05, 10), (2e-05, 10), (4e-05, 10), (8e-05, 10), (1.6e-04, 10), (3.2e-04, 20)]
prb_em_vtem.Solver = PardisoSolver
prb_em_vtem.pair(survey_vtem)
F_vtem = prb_em_vtem.fields(sigmaInf)
data_vtem = survey_vtem.dpred(sigmaInf, f=F_vtem)
cur = []
for t in prb_em_vtem.times:
    cur.append(src_vtem.waveform.eval(t))
cur = np.hstack(cur)

eref_vtem = geteref(F_vtem[src_vtem, 'eSolution', :], mesh, option=None, tInd=20) 
rxloc = np.array([[0., 0., 30.]])
srcloc = np.array([[0., 0., 30.]])
rx_ip_vtem = EM.TDEM.Rx.Point_dbdt(rxloc, np.logspace(np.log10(2e-5), np.log10(0.009), 51), 'z')
src_ip_vtem = EM.TDEM.Src.CircularLoop([rx_ip_vtem], waveform=EM.TDEM.Src.RampOffWaveform(offTime=0.), loc=srcloc)
dt = 1.47e-3
survey_ip_vtem = Survey([src_ip_vtem])
t1, t2, t3 = dt, t0-0.001365, t0
prb_ip_vtem = Problem3D_Inductive(
    mesh, 
    sigmaInf=sigmaInf, 
    eta=eta, 
    tau=tau, 
    c=c, 
    actinds = ~airind,
    tlags = [0., t1, t2, t3]
)
prb_ip_vtem.Solver = PardisoSolver
prb_ip_vtem.pair(survey_ip_vtem)
prb_ip_vtem.set_eref(eref_vtem)
ip_vtem_approx = survey_ip_vtem.dpred(m)