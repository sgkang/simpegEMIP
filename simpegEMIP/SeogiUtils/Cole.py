from SimPEG import Utils
import numpy as np

def petafun(eta, tau, t):
    return Utils.mkvc(eta/(1-eta)/tau*np.exp(-1/(1-eta)/tau*t))

def petafunscalar(eta, tau, t):
    return eta/(1-eta)/tau*np.exp(-1/(1-eta)/tau*t)   
    