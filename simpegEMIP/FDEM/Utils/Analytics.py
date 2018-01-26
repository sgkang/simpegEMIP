import numpy as np
from scipy.constants import mu_0, pi, epsilon_0
from simpegEMIP.FDEM.Utils import ColeColePelton


def hzAnalyticDipoleF(r, f, sigma, option="secondary"):
    w = 2*pi*f
    k = np.sqrt(-1j*sigma*mu_0*w + w**2*mu_0*epsilon_0)
    t1 = -1/(2*pi*(k**2)*(r**5))
    t2 = (9 + 9*1j*k*r - 4*(k**2)*(r**2) - 1j*(k**3)*(r**3)) * np.exp(-1j*k*r) - 9
    hz = t1*t2
    hzp = -1/(4*np.pi*r**3)
    if option=="secondary":
    	hz -= hzp
    return hz

def hzAnalyticDipoleF_CC(r, f, func = ColeColePelton, sigmaInf=None, eta=0.1, tau=0.1, c=1):
    sigma = func(f, sigmaInf, eta, tau, c)
    hz = hzAnalyticDipoleF(r, f, sigma)
    return hz

def hzAnalyticCentLoopF(a, f, sigma, option="secondary", I=1.):
    mu_0 = 4*np.pi*1e-7
    w = 2*np.pi*f
    k = np.sqrt(-1j*w*mu_0*sigma)
    Hz = -I/(k**2*a**3)*(3-(3+3*1j*k*a-k**2*a**2)*np.exp(-1j*k*a))
    if option == 'secondary':
        Hzp = I/2./a
        Hz = Hz-Hzp
    return Hz

def hzAnalyticCentLoopF_CC(a, f, func = ColeColePelton, sigmaInf=None, eta=0.1, tau=0.1, c=1):
    sigma = func(f, sigmaInf, eta, tau, c)
    hz = hzAnalyticCentLoopF(a, f, sigma)
    return hz

if __name__ == '__main__':
	print ('kang')
