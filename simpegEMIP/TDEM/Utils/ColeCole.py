import numpy as np
from scipy.special import erfc


def ColeCole(f, sigma0=None, sigmaInf=None, eta=0.1, tau=0.1, c=1):
    if sigma0 is None and sigmaInf is None:
        raise ValueError('Error: either sigma0 or sigmaInf must be set')
    elif sigma0 is not None and sigmaInf is not None:
        raise ValueError('Error: both sigma0 or sigmaInf are set')
    if sigmaInf is None:
        sigmaInf = sigma0/(1.-eta)

    w = 2*np.pi*f
    return sigmaInf*(1 - eta/(1 + (1-eta)*(1j*w*tau)**c))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    frequency = np.logspace(-3, 3, 61)
    sigma = ColeCole(frequency, sigmaInf=1e-2)
    show = True
    if show:
        fig, ax = plt.subplots(1,1, figsize = (5,5))
        ax1 = ax.twinx()
        ax.semilogx(frequency, sigma.real, 'k')
        ax1.semilogx(frequency, sigma.imag, 'r')
        plt.show()
