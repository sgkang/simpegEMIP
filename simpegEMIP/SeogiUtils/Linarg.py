from SimPEG import Utils
import numpy as np

def linearmap(x, ymin, ymax, flag='linear'):
    """
        Maps given function in the range of ymin, ymax
    """
    if flag =='log':

    	x = np.log10(x)

    	ymin = np.log10(ymin)
    	ymax = np.log10(ymax)

    xmin = Utils.mkvc(x).min()
    xmax = Utils.mkvc(x).max()
    y = (ymax-ymin)*(x-xmin)/(xmax-xmin) + ymin

    if flag == 'log':
    	return 10**(y)

    return y
