from scipy.constants import h,c,k
from numpy import pi,exp

def BlackBody(wavlen, T):
    """ black-body spectrum in logarithmic interval
    wavlen: float, any shape
      wavelength / m
    T: float, scalar
      temperature in K
    return: B
      B: float, same shape as wavlen 
        dimensionless spectrum such that
        int_0^{infty} B d(wavlen)/wavlen = 1
    """
    x = h*c/wavlen/k/T
    return 15*(x/pi)**4/(exp(x) - 1)
