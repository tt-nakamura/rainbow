import numpy as np
import ctypes as ct

j = ct.cdll.LoadLibrary('libriccati_jn.so')
j.riccati_jn.argtypes = (ct.POINTER(ct.c_double), ct.c_int, ct.c_double)

def riccati_jn(n,x):
    """ riccati-bessel function psi_n(x) of first kind
    defined as x times spherical bessel function j_n(x)
    n: int, scalar
    x: float, scalar
    return: psi, dpsi
      psi: float, 1d-array (length n+1)
        psi_n(x) for n=0,1,...,n
      dpsi: float, 1d-array (length n+1)
        (d/dx)psi_n(x) for n=0,1,...,n
    comment: this program was written because
      scipy.special.riccati_jn dosen' work when x > 4400.
    """
    psi = np.empty(n+1)
    psi_p = psi.ctypes.data_as(ct.POINTER(ct.c_double))
    j.riccati_jn(psi_p, n, x)
    dpsi = psi[:-1] - np.arange(1,n+1)*psi[1:]/x
    dpsi = np.r_[psi[0]/x - psi[1], dpsi]
    return psi,dpsi
