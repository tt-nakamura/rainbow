import numpy as np
from Rainbow import Rainbow
from MieRainbow import MieRainbow
from RGB import RGBFromWavlen
from RefractiveIndex import IndexFromWavlen
from BlackBody import BlackBody

def WavlenAveraging(rbow, theta, a=None, RGB_ONLY=True,
                    wavlen=(380e-9, 700e-9), N_wavlen=16,
                    T=5783, **kw):
    """ average rainbow light over black-body spectrum
    rbow: Rainbow class (NOT Rainbow object)
      either Rainbow, YoungRainbow, AiryRainbow or MieRainbow
    theta: float, 1d-array
      angle between sun and raindrop / radian
    a: float, scalar
      radius of waterdrop / m
      a is ignored if rbow == Rainbow (geometric optics)
    RGB_ONLY: bool, scalar
      if True, only RGB luminance is computed,
      if False, white light luminance is computed in 4-th component
    wavlen: float, tuple
      wavelength range of averaging / m
    N_wavlen: int, scalar
      number of averaging points
    T: float, scalar
      temperature of black-body spectrum / K
    kw: dictionary
      keyword arguments passed to Rainbow.averaged_intensity()
      except that kw['ord_max'] is passed to MieRainbow.__init__()
    return: I
      I: float, shape(3, len(theta)) or (4, len(theta))
        I[0,1,2] is R,G,B luminance.
        if RGB_ONLY is False, I[3] is white light luminance.
        0 <= I <= 1
    """
    wavlen = np.linspace(wavlen[0], wavlen[1], N_wavlen)
    m = IndexFromWavlen(wavlen)
    if rbow == Rainbow:
        r = [rbow(m) for m in m]
    elif issubclass(rbow, Rainbow):
        x = 2*np.pi*a/wavlen
        if rbow == MieRainbow and 'ord_max' in kw:
            ord_max = kw.pop('ord_max')
            r = [rbow(m, x, ord_max) for m,x in zip(m,x)]
        else:
            r = [rbow(m,x) for m,x in zip(m,x)]
    else:
        raise RuntimeError('rbow is not Rainbow')

    I = [r.averaged_intensity(theta, **kw) for r in r]
    S = BlackBody(wavlen, T)/wavlen
    s = np.sum(S)
    RGB = RGBFromWavlen(wavlen)
    i = np.einsum('ji,i,ik', RGB,S,I)/s

    if RGB_ONLY: return i
    else: return np.vstack((i, np.dot(S,I)/s))
