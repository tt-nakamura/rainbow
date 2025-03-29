import numpy as np
from Rainbow import Rainbow
from MieRainbow import MieRainbow
from RefractiveIndex import IndexFromWavlen
from scipy.stats import lognorm
from WavlenAveraging import WavlenAveraging

def DropsizeAveraging(rbow, theta, a,
                      sigma=0, width=0.95,
                      wavlen=None, rv=None,
                      N_dropsize=16, **kw):
    """ average rainbow light over raindrop sizes
    rbow: Rainbow class (NOT Rainbow object)
      either Rainbow, YoungRainbow, AiryRainbow or MieRainbow
    theta: float, 1d-array
      angle between sun and raindrop / radian
    a: float, scalar
      mean radius of raindrop distribution / m
    sigma: float, scalar
      standard deviation of a / m
      if sigma <= 0, averaging is not performed
    width: float, scalar
      width of averaging interval (0 < width < 1)
    wavlen: float, scalar
       wavelength / m
       if wavlen is None, wavelength averaging is performed over
       the solar spectrum (see WavlenAveraging.py)
    rv: instance of scipy.stats.rv_continuous
       distribution function of raindrop sizes
       if rv is None, lognormal distribution is used
    N_dropsize: int, scalar
      number of averaging points
    kw: dictionary
      keyword arguments passed to Rainbow.averaged_intensity()
      except that kw['ord_max'] is passed to MieRainbow.__init__()
    return: I
      I: float, shape(len(theta),) if wavlen is not None,
                shape(3 or 4, len(theta)) if wavlen is None
        averaged intensity
    """
    a = np.atleast_1d(a)
    if sigma > 0: 
        if rv is None: rv = lognorm(1) # lognormal distribution
        x1,x2 = rv.ppf(((1-width)/2, (1+width)/2))
        x = np.linspace(x2, x1, N_dropsize)
        p = rv.pdf(x)
        # radius of raindrop (in decreasing order)
        a = a + sigma*(x - rv.mean())/rv.std()
        if a[-1]<0: raise RuntimeError("a<0")

    if wavlen is None:
        I = [WavlenAveraging(rbow, theta, a, **kw) for a in a]
    else:
        m = IndexFromWavlen(wavlen)
        x = 2*np.pi*a/wavlen
        if rbow == MieRainbow and 'ord_max' in kw:
            ord_max = kw.pop('ord_max')
            r = [rbow(m, x, ord_max) for x in x]
        elif issubclass(rbow, Rainbow):
            r = [rbow(m,x) for x in x]
        else:
            raise RuntimeError('rbow is not Rainbow')

        I = [r.averaged_intensity(theta, **kw) for r in r]

    if sigma <= 0: return np.squeeze(I)

    I = np.einsum('i,i...', p,I)
    return I/np.sum(p)
