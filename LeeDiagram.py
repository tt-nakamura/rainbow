# reference:
#   R. L. Lee, Jr. "Mie theory, Airy theory, and
#     the natural rainbow" Applied Optics 37 (1998) 1506

import numpy as np
from WavlenAveraging import WavlenAveraging
from scipy.stats import norm
from scipy.ndimage import convolve1d

def LeeDiagram(rbow, theta, a,
               sigma=0, h=0.05, width=0.95, **kw):
    """ diagram of rainbow colors on (theta,a) plane
    rbow: Rainbow class (NOT Rainbow object)
      either AiryRainbow or MieRainbow
    theta: float, 1d-array
      angle between sun and raindrop / radian
    a: float, 1d-array
      radius of raindrop / m
      assume a is geomspace in increasing order.
      if sigma >= h, then a is mean radius
    sigma: float, scalar
      log(1 + sigma) == standard deviation of log(a)
      assuming normal distribution of log(a).
      if sigma < h, averaging over a is not performed
    h: float, scalar
      log(1+h) == step size of averaging integral over log(a)
      h is ignored if sigma < h
    width: float, scalar
      range of averaging integral over log(a) (0 < width < 1)
      width is ignored if sigma < h
    kw: dictionary
      keyword arguments passed to WavlenAveraging.
    return: img
      img: float, shape(len(a), len(theta), 3)
        Lee diagram, to be input to matplotlib.pyplot.imshow()
    comment: if rbow == Mie, averaging over a is very
      time consuming. To reduce time, set kw['N_wavlen'] = 16.
    """
    if sigma < h:
        I = [WavlenAveraging(rbow, theta, a, **kw).T
             for a in a[::-1]] # decreasing order
    else: # averaging over radius of raindrop
        dlna = np.diff(np.log(a))
        if not np.allclose(dlna, dlna[0]):
            raise RuntimeError('a is not geomspace')
        N = int(np.ceil(dlna[0]/np.log1p(h)))
        h = dlna[0]/N
        b = np.geomspace(a[0], a[-1], N*len(dlna) + 1)

        d = np.log1p(sigma)*norm.ppf((1+width)/2)
        M = int(np.floor(d/h))
        x = h*np.arange(-M, M+1)
        y = np.exp(x)
        w = norm.pdf(x)
        b = np.r_[b[0]*y[:M], b, b[-1]*y[M+1:]]
        I = [WavlenAveraging(rbow, theta, b, **kw).T
             for b in b[::-1]] # decreasing order
        I = convolve1d(I, w, axis=0) # averaging by convolution
        I = I[M:-M:N]

    img = I/np.max(I, axis=(1,2)).reshape(len(a),1,1)
    return img[::-1]
