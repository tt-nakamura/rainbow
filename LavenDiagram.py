# reference:
#   Philip Laven
#     www.philiplaven.com/p2b.html

import numpy as np
from DropsizeAveraging import DropsizeAveraging
from scipy.interpolate import interp1d

def LavenDiagram(rbow, a, x1, x2, y1, y2, NX,
                 N_theta=256, interp='linear', **kw):
    """ computer graphics of rainbow
    rbow: Rainbow class (NOT Rainbow object)
      either AiryRainbow or MieRainbow
    a: float, scalar or 1d-array
      radius of raindrop / m
    x1,x2,y1,y2: float, scalars
      left,right,bottom,top of drawing region / radian
      measured from anti-solar point
    NX: int, scalar
      number of pixels in [x1,x2]
    N_theta: int, scalar
      number of evaluation points of rainbow intensity
      (to be interpolated)
    interp: string or int, scalar
      what kind of interpolation to use in interp1d
    kw: dictionary
      keyword arguments passed to DropsizeAveraging
      if 'a_sigma' in kw, then dropsize averaging is
      performed with (a, a_sigma) == (mean, std dev)
    return: img
      img: float, shape(NY,NX,3) or (len(a),NY,NX,3)
        where NY = image height / pixel.
        computer graphics of rainbow
        to be input to matplotlib.pyplot.imshow()
    """
    th1 = np.pi - np.hypot(x2,y2)
    th2 = np.pi - np.hypot(x1,y1)
    # evaluation points of rainbow intensity
    theta = np.linspace(th1, th2, N_theta)
    # image height / pixel
    NY = int(np.round(NX/(x2-x1)*(y2-y1)))

    x,y = np.meshgrid(np.linspace(x1,x2,NX),
                      np.linspace(y1,y2,NY))
    th = np.hypot(x,y)

    img = []
    for a in np.atleast_1d(a):
        I = DropsizeAveraging(rbow, theta, a, **kw)
        I = interp1d(theta, I, interp)
        I = np.moveaxis(I(np.pi - th), 0, -1)
        img.append(I/np.max(I)) # normalize

    return np.squeeze(img)
