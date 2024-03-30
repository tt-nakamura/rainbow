# reference:
#   Dan Bruton
#     www.physics.sfasu.edu/astro/color/spectra.html

import numpy as np
from scipy.interpolate import interp1d

# wavelength for RGB / m
wavlen0 = [380e-9, 440e-9, 490e-9, 510e-9, 580e-9, 645e-9, 780e-9]
# wavelength for fading factor / m
wavlen1 = [380e-9, 420e-9, 700e-9, 780e-9]

red   = [1, 0, 0, 0, 1, 1, 1]
green = [0, 0, 1, 1, 1, 0, 0]
blue  = [1, 1, 1, 0, 0, 0, 0]
factor = [0.3, 1, 1, 0.3] # fading factor

RGB = interp1d(wavlen0, (red, green, blue), fill_value=(0,0))
F = interp1d(wavlen1, factor, fill_value=(0,0))

def RGBFromWavlen(wavlen, gamma=0.8):
    """
    wavlen: float, any shape
      wavelength / m
    gamma: float scalar
      gamma correction
    return R,G,B:
      R,G,B: float, same shape as wavlen
        luminance (0 <= R,G,B <= 1)
    """
    return (RGB(wavlen)*F(wavlen))**gamma
