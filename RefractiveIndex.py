# refrence:
#   D. K. Lynch and W. Livingston
#     "Color and Light in Nature" p117

from scipy.interpolate import interp1d

# wavelength / m
wavlen = [1000e-9, 900e-9, 800e-9, 700e-9, 650e-9, 600e-9,
          550e-9, 500e-9, 450e-9, 400e-9, 350e-9, 300e-9]

# refractive index
m =[1.3277, 1.3285, 1.3294, 1.3309, 1.3318, 1.3335,
    1.3344, 1.3364, 1.3411, 1.3440, 1.3501, 1.3532]

IndexFromWavlen = interp1d(wavlen, m, 'cubic', fill_value='extrapolate')
WavlenFromIndex = interp1d(m, wavlen, 'cubic', fill_value='extrapolate')
