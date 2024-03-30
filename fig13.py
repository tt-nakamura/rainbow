import numpy as np
import matplotlib.pyplot as plt
from Rainbow import Rainbow
from YoungRainbow import YoungRainbow
from AiryRainbow import AiryRainbow
from MieRainbow import MieRainbow
from RefractiveIndex import WavlenFromIndex
from scipy.constants import degree

m = 1.331 # refractive index
a = 2e-4 # radius of raindrop / m
label = ['Descartes', 'Young', 'Airy', 'Mie']

l = WavlenFromIndex(m) # wavelength of red light / m
x = 2*np.pi*a/l

R = Rainbow(m)
Y = YoungRainbow(m,x)
A = AiryRainbow(m,x)
M = MieRainbow(m, x, ord_max=None)

plt.figure(figsize=(5,8))

# primary rainbow
theta_deg = np.linspace(136, 143, 200)
theta = theta_deg*degree

# perpendicular polarization
plt.subplot(411)

plt.plot(theta_deg, R.averaged_intensity(theta, 1, 1))
plt.plot(theta_deg, Y.averaged_intensity(theta, 1, 1))
plt.plot(theta_deg, A.averaged_intensity(theta, 1, 1))
plt.plot(theta_deg, M.averaged_intensity(theta, None, 1))

plt.legend(label)
plt.axis([136, 143, 0, 1.3])
plt.text(136, 1.3, '(a)  primary perpendicular', va='top')
plt.text(140.5, 1, r'$\lambda$ = 694 nm', ha='right')
plt.text(140.5, 0.85, '$a$ = 0.2 mm', ha='right')
plt.ylabel('$I$ = intensity')

# parallel polarization
plt.subplot(412)

plt.plot(theta_deg, R.averaged_intensity(theta, 1, 2))
plt.plot(theta_deg, Y.averaged_intensity(theta, 1, 2))
plt.plot(theta_deg, A.averaged_intensity(theta, 1, 2))
plt.plot(theta_deg, M.averaged_intensity(theta, None, 2))

plt.axis([136, 143, 0, 0.08])
plt.text(136, 0.08, '(b)  primary parallel', va='top')
plt.ylabel('$I$ = intensity')

# secondary rainbow
theta_deg = np.linspace(124, 131, 200)
theta = theta_deg*degree

# perpendicular polarization
plt.subplot(413)

plt.plot(theta_deg, R.averaged_intensity(theta, 2, 1))
plt.plot(theta_deg, Y.averaged_intensity(theta, 2, 1))
plt.plot(theta_deg, A.averaged_intensity(theta, 2, 1))
plt.plot(theta_deg, M.averaged_intensity(theta, None, 1))

plt.axis([124, 131, 0, 0.2])
plt.text(124, 0.2, '(c)  secondary perpendicular', va='top')
plt.ylabel('$I$ = intensity')

# parallel polarization
plt.subplot(414)

plt.plot(theta_deg, R.averaged_intensity(theta, 2, 2))
plt.plot(theta_deg, Y.averaged_intensity(theta, 2, 2))
plt.plot(theta_deg, A.averaged_intensity(theta, 2, 2))
plt.plot(theta_deg, M.averaged_intensity(theta, None, 2))

plt.axis([124, 131, 0, 0.024])
plt.text(124, 0.024, '(d)  secondary parallel', va='top')
plt.ylabel('$I$ = intensity')
plt.xlabel(r'$\theta$ = angle between sun and raindrop / deg')

plt.tight_layout()
plt.savefig('fig13.eps')
plt.show()
