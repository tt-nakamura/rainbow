import numpy as np
import matplotlib.pyplot as plt
from MieRainbow import MieRainbow
from RefractiveIndex import WavlenFromIndex
from scipy.constants import degree

a = 2e-4 # radius of raindrop / m
m = [1.331, 1.333, 1.335, 1.337]# refractive index
color = ['r', 'y', 'g', 'b']

l = WavlenFromIndex(m)# wavelength / m
rbow = [MieRainbow(m, 2*np.pi*a/l) for m,l in zip(m,l)]

plt.figure(figsize=(5,8))

# primary rainbow
theta = np.linspace(137, 142, 100)*degree

# perpendicular polarization
plt.subplot(411)
for i,r in enumerate(rbow):
    I = r.intensity(theta, order=1, pol=1) # Debye Series
    plt.plot(theta/degree, I, color[i])

plt.axis([137, 142, 0, 1.2])
plt.text(137, 1.2, '(a)  primary perpendicular', va='top')
plt.ylabel('$I$ = intensity')

# parallel polarization
plt.subplot(412)
for i,r in enumerate(rbow):
    I = r.intensity(theta, order=1, pol=2) # Debye Series
    plt.plot(theta/degree, I, color[i])

plt.axis([137, 142, 0, 0.1])
plt.text(137, 0.1, '(b)  primary parallel', va='top')
plt.ylabel('$I$ = intensity')

# secondary rainbow
theta = np.linspace(125, 130, 500)*degree

# perpendicular polarization
plt.subplot(413)
for i,r in enumerate(rbow):
    I = r.intensity(theta, order=2, pol=1) # Debye Series
    plt.plot(theta/degree, I, color[i])

plt.axis([125, 130, 0, 0.2])
plt.text(125, 0.2, '(c)  secondary perpendicular', va='top')
plt.legend()
plt.ylabel('$I$ = intensity')

# parallel polarization
plt.subplot(414)
for i,r in enumerate(rbow):
    I = r.intensity(theta, order=2, pol=2) # Debye Series
    plt.plot(theta/degree, I, color[i])

plt.axis([125, 130, 0, 0.03])
plt.text(125, 0.03, '(d)  secondary parallel', va='top')
plt.ylabel('$I$ = intensity')
plt.xlabel(r'$\theta$ = angle between sun and raindrop / deg')

plt.tight_layout()
plt.savefig('fig12.eps')
plt.show()
