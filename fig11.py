import numpy as np
import matplotlib.pyplot as plt
from MieRainbow import MieRainbow

deg = np.pi/180
a = 2e-4 # radius of raindrop / m
m = [1.331, 1.333, 1.335, 1.337]# refractive index
l = [656.3e-9, 589.3e-9, 532.6e-9, 486.1e-9]# wavelength / m
color = ['r', 'y', 'g', 'b']
label = ['RED', 'YELLOW', 'GREEN', 'BLUE']

rbow = [MieRainbow(m, 2*np.pi*a/l) for m,l in zip(m,l)]

plt.figure(figsize=(8, 3.5))
plt.suptitle('Mie theory of light scattering', y=1)

# primary rainbow
theta = np.linspace(136*deg, 143*deg, 200)
plt.subplot(1,2,2)
for i,r in enumerate(rbow):
    I = r.intensity(theta, 1)
    plt.plot(theta/deg, I, color[i])

plt.axis([136, 143, 0, 1.2])
plt.legend(label)
plt.xlabel(r'$\theta$ = angle between sun and raindrop')
plt.tick_params(axis='y', labelleft=False)

# secondary rainbow
theta = np.linspace(124*deg, 131*deg, 250)
plt.subplot(1,2,1)
for i,r in enumerate(rbow):
    I = r.intensity(theta, 1)
    plt.plot(theta/deg, I, color[i])

plt.axis([124, 131, 0, 1.2])
plt.legend(label)
plt.xlabel(r'$\theta$ = angle between sun and raindrop')
plt.ylabel('I = intensity of rainbow light')

plt.tight_layout()
plt.show()
