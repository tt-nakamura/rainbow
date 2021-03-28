import numpy as np
import matplotlib.pyplot as plt
from YoungRainbow import YoungRainbow

deg = np.pi/180
a = [1e-4, 2e-4, 3e-4, 4e-4] # radius of raindrop / m
m = 1.331 # refractive index
l = 656.3e-9 # wavelength of red light / m
color = ['r', 'r--', 'r:', 'r-.']
label = ['a = 0.1 mm', '0.2 mm', '0.3 mm', '0.4 mm']

r = [YoungRainbow(m, 2*np.pi*a/l) for a in a]

theta = np.linspace(137*deg, 142*deg, 100)

for i,r in enumerate(r):
    I = r.intensity(theta, 1)
    I[I==0] = np.nan
    plt.plot(theta/deg, I, color[i])

plt.vlines(r.theta_r[0]/deg, 0, 2, 'r')
plt.axis([137, 142, 0, 1.2])
plt.legend(label, markerfirst=False)
plt.xlabel(r'$\theta$ = angle between sun and raindrop')
plt.ylabel('I = intensity of rainbow light')
plt.title('a = radius of spherical raindrop')
plt.show()
