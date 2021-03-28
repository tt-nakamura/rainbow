import numpy as np
import matplotlib.pyplot as plt
from Rainbow import Rainbow
from YoungRainbow import YoungRainbow
from AiryRainbow import AiryRainbow
from MieRainbow import MieRainbow

deg = np.pi/180
m = 1.331 # refractive index
a = 2e-4 # radius of raindrop / m
l = 656.3e-9 # wavelength of red light / m
label = ['Descartes', 'Young', 'Airy', 'Mie']

x = 2*np.pi*a/l
rbow = [Rainbow(m),
        YoungRainbow(m,x),
        AiryRainbow(m,x),
        MieRainbow(m,x)]

plt.figure(figsize=(8, 3.5))
plt.suptitle('interference fringes smeared out by finite source size', y=1)

# primary rainbow
theta = np.linspace(136*deg, 143*deg, 100)
plt.subplot(1,2,2)
for i,r in enumerate(rbow):
    I = r.averaged_intensity(theta, 1)
    plt.plot(theta/deg, I)

plt.axis([136, 143, 0, 1.2])
plt.legend(label)
plt.xlabel(r'$\theta$ = angle between sun and raindrop')
plt.tick_params(axis='y', labelleft=False)

# secondary rainbow
theta = np.linspace(124*deg, 131*deg, 200)
plt.subplot(1,2,1)
for i,r in enumerate(rbow):
    I = r.averaged_intensity(theta, 1)
    plt.plot(theta/deg, I)

plt.axis([124, 131, 0, 1.2])
plt.legend(label)
plt.xlabel(r'$\theta$ = angle between sun and raindrop')
plt.ylabel('I = intensity of rainbow light')

plt.tight_layout()
plt.show() # takes about a minute
