import numpy as np
import matplotlib.pyplot as plt
from YoungRainbow import YoungRainbow

deg = np.pi/180
a = 2e-4 # radius of raindrop / m
m = [1.331, 1.333, 1.335, 1.337]# refractive index
l = [656.3e-9, 589.3e-9, 532.6e-9, 486.1e-9]# wavelength / m
color = ['r', 'y', 'g', 'b']
label = ['RED', 'YELLOW', 'GREEN', 'BLUE']

rbow = [YoungRainbow(m, 2*np.pi*a/l) for m,l in zip(m,l)] 

plt.figure(figsize=(8, 3.5))
plt.suptitle('Young theory of interference', y=1)

# primary rainbow
theta = np.linspace(132*deg, 142*deg, 200)
plt.subplot(1,2,2)
for i,r in enumerate(rbow):
    I = r.intensity(theta, 1); I[I==0] = np.nan
    plt.plot(theta/deg, I, color[i])
    plt.vlines(r.theta_r[0]/deg, 0, 2, color[i])

plt.axis([137, 142, 0, 1.2])
plt.legend(label)
plt.xlabel(r'$\theta$ = angle between sun and raindrop')
plt.tick_params(axis='y', labelleft=False)

# secondary rainbow
theta = np.linspace(125*deg, 130*deg, 200)
plt.subplot(1,2,1)
for i,r in enumerate(rbow):
    I = r.intensity(theta, 1); I[I==0] = np.nan
    plt.plot(theta/deg, I, color[i])
    plt.vlines(r.theta_r[1]/deg, 0, 2, color[i])

plt.axis([125, 130, 0, 1.2])
plt.legend(label, loc='upper left')
plt.xlabel(r'$\theta$ = angle between sun and raindrop')
plt.ylabel('I = intensity of rainbow light')

plt.tight_layout()
plt.show()
