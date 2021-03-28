import numpy as np
import matplotlib.pyplot as plt
from Rainbow import Rainbow

deg = np.pi/180
m = [1.331, 1.333, 1.335, 1.337]# refractive index
color = ['r', 'y', 'g', 'b']
label = ['RED', 'YELLOW', 'GREEN', 'BLUE']

rbow = [Rainbow(m) for m in m]

plt.figure(figsize=(8, 3.5))
plt.suptitle('Descartes theory of geometric optics', y=1)

# primary rainbow
theta = np.linspace(137*deg, 142*deg, 100)
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
