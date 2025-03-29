import numpy as np
import matplotlib.pyplot as plt
from YoungRainbow import YoungRainbow
from DropsizeAveraging import DropsizeAveraging
from RefractiveIndex import WavlenFromIndex
from scipy.constants import degree

a = 2e-4 # mean radius of raindrop / m
sigma = [1e-5, 5e-5, 1e-4, 2e-4] # standard deviation / m
m = 1.331 # refractive index
label = [r'$\sigma$ = 0.01 mm', '0.05 mm', '0.1 mm', '0.2 mm']

l = WavlenFromIndex(m) # wavelength of red light / m

theta = np.linspace(137, 142, 100) * degree

plt.figure(figsize=(5, 3.2))

for i,sigma in enumerate(sigma):
    I = DropsizeAveraging(YoungRainbow, theta,
                          a, sigma, wavlen=l,
                          order=1, pol=1, r=0) # point source
    I[I==0] = np.nan
    plt.plot(theta/degree, I)

r = YoungRainbow(m, 2*np.pi*a/l)
plt.vlines(r.theta_r[0]/degree, 0, 2, 'k')
plt.axis([137, 142, 0, 1.2])
plt.legend(label, markerfirst=False)
plt.text(140.1, 1.1, 'primary perpendicular', ha='right')
plt.text(140.1, 1.0, r'$\langle a\rangle$ = 0.2 mm', ha='right')
plt.text(140.1, 0.9, r'$\lambda$ = 694 nm', ha='right')
plt.xlabel(r'$\theta$ = angle between sun and raindrop / deg')
plt.ylabel('$I$ = intensity of rainbow light')

plt.tight_layout()
plt.savefig('fig7.eps')
plt.show()
