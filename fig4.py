import numpy as np
import matplotlib.pyplot as plt

m = 1.333 # refractive index
deg = np.pi/180
alpha = np.linspace(0, np.pi/2, 100) # angle of incidence
beta = np.arcsin(np.sin(alpha)/m) # angle of reflection by snell law
gamma1 = 2*alpha - 4*beta + np.pi # angle of scattering for primary ray
gamma2 = 6*beta - 2*alpha # angle of scattering for secondary ray

plt.figure(figsize=(5, 3.75))

plt.axis([0, 90, 0, 180])
plt.plot(alpha/deg, gamma1/deg, label='primary')
plt.plot(alpha/deg, gamma2/deg, label='secondary')
plt.xlabel(r'$\alpha$ = angle of incidence / deg')
plt.ylabel(r'$\gamma$ = angle of deviation / deg')
plt.legend()

plt.tight_layout()
plt.savefig('fig4.eps')
plt.show()
