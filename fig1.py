import numpy as np
import matplotlib.pyplot as plt

m = 1.333 # refractive index
title = ['primary', 'secondary']

alpha = np.arccos(np.sqrt((m**2 - 1)/np.r_[3,8]))

plt.figure(figsize=(5, 2.5))
           
for i,alpha in enumerate(alpha):
    plt.subplot(1,2,i+1)
    beta = np.arcsin(np.sin(alpha)/m)
    th = np.pi - alpha - np.arange(i+3)*(np.pi - 2*beta)
    th1 = th[-1] - alpha # direction of exiting ray
    z = np.exp(1j*th)
    z = np.r_[-2+1j*np.imag(z[0]), z, z[-1] + 3*np.exp(1j*th1)]
    plt.plot(np.real(z), np.imag(z), 'r')

    z = np.exp(1j*np.linspace(0, 2*np.pi, 32)) # draw circle
    plt.plot(np.real(z), np.imag(z), 'b')
    plt.axis('scaled')
    plt.axis('off')
    plt.box('off')
    plt.axis([-1.5,1.01,-1.7,1.01])
    plt.title(title[i])

plt.tight_layout()
plt.savefig('fig1.eps')
plt.show()
