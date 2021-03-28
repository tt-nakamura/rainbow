import numpy as np
import matplotlib.pyplot as plt

m = 1.33 # refractive index
title = ['primary', 'secondary']

plt.figure(figsize=(6,3))
           
for n in [1,2]:
    plt.subplot(1,2,n)
    for b in np.linspace(0,1,16):
        alpha = np.arcsin(b)
        beta = np.arcsin(b/m)
        th = np.pi - alpha - np.arange(n+2)*(np.pi - 2*beta)
        th1 = th[-1] - alpha # direction of exiting ray
        z = np.exp(1j*th)
        z = np.r_[-2+1j*np.imag(z[0]), z, z[-1] + 3*np.exp(1j*th1)]
        plt.plot(np.real(z), np.imag(z))

    z = np.exp(1j*np.linspace(0, 2*np.pi, 32)) # draw circle
    plt.plot(np.real(z), np.imag(z), 'k')
    plt.axis('equal')
    plt.axis([-2,2,-2,2])
    plt.title(title[n-1])

plt.show()
