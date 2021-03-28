import numpy as np
from scipy.special import riccati_jn,riccati_yn,lpmn
from Rainbow import Rainbow

class MieRainbow(Rainbow):
    """ Mie theory of light scattering by spheres
    reference: C. F. Bohren and D. R. Huffman
      "Absorption and Scattering of Light by Small Particles" chapter 4
    """
    def __init__(self, m, x):
        """
        m = refractive index of raindrop
        x = 2pi*(radius of raindrop)/(wavelength of light)
        """
        mx = m*x
        n = int(x + 4*x**(1/3) + 2.5)
        psi,dpsi = riccati_jn(n,x)
        chi,dchi = riccati_yn(n,x)
        psm,dpsm = riccati_jn(n,mx)
        xi = psi - chi*1j
        dxi = dpsi - dchi*1j
        # Bohren and Huffman, eq(4.56),(4.57)
        a = (m*psi*dpsi - psi*dpsm)/(m*psm*dxi - xi*dpsm)
        b = (psm*dpsi - m*psi*dpsm)/(psm*dxi - m*xi*dpsm)

        self.n = n
        self.a = a[1:]
        self.b = b[1:]
        self.x = x

    def scattering_amplitude(self, theta, pol=1):
        """
        theta = angle of scattering
        pol = polarization state (0,1,2)
              0 for no polarization
              1 for vertical polarization
              2 for parallel polarization
        """
        c = np.atleast_1d(np.cos(theta)) # vectorize
        s = np.expand_dims(np.sin(theta), -1)
        p = np.moveaxis([lpmn(1, self.n, c) for c in c], 0, 2)
        pi = p[0,1,:,1:]/s
        tau = -p[1,1,:,1:]*s
        n = np.arange(1, self.n + 1)
        k = (2*n+1)/n/(n+1) # Bohren and Huffman, eq(4.74)
        if pol==1:   S = np.dot(self.a * pi + self.b * tau, k)
        elif pol==2: S = np.dot(self.a * tau + self.b * pi, k)
        else: raise RuntimeError("pol==1 or pol==2")
        return np.squeeze(S)/self.x

    def intensity(self, theta, pol=0):
        """ scattering amplitude squared """
        I = 0
        if pol==0 or pol==1:
            I += np.abs(self.scattering_amplitude(theta, 1))**2
        if pol==0 or pol==2:
            I += np.abs(self.scattering_amplitude(theta, 2))**2
        if pol==0:
            I *= 0.5
        return I

    def degree_of_polarization(self, theta):
        """ Bohren and Huffman, eq(4.78) """
        I1 = self.intensity(theta, 1)
        I2 = self.intensity(theta, 2)
        return (I1-I2)/(I1+I2)

    def scattering_efficiency(self):
        """ Bohren and Huffman, eq(4.61) """
        n = np.arange(1, self.n + 1)
        return 2*(np.linalg.norm(np.sqrt(2*n+1)*self.a)
                  + np.linalg.norm(self.b))/self.x**2

    def extinction_efficiency(self):
        """ Bohren and Huffman, eq(4.62) """
        n = np.arange(1, self.n + 1)
        return 2*np.sum((2*n+1)*np.real(self.a + self.b))/self.x**2
