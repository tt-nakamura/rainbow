import numpy as np
from Rainbow import Rainbow

class YoungRainbow(Rainbow):
    """ Young theory of interference """
    def __init__(self, m, x):
        """
        m = refractive index of raindrop
        x = 2pi * (radius of raindrop)/(wavelength of lignt)
        """
        super().__init__(m)
        self.x = x

    def phase(self, alpha, order):
        """
        alpha = angle of incidence
        order = primary or secondary (0 or 1)
        return ray's travelling time (up to additive const)
        """
        beta = np.arcsin(np.sin(alpha)/self.m)
        return self.x*(2*(1-np.cos(alpha)) + 2*(order+2)*self.m*np.cos(beta))

    def intensity(self, theta, pol=0):
        """
        theta = angle between sun and raindrop
        return sum of intensities of two rays (if two exist)
        pol = polarization state (see Rainbow.py)
        """
        if hasattr(theta, '__len__'): # vectorize
            return np.asarray([self.intensity(th,pol) for th in theta])

        order = self.order(theta)
        if order is None: return 0
        
        alpha = self.angle_of_incidence(theta, order)
        I = self.ray_intensity(alpha, order, pol)
        if ((order==0 and theta <= self.theta_g[0]) or
            (order==1 and theta >= self.theta_g[1])):
            alpha2 = self.angle_of_incidence(theta, order, np.pi/2)
            I2 = self.ray_intensity(alpha2, order, pol)
            I += I2 + 2*np.sqrt(I*I2)*np.sin(self.phase(alpha2, order)
                                             - self.phase(alpha, order))
        return I
