import numpy as np
from scipy.special import airy
from Rainbow import Rainbow

class AiryRainbow(Rainbow):
    """ Airy theory of diffraction
    references:
      H. C. van de Hulst
        "Light Scattering by Small Particles" section 13.2
      G. P. Koennen and J. H. de Boer
        "Polarlized Rainbow" Applied Optics 18 (1979) 1961
    """
    def __init__(self, m, x):
        """
        m: float, scalar
          refractive index of raindrop
        x: float, scalar
          2pi*(radius of raindrop)/(wavelength of light)
        """
        super().__init__(m)
        self.x = x

    def intensity(self, theta, order=1, pol=1):
        """ stationary phase approximation of Kirchhoff integral
        theta: float, any shape
          angle between sun and raindrop / radian
        order: int, scalar
          primary or secondary (1 or 2)
        pol: int, scalar
          polarization state (0,1,2)
          0 for unpolarized light
          1 for perpendicular polarization
          2 for parallel polarization
        return: I
          I: float, same shape as theta
            Kirchhoff integral squared
        """
        if   order==1: a = 3/4
        elif order==2: a = 8/9
        else: raise RuntimeError("bad order")

        j = order - 1
        alpha = self.alpha_r[j] # rainbow angle
        beta = self.beta_r[j]
        sa,ca = np.sin(alpha), np.cos(alpha)
        tau = (a*sa*self.x)**(1/3)
        theta0 = tau/self.x/ca # fringe width
        if order==2: theta0 = -theta0
        x = (self.theta_r[j] - theta)/theta0
        Ai = airy(x)

        if pol!=2: # perpendicular polarization
            R = np.sin(alpha - beta)/np.sin(alpha + beta)
            e = R**order * (1 - R**2)
            I = (e * Ai[0])**2
        else: I = 0

        if pol!=1: # perpendicular polarization
            R = np.tan(beta-alpha)/np.tan(alpha+beta)
            e = R**order * (1 - R**2)
            alpha_b = np.arctan(self.m) # Brewster angle
            t = (alpha - alpha_b) * tau
            if order==1: # Koennen and de Boer, eq(17)
                J = Ai[0]**2 + (Ai[1]/t)**2
            else:
                J = ((1 - x/t**2)*Ai[0])**2 + (2*Ai[1]/t)**2
            I += e**2 * J

        if pol==0: I /= 2
        return 2*np.pi*sa/self.x/theta0**2/np.sin(theta)*I
