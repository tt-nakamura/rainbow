import numpy as np
from scipy.special import airy
from scipy.integrate import solve_ivp
from Rainbow import Rainbow

class AiryRainbow(Rainbow):
    """ Airy theory of diffraction
    reference: H. C. van de Hulst
      "Light Scattering by Small Particles" section 13.2
    """
    def __init__(self, m, x):
        """
        m = refractive index of raindrop
        x = 2pi*(radius of raindrop)/(wavelength of light)
        """
        super().__init__(m)
        # fringe width for primary and secondary rays
        theta0 = ([3/4, 8/9]*np.sin(self.alpha_r)*x
                  )**(1/3)/x/np.cos(self.alpha_r)
        # Fresnel formula for vertical polarization
        R1 = (np.sin(self.alpha_r - self.beta_r)
              /np.sin(self.alpha_r + self.beta_r))**2
        # Fresnel formula for parallel polarization
        R2 = (np.tan(self.alpha_r - self.beta_r)
              /np.tan(self.alpha_r + self.beta_r))**2
        e1 = R1**[1,2]*(1-R1)**2
        e2 = R2**[1,2]*(1-R2)**2
        # reflection coeff for pol=0,1,2
        e = np.c_[(e1+e2)/2, e1, e2].T

        self.A = 2*np.pi*e*np.sin(self.alpha_r)/x/theta0**2
        self.theta0 = theta0
        self.theta1 = np.sum(self.theta_r)/2

    def intensity(self, theta, pol=0):
        """ stationary phase approximation of Kirchhoff integral
        theta = angle between sun and raindrop
        pol = polarization state (0,1,2)
        return Kirchhoff integral squared
        """
        A = self.A[pol]
        t = np.expand_dims(theta,-1) # vectorize
        x = ((self.theta_r - t)/self.theta0).T
        p = theta >= self.theta1 # primary or secondary
        A = np.where(p, A[0], A[1])
        x = np.where(p, x[0],-x[1])
        return A/np.sin(theta)*(airy(x)[0])**2
