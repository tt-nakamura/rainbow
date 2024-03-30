import numpy as np
from Rainbow import Rainbow

class YoungRainbow(Rainbow):
    """ Young theory of interference """
    def __init__(self, m, x):
        """
        m: float, scalar
          refractive index of raindrop
        x: float, scalar
          2pi * (radius of raindrop)/(wavelength of lignt)
        """
        super().__init__(m)
        self.x = x

    def phase(self, alpha, order):
        """
        alpha: float, scalar
          angle of incidence / radian
        order: int, scalar
          primary or secondary (1 or 2)
        return: phi
          phi: float, scalar
            optical path length / radian
            (up to additive const)
        """
        beta = np.arcsin(np.sin(alpha)/self.m)
        return self.x*(2*(1-np.cos(alpha)) + 2*(order+1)*self.m*np.cos(beta))

    def ray_amplitude(self, alpha, order, pol):
        """
        alpha: float, scalar
          angle of incidence / radian
        order: int, scalar
          primary or secondary (1 or 2)
        pol: int, scalar
          polarization state (0,1,2);
          0 for unpolarized light
          1 for perpendicular polarization
          2 for parallel polarization
        return: A
          A: complex, scalar if pol==1 or 2,
                      1d-array if pol==0
            amplitude of outgoing ray.
            if pol==0, A[0] and A[1] are for pependicular and
            parallel polarizaions, respectively
        """
        beta = np.arcsin(np.sin(alpha)/self.m)
        db_da = np.cos(alpha)/self.m/np.cos(beta)
        if   order==1: gamma = 2*alpha - 4*beta + np.pi
        elif order==2: gamma = 6*beta - 2*alpha
        else: raise RuntimeError("bad order")
        dg_da = 2*(1 - (order+1)*db_da)

        if pol==1:# Fresnel formula for perpendicular polarization
            R = np.sin(alpha-beta)/np.sin(alpha+beta)
        elif pol==2:# Fresnel formula for parallel polarization
            R = np.tan(beta-alpha)/np.tan(alpha+beta)
        elif pol==0:
            R = np.r_[np.sin(alpha-beta)/np.sin(alpha+beta),
                      np.tan(beta-alpha)/np.tan(alpha+beta)]
        else: raise RuntimeError("bad polarization")

        e = R**order * (1 - R**2)
        A = np.sin(2*alpha)/2/np.sin(gamma)/dg_da
        p = self.phase(alpha, order)
        return e*np.sqrt(complex(A))*np.exp(1j*p)

    def intensity(self, theta, order=1, pol=0):
        """
        theta: float, any shape
          angle between sun and raindrop / radian
        order: int, scalar
          primary or secondary (1 or 2)
        pol: int, scalar
          polarization state (0,1,2);
          0 for unpolarized light
          1 for perpendicular polarization
          2 for parallel polarization
        return: I
          I: float, same shape as theta
            sum of intensities of two rays (if two exist)
        """
        if not np.isscalar(theta): # vectorize
            t = np.asarray(theta)
            I = [self.intensity(t, order, pol) for t in t.flat]
            return np.reshape(I, t.shape)

        if((order==1 and theta <= self.theta_r[0]) or
           (order==2 and theta >= self.theta_r[1])): return 0
        
        alpha = self.angle_of_incidence(theta, order)
        A = self.ray_amplitude(alpha, order, pol)
        if ((order==1 and theta <= self.theta_g[0]) or
            (order==2 and theta >= self.theta_g[1])):
            alpha = self.angle_of_incidence(theta, order, np.pi/2)
            A += self.ray_amplitude(alpha, order, pol)

        A = np.abs(A)**2
        if pol==0: return np.mean(A)
        else: return A
