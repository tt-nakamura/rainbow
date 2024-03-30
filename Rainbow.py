import numpy as np
from scipy.optimize import newton
from scipy.constants import arcsec

class Rainbow:
    """ Descartes theory of geometric optics """
    def __init__(self, m):
        """
        m: float, scalar
          refractive index of raindrop
        """
        # rainbow angle of incidence for primary and secondary
        a = np.arccos(np.sqrt((m*m-1)/np.r_[3,8]))
        # rainbow angle of refraction
        b = np.arcsin(np.sin(a)/m)
        # rainbow angle of scattering
        r = np.r_[2*a[0] - 4*b[0] + np.pi,
                  6*b[1] - 2*a[1]]
        # grazing incidence
        g = np.r_[2*np.pi - 4*np.arcsin(1/m),
                  6*np.arcsin(1/m) - np.pi]
        self.m = m
        self.alpha_r = a
        self.beta_r = b
        self.theta_r = r
        self.theta_g = g

    def angle_of_incidence(self, theta, order, alpha=0):
        """
        theta: float, scalar
          angle between sun and raindrop / radian
          assume 0 <= theta <= pi
        order: int, scalar
          primary or secondary (1 or 2)
        alpha: float, scalar
          initial guess for the angle of incidence / radian
        return: alpha
          alpha: float, scalar
            angle of incidence / radian
        """
        if   order==1: beta = lambda x: (2*x + np.pi - theta)/4
        elif order==2: beta = lambda x: (2*x + theta)/6
        else: raise RuntimeError("bad order")
        d_beta = 1/(order+1)
        return newton(lambda x: np.sin(x) - self.m*np.sin(beta(x)), alpha,
                      lambda x: np.cos(x) - self.m*np.cos(beta(x))*d_beta)

    def ray_intensity(self, alpha, order, pol):
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
        return: I
          I: float, scalar
            intensity of outgoing ray
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
        e = np.mean((R**order * (1 - R**2))**2)
        return e * np.sin(2*alpha)/2/np.sin(gamma)/np.abs(dg_da)

    def intensity(self, theta, order=1, pol=0):
        """
        theta: float, any shape
          angle between sun and raindrop / radian
          assume 0 <= theta <= pi
        order: int, scalar
          primary or secondary (1 or 2)
        pol: int, scalar
          polarization state (0,1,2)
        return: I
          I: float, same shape as theta
            sum of intensities of two rays (if two exist)
        """
        if not np.isscalar(theta): # vectorize
            t = np.asarray(theta)
            I = [self.intensity(t, order, pol) for t in t.flat]
            return np.reshape(I, t.shape)

        # Alexander's dark band
        if((order==1 and theta <= self.theta_r[0]) or
           (order==2 and theta >= self.theta_r[1])): return 0

        alpha = self.angle_of_incidence(theta, order)
        I = self.ray_intensity(alpha, order, pol)

        # additional ray
        if((order==1 and theta <= self.theta_g[0]) or
           (order==2 and theta >= self.theta_g[1])):
            alpha = self.angle_of_incidence(theta, order, np.pi/2)
            I += self.ray_intensity(alpha, order, pol)

        return I

    SUN_RADIUS = 1919/2*arcsec # radian

    def averaged_intensity(self, theta, order=1, pol=0,
                           r=SUN_RADIUS, dx=1e-3):
        """ intensity averaged over finite source size
        theta: float, scalar or 1d-array
          angle between sun and raindrop / radian
        order: int, scalar
          primary or secondary (1 or 2)
        pol: int, scalar
          polarization state (0,1,2)
        r: float, scalar
          radius of source (of disk shape) / radian
          if r<dx, averaging is not performed
        dx: float, scalar
          step size of integration
        return: I
          I: float, same shape as theta
            averaged intensity
        """
        if r<dx: return self.intensity(theta, order ,pol)
            
        if not np.isscalar(theta):
            dt = np.diff(theta)
            if not np.allclose(dt, dt[0]):
                raise RuntimeError('theta is not linspace')
            N = int(np.ceil(dt[0]/dx))
            dx = dt[0]/N
            t = np.linspace(theta[0], theta[-1], N*len(dt) + 1)
        else: t = [theta]

        M = int(np.floor(r/dx))
        r2 = r**2
        x = dx*np.arange(-M, M+1)
        w = np.sqrt(r2 - x**2)
        t = np.r_[t[0] + x[:M], t, t[-1] + x[M+1:]]
        I = self.intensity(t, order, pol)
        I = np.convolve(I, w, 'valid') # fast averaging by convolution
        if not np.isscalar(theta): I = I[::N]
        else: I = np.squeeze(I)
        return I*dx*2/np.pi/r2
