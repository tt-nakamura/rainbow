import numpy as np
from scipy.optimize import newton
from scipy.integrate import quad

SOLAR_RADIUS = 1919/2/3600/180*np.pi # radian

class Rainbow:
    """ Descartes theory of geometric optics """
    def __init__(self, m):
        """ m = refractive index of raindrop """
        alpha_r = [np.arccos(np.sqrt((m*m-1)/3)), # primary
                   np.arccos(np.sqrt((m*m-1)/8))] # secondary
        beta_r = [np.arccos(2*np.sqrt((m*m-1)/3)/m),
                  np.arccos(3*np.sqrt((m*m-1)/8)/m)]
        theta_r = [2*alpha_r[0] - 4*beta_r[0] + np.pi,
                   6*beta_r[1] - 2*alpha_r[1]]
        theta_g = [2*np.pi - 4*np.arcsin(1/m),
                   6*np.arcsin(1/m) - np.pi]
        self.m = m
        self.alpha_r = np.asarray(alpha_r) # rainbow angle of incidence
        self.beta_r  = np.asarray(beta_r)  # rainbow angle of refraction
        self.theta_r = np.asarray(theta_r) # rainbow angle of scattering
        self.theta_g = np.asarray(theta_g) # grazing incidence

    def order(self, theta):
        """ return primary or sedondary (0 or 1)
            for given scattenring angle theta
            assume 0 <= theta <= pi
        """
        if   theta > self.theta_r[0]: return 0
        elif theta < self.theta_r[1]: return 1
        else: return None

    def angle_of_incidence(self, theta, order, alpha=0):
        """
        theta = angle between sun and raindrop / radian
        order = primary or secondary (0 or 1)
        alpha = initial guess for the angle of incidence / radian
        return alpha = angle of incidence
        assume 0 <= theta <= pi
        """
        if order==0:   beta = lambda x: (2*x + np.pi - theta)/4
        elif order==1: beta = lambda x: (2*x + theta)/6
        else: return None
        d_beta = 1/(order+2)
        return newton(lambda x: np.sin(x) - self.m*np.sin(beta(x)), alpha,
                      lambda x: np.cos(x) - self.m*np.cos(beta(x))*d_beta)

    def ray_intensity(self, alpha, order, pol=0):
        """
        alpha = angle of incidence / radian
        order = primary or secondary (0 or 1)
        pol = polarization state (0,1,2);
              0 for no polarization
              1 for vertical polarization
              2 for parallel polarization
        """
        beta = np.arcsin(np.sin(alpha)/self.m)
        db_da = np.cos(alpha)/self.m/np.cos(beta)
        if order==0:   gamma = 2*alpha - 4*beta + np.pi
        elif order==1: gamma = 6*beta - 2*alpha
        else: return None
        dg_da = 2*(1 - (order+2)*db_da)
        e = 0
        if pol==0 or pol==1:# Fresnel formula for vertical polarization
            R = (np.sin(alpha-beta)/np.sin(alpha+beta))**2
            e += R**(order+1) * (1-R)**2
        if pol==0 or pol==2:# Fresnel formula for parallel polarization
            R = (np.tan(alpha-beta)/np.tan(alpha+beta))**2
            e += R**(order+1) * (1-R)**2
        if pol==0: e /= 2
        return e*np.sin(2*alpha)/2/np.sin(gamma)/np.abs(dg_da)

    def intensity(self, theta, pol=0):
        """
        theta = angle between sun and raindrop
        return sum of intensities of two rays (if two exist)
        pol = polarization state (0,1,2)
        assume 0 <= theta <= pi
        """
        if not np.isscalar(theta): # vectorize
            return np.asarray([self.intensity(th,pol) for th in theta])

        order = self.order(theta)
        if order is None: return 0

        alpha = self.angle_of_incidence(theta, order)
        I = self.ray_intensity(alpha, order, pol)
        if ((order==0 and theta <= self.theta_g[0]) or
            (order==1 and theta >= self.theta_g[1])):
            alpha = self.angle_of_incidence(theta, order, np.pi/2)
            I += self.ray_intensity(alpha, order, pol)
        return I

    def averaged_intensity(self, theta, pol=0, r=SOLAR_RADIUS):
        """
        theta = angle between sun and raindrop / radian
        pol = polarization state (0,1,2)
        r = radius of source (of disk shape) / radian
        return intensity smeared out by finite source size
        """
        if not np.isscalar(theta): # vectorize
            return np.asarray([self.averaged_intensity(th,pol,r)
                               for th in theta])
        r2 = r**2
        s = quad(lambda x: self.intensity(theta + x, pol)*np.sqrt(r2 - x**2),
                 -r,r)[0] # smear out by integrating over disk surface
        return s*2/np.pi/r2
