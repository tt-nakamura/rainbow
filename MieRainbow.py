import numpy as np
from scipy.special import riccati_jn, riccati_yn, lpmn
from Rainbow import Rainbow

class MieRainbow(Rainbow):
    """ Mie theory of light scattering by spheres
    reference:
      C. F. Bohren and D. R. Huffman
        "Absorption and Scattering of Light by Small Particles" chapter 4
      E. A. Hovenac and J. A. Lock
        Journal of the Optical Society of America A9 (1992) 781
    """
    def __init__(self, m, x, ord_max=2):
        """
        m: float, scalar
          refractive index
        x: float, scalar
          2pi*(radius of raindrop)/(wavelength of light)
        comment:
          if x is too large (> 4400), this program doesn't work
          (because riccati_jn returns nan)
        """
        y = m*x
        n = int(x + 4*x**(1/3) + 2.5)
        Jx,dJx = riccati_jn(n,x)
        Nx,dNx = riccati_yn(n,x)
        Jy,dJy = riccati_jn(n,y)
        Ny,dNy = riccati_yn(n,y)
        H1x = Jx + Nx*1j
        dH1x = dJx + dNx*1j
        a,b,c,d = Jx*dJy, dJx*Jy, H1x*dJy, dH1x*Jy
        a,b = (a - m*b)/(c - m*d), (m*a - b)/(m*c - d)

        self.x = x
        self.n = n
        self.a = a[1:]
        self.b = b[1:]

        if ord_max is None: return

        # Debye Series (Hovenac and Lock)
        H1y = Jy + Ny*1j
        H2x = np.conj(H1x)
        H2y = np.conj(H1y)
        dH1y = dJy + dNy*1j
        dH2x = np.conj(dH1x)
        dH2y = np.conj(dH1y)

        a,b = H1x*dH2y, dH1x*H2y
        Da,Db = a - m*b, m*a - b

        T12a,T12b = -2j/Da, -2j/Db
        T21a,T21b = m*T12a, m*T12b

        a,b = dH1x*H1y, H1x*dH1y
        R11a,R11b = (m*a - b)/Da, (a - m*b)/Db
        a,b = dH2x*H2y, H2x*dH2y
        R22a,R22b = (m*a - b)/Da, (a - m*b)/Db

        order = np.arange(ord_max+1)
        ap = R11a[:,np.newaxis]**order
        bp = R11b[:,np.newaxis]**order
        ap = -T21a * ap.T * T12a
        bp = -T21b * bp.T * T12b
        ap = np.vstack((1-R22a, ap))/2
        bp = np.vstack((1-R22b, bp))/2

        self.ap = ap[:,1:]
        self.bp = bp[:,1:]

    data = {'theta': np.array(0),
            'n_max': 0} # to save

    def scattering_amplitude(self, theta, order=None, pol=0):
        """
        theta: float, any shape
          angle of scattering / radian
        order: int, scalar or 1d-array
          number of internal reflections (>= -1).
          order == p-1 of Hovenac and Lock (p>=0).
          if order is None, Debye series is not used and
          full amplitude (for all orders) is computed.
          if order is 1d-array, sum of amplitudes for
          those orders is computed.
        pol: int, scalar
          polarization state (0,1,2)
          0 for unpolarized light
          1 for perpendicular polarization
          2 for parallel polarization
        return: S
          S: complex, same shape as theta if pol==1 or 2
            scattering amplitude
            Bohren and Huffman eq(4.74), DIVIDE BY x.
            if pol==0, S.shape = (2,) + theta.shape and
            S[0] and S[1] are for perpendicular and parallel
            polarizations, respectively
        """
        t = np.asarray(theta)
        if(self.data['n_max'] >= self.n and
           self.data['theta'].size == t.size  and
           np.all(self.data['theta'] == t.reshape(-1))):
           pi = self.data['pi'][:,:self.n] # load data
           tau = self.data['tau'][:,:self.n]
        else:
            c,s = np.cos(theta), np.sin(theta)
            P = [lpmn(1, self.n, c) for c in c.flat]
            P = np.moveaxis(P, 0, -1)
            pi = np.moveaxis(P[0,1,1:]/s, -1, -2)
            tau = np.moveaxis(-P[1,1,1:]*s, -1, -2)
            self.data['theta'] = t.flatten() # save data
            self.data['n_max'] = self.n
            self.data['pi'] = pi
            self.data['tau'] = tau

        n = np.arange(1, self.n + 1)
        k = (2*n+1)/n/(n+1)
        if order is None:
            a,b = self.a, self.b
        else:
            p = np.asarray(order) + 1
            a,b = self.ap[p], self.bp[p]

        a,b = np.expand_dims((a,b), -2)
        if   pol==1: S = np.dot(a * pi + b * tau, k)
        elif pol==2: S = np.dot(a * tau + b * pi, k)
        elif pol==0: S = [np.dot(a * pi + b * tau, k),
                          np.dot(a * tau + b * pi, k)]
        else: raise RuntimeError("bad polarization")
        if a.ndim>2: S = np.sum(S, axis=-2)
        if pol==0: S = np.reshape(S, (2,) + theta.shape)
        else: S = S.reshape(theta.shape)
        return S/self.x # divide by x

    def intensity(self, theta, order=None, pol=0):
        """ scattering amplitude squared
        see scattering_amplitude() for theta, order, pol
        return: I
          I: float, same shape as theta
            intensity of scattered light.
            if pol==0, I = (|S[0]|^2 + |S[1]|^2)/2
        """
        I = np.abs(self.scattering_amplitude(theta,order,pol))**2
        if pol==0: return np.mean(I, axis=0)
        else: return I
        
    def degree_of_polarization(self, theta, order=None):
        """ Bohren and Huffman, eq(4.78) """
        I1 = self.intensity(theta,order,1)
        I2 = self.intensity(theta,order,2)
        return (I1-I2)/(I1+I2)
