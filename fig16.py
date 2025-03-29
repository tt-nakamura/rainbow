import numpy as np
import matplotlib.pyplot as plt
from WavlenAveraging import WavlenAveraging
from Rainbow import Rainbow
from YoungRainbow import YoungRainbow
from AiryRainbow import AiryRainbow
from MieRainbow import MieRainbow
from scipy.constants import degree

a = 2e-4 # radius of raindrop

N = 128 # number of grid points
img = np.ones((23,N,3))

plt.figure(figsize=(5,5.5))

# primary
theta = np.linspace(137, 143, N) * degree

# penpendicular polarization
R = WavlenAveraging(Rainbow, theta, order=1, pol=1)
Y = WavlenAveraging(YoungRainbow, theta, a, order=1, pol=1)
A = WavlenAveraging(AiryRainbow, theta, a, order=1, pol=1)
M = WavlenAveraging(MieRainbow, theta, a, order=None, pol=1)

R = R.T/np.max(R) # normalize
Y = Y.T/np.max(Y)
A = A.T/np.max(A)
M = M.T/np.max(M)

plt.subplot(411)
img[:5] = np.tile(R, (5,1,1))
img[6:11] = np.tile(Y, (5,1,1))
img[12:17] = np.tile(A, (5,1,1))
img[18:] = np.tile(M, (5,1,1))

plt.imshow(img)
plt.text(0, 0, '(a)  primary perpendicular', va='bottom')
plt.xticks(np.r_[0:N-1:7j], ('137','138','139','140','141','142','143'))
plt.tick_params(left=False, labelleft=False)
plt.box('off')

# parallel polarization
R = WavlenAveraging(Rainbow, theta, order=1, pol=2)
Y = WavlenAveraging(YoungRainbow, theta, a, order=1, pol=2)
A = WavlenAveraging(AiryRainbow, theta, a, order=1, pol=2)
M = WavlenAveraging(MieRainbow, theta, a, order=None, pol=2)

R = R.T/np.max(R) # normalize
Y = Y.T/np.max(Y)
A = A.T/np.max(A)
M = M.T/np.max(M)

plt.subplot(412)
img[:5] = np.tile(R, (5,1,1))
img[6:11] = np.tile(Y, (5,1,1))
img[12:17] = np.tile(A, (5,1,1))
img[18:] = np.tile(M, (5,1,1))

plt.imshow(img)
plt.text(0, 0, '(b)  primary parallel', va='bottom')
plt.xticks(np.r_[0:N-1:7j], ('137','138','139','140','141','142','143'))
plt.tick_params(left=False, labelleft=False)
plt.box('off')

# secondary
theta = np.linspace(124, 130, N) * degree

# penpendicular polarization
R = WavlenAveraging(Rainbow, theta, order=2, pol=1)
Y = WavlenAveraging(YoungRainbow, theta, a, order=2, pol=1)
A = WavlenAveraging(AiryRainbow, theta, a, order=2, pol=1)
M = WavlenAveraging(MieRainbow, theta, a, order=None, pol=1)

R = R.T/np.max(R) # normalize
Y = Y.T/np.max(Y)
A = A.T/np.max(A)
M = M.T/np.max(M)

plt.subplot(413)
img[:5] = np.tile(R, (5,1,1))
img[6:11] = np.tile(Y, (5,1,1))
img[12:17] = np.tile(A, (5,1,1))
img[18:] = np.tile(M, (5,1,1))

plt.imshow(img)
plt.text(0, 0, '(c)  secondary perpendicular', va='bottom')
plt.xticks(np.r_[0:N-1:7j], ('124','125','126','127','128','129','130'))
plt.tick_params(left=False, labelleft=False)
plt.box('off')

# parallel polarization
R = WavlenAveraging(Rainbow, theta, order=2, pol=2)
Y = WavlenAveraging(YoungRainbow, theta, a, order=2, pol=2)
A = WavlenAveraging(AiryRainbow, theta, a, order=2, pol=2)
M = WavlenAveraging(MieRainbow, theta, a, order=None, pol=2)

R = R.T/np.max(R) # normalize
Y = Y.T/np.max(Y)
A = A.T/np.max(A)
M = M.T/np.max(M)

plt.subplot(414)
img[:5] = np.tile(R, (5,1,1))
img[6:11] = np.tile(Y, (5,1,1))
img[12:17] = np.tile(A, (5,1,1))
img[18:] = np.tile(M, (5,1,1))

plt.imshow(img)
plt.text(0, 0, '(d)  secondary parallel', va='bottom')
plt.xticks(np.r_[0:N-1:7j], ('124','125','126','127','128','129','130'))
plt.tick_params(left=False, labelleft=False)
plt.box('off')

plt.xlabel(r'$\theta$ = angle between sun and raindrop / deg')

plt.tight_layout()
plt.savefig('fig16.eps')
plt.show()
