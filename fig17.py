# reference:
#   Philip Laven
#     www.philiplaven.com/p2b.html

import numpy as np
import matplotlib.pyplot as plt
from WavlenAveraging import WavlenAveraging
from AiryRainbow import AiryRainbow as Rainbow
from scipy.constants import degree
from scipy.interpolate import interp1d

a = [5e-4, 3e-4, 2e-4, 1e-4, 5e-5, 2e-5] # radius of raindrop
label = ['$a$ = 0.5 mm', '$a$ = 0.3 mm', '$a$ = 0.2 mm',
         '$a$ = 0.1 mm', '$a$ = 0.05 mm', '$a$ = 0.02 mm']
W = 256 # image width / pixel
N = 256 # number of grid points to evaluate intensity
phi = 30*degree # rainbow extent
th1 = 42*degree # radius at bottom right
th2 = 43*degree # radius at top left

x = th1*np.sin(phi) # image width / radian
y1 = th1*np.cos(phi) # bottom line from antisolar point
y2 = th2 # top line from antisolar point

# evaluation points of rainbow intensity
theta = np.linspace(np.pi - np.hypot(x,y2), np.pi - y1, N)

H = int(W/x*(y2-y1)) # image height / pixel

x,y = np.meshgrid(np.linspace(0,x,W),
                  np.linspace(y1,y2,H))
th = np.hypot(x,y)

plt.figure(figsize=(5,10))

for i,a in enumerate(a):
    I = WavlenAveraging(Rainbow, theta, a, order=1, pol=1) # unpolarized
    I_interp = interp1d(theta, I)
    img = I_interp(np.pi - th)
    img = np.moveaxis(img,0,-1) / np.max(img) # normalize
    plt.subplot(6,1,i+1)
    plt.imshow(img[::-1])
    plt.tick_params(left=False, labelleft=False,
                    bottom=False, labelbottom=False)
    plt.box('off')
    plt.axis('scaled')
    plt.text(W,0, label[i], ha='right', va='top', color='w')

plt.tight_layout()
plt.savefig('fig17.eps')
plt.show()
