import numpy as np
import matplotlib.pyplot as plt
from LavenDiagram import LavenDiagram
from MieRainbow import MieRainbow as Rainbow
#from AiryRainbow import AiryRainbow as Rainbow
from scipy.constants import degree

a = [5e-4, 3e-4, 2e-4, 1e-4, 5e-5, 2e-5] # radius of raindrop
label = ['$a$ = 0.5 mm', '$a$ = 0.3 mm', '$a$ = 0.2 mm',
         '$a$ = 0.1 mm', '$a$ = 0.05 mm', '$a$ = 0.02 mm']
W = 256 # image width / pixel
N = 512 # number of grid points to evaluate intensity
phi = 30*degree # rainbow extent
th1 = 42*degree # radius at bottom right
th2 = 51*degree # radius at top left
gamma = 0.4 # gamma correction

x = th1*np.sin(phi) # image width / radian
y1 = th1*np.cos(phi) # bottom line from antisolar point
y2 = th2 # top line from antisolar point

plt.figure(figsize=(8,7.97))

L = LavenDiagram(Rainbow, a, 0,x,y1,y2, W,N)

for i,L in enumerate(L):
    plt.subplot(3,2,i+1)
    plt.imshow(L**gamma, origin='lower')
    plt.tick_params(left=False, labelleft=False,
                    bottom=False, labelbottom=False)
    plt.box('off')
    plt.axis('scaled')
    plt.text(W/2 ,len(L)-1, label[i], va='bottom', ha='center')

plt.tight_layout()
plt.savefig('fig19.eps')
plt.show() # takes about 1 minute
