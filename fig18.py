import numpy as np
import matplotlib.pyplot as plt
from LeeDiagram import LeeDiagram
from MieRainbow import MieRainbow as Rainbow
#from AiryRainbow import AiryRainbow as Rainbow
from scipy.constants import degree

H,W = 64,127 # image height and width / pixel
theta = np.linspace(124, 143, (W-1)//6*19 + 1)*degree
a = np.geomspace(1e-5, 1e-3, H) # radius of raindrop

# dropsize averaging (log(1 + sigma) == stddev of log(a))
L = LeeDiagram(Rainbow, theta, a, sigma=0.5, N_wavlen=16)

plt.figure(figsize=(5,5.4))

# primary
L1 = L[:,-W:]

plt.subplot(211)
plt.imshow(L1, origin='lower')
plt.text(0, H-1,'(a)  primary unpolarized', va='bottom')
plt.xticks(np.r_[0:W-1:7j], ('137','138','139','140','141','142','143'))
plt.yticks(np.r_[0:H-1:3j], ('0.01', '0.1', '1'))
plt.ylabel('$a$ = radius of raindrop / mm')
plt.box('off')

# secondary
L2 = L[:,:W]
L2 /= np.max(L2, axis=(1,2)).reshape(H,1,1) # normalize

plt.subplot(212)
plt.imshow(L2, origin='lower')
plt.text(0, H-1,'(b)  secondary unpolarized', va='bottom')
plt.xticks(np.r_[0:W-1:7j], ('124','125','126','127','128','129','130'))
plt.yticks(np.r_[0:H-1:3j], ('0.01', '0.1', '1'))
plt.ylabel('$a$ = radius of raindrop / mm')
plt.box('off')

plt.xlabel(r'$\theta$ = angle between sun and raindrop / deg')

plt.tight_layout()
plt.savefig('fig18.eps')
plt.show() # takes about 11 minutes
