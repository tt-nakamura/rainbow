import numpy as np
import matplotlib.pyplot as plt
from Rainbow import Rainbow
from scipy.constants import degree

m = [1.331, 1.333, 1.335, 1.337]# refractive index
color = ['r', 'y', 'g', 'b']

rbow = [Rainbow(m) for m in m]

plt.figure(figsize=(5,8))

# primary rainbow
theta = np.linspace(137, 142, 103)*degree

# perpendicular polarization
plt.subplot(411)
for i,r in enumerate(rbow):
    I = r.intensity(theta, order=1, pol=1)
    I[I==0] = np.nan
    plt.plot(theta/degree, I, color[i])
    plt.vlines(r.theta_r[0]/degree, 0, 2, color[i],
               label='$m$ = %g'%m[i])

plt.axis([137, 142, 0, 1.2])
plt.text(137, 1.2, '(a)', va='top')
plt.text(142, 0, 'primary perpendicular', va='bottom', ha='right')
plt.legend()
plt.ylabel('$I$ = intensity')

# parallel polarization
plt.subplot(412)
for i,r in enumerate(rbow):
    I = r.intensity(theta, order=1, pol=2)
    I[I==0] = np.nan
    plt.plot(theta/degree, I, color[i])
    plt.vlines(r.theta_r[0]/degree, 0, 2, color[i])

plt.axis([137, 142, 0, 0.1])
plt.text(137, 0.1, '(b)', va='top')
plt.text(142, 0, 'primary parallel', va='bottom', ha='right')
plt.ylabel('$I$ = intensity')

# secondary rainbow
theta = np.linspace(125, 130, 100)*degree

# perpendicular polarization
plt.subplot(413)
for i,r in enumerate(rbow):
    I = r.intensity(theta, order=2, pol=1)
    I[I==0] = np.nan
    plt.plot(theta/degree, I, color[i])
    plt.vlines(r.theta_r[1]/degree, 0, 2, color[i])

plt.axis([125, 130, 0, 0.2])
plt.text(125, 0.2, '(c)  secondary perpendicular', va='top')
plt.legend()
plt.ylabel('$I$ = intensity')

# parallel polarization
plt.subplot(414)
for i,r in enumerate(rbow):
    I = r.intensity(theta, order=2, pol=2)
    I[I==0] = np.nan
    plt.plot(theta/degree, I, color[i])
    plt.vlines(r.theta_r[1]/degree, 0, 2, color[i])

plt.axis([125, 130, 0, 0.03])
plt.text(125, 0.03, '(d)  secondary parallel', va='top')
plt.ylabel('$I$ = intensity')
plt.xlabel(r'$\theta$ = angle between sun and raindrop / deg')

plt.tight_layout()
plt.savefig('fig5.eps')
plt.show()
