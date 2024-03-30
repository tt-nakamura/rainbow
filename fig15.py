import numpy as np
import matplotlib.pyplot as plt
from RefractiveIndex import IndexFromWavlen

wavlen_nm = np.linspace(300, 1000, 100)
wavlen = wavlen_nm * 1e-9
m = IndexFromWavlen(wavlen)

plt.figure(figsize=(5,2.3))

plt.plot(wavlen_nm, m)
plt.xlabel(r'$\lambda$ = wave length / nm')
plt.ylabel('$m$ = refractive index')

plt.tight_layout()
plt.savefig('fig15.eps')
plt.show()
