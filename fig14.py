import numpy as np
import matplotlib.pyplot as plt
from RGB import wavlen0,wavlen1,red,green,blue,factor

plt.figure(figsize=(5,4))

wavlen_nm = np.asarray(wavlen0)*1e9

plt.subplot(211)
plt.plot(wavlen_nm, red, 'r')
plt.plot(wavlen_nm, green, 'g')
plt.plot(wavlen_nm, blue, 'b')
plt.ylabel('red, green, blue')
plt.xlim((355,800))
plt.text(355, plt.ylim()[1], '(a)', va='top')

wavlen_nm = np.asarray(wavlen1)*1e9

plt.subplot(212)
plt.plot(wavlen_nm, factor)
plt.ylabel('$f$ = fading factor')
plt.xlabel(r'$\lambda$ = wave length / nm')
plt.xlim((355,800))
plt.text(355, plt.ylim()[1], '(b)', va='top')

plt.tight_layout()
plt.savefig('fig14.eps')
plt.show()




