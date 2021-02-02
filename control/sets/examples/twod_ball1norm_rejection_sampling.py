import numpy as np

import control.sets as cset

import matplotlib.pyplot as plt

B1 = cset.Ball1NormSupportSet(np.zeros(2), 1.0)
X = cset.UniformRejectionSampler(B1).sample(3000)

plt.scatter(X[0, :], X[1, :])
plt.show()