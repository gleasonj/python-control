import control.sets as cset

import numpy as np

import matplotlib.pyplot as plt

B2 = cset.Ball2NormSupportSet(np.zeros(2), 1.0)
H2 = cset.Hyperrectangle(-np.ones(2), np.ones(2))

S = B2 * H2

X = cset.UniformRejectionSampler(S).sample(3000)

fig = plt.figure()
ax = fig.add_subplot(121)
ax.scatter(X[0, :], X[1, :])

ax = fig.add_subplot(122)
ax.scatter(X[2, :], X[3, :])

plt.show()