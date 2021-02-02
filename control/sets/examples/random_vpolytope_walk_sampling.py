import numpy as np

import control.sets as cset

import matplotlib.pyplot as plt

np.random.seed(90210)

V = cset.VPolytope(np.random.uniform(-1.0, 1.0, (100, 2)).T)
sampler = cset.ConvexWalkSampler(V)
X = sampler.sample(5000)

plt.scatter(X[0, :], X[1, :], c='b')
plt.scatter(V.V[0, :], V.V[1, :], c='g')
plt.show()