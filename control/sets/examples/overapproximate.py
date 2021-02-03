import numpy as np

import control.sets as cset

import matplotlib.pyplot as plt

np.random.seed(8675309)

T = cset.Ball2NormSupportSet(np.zeros(2), 1.0)
S = cset.overapproximate(T, np.vstack((np.eye(2), -np.eye(2))).T)
P = cset.overapproximate(T, 10)

X = cset.UniformRejectionSampler(S).sample(3000)
Y = cset.UniformRejectionSampler(P).sample(3000)
Z = cset.UniformRejectionSampler(T).sample(3000)

plt.scatter(X[0, :], X[1, :], c='b')
plt.scatter(Y[0, :], Y[1, :], c='r')
plt.scatter(Z[0, :], Z[1, :], c='g')
plt.show()