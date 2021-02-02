import numpy as np

import control.sets as cset

import matplotlib.pyplot as plt

A = cset.HPolytope(np.vstack((np.eye(2), -np.eye(2))), np.array([0, 0, 1, 1]))
B = cset.HPolytope(np.vstack((-np.eye(2), np.eye(2))), np.array([0, 0, 1, 1]))

X = cset.UniformRejectionSampler(A | B).sample(1000)

plt.scatter(X[0, :], X[1, :])
plt.show()