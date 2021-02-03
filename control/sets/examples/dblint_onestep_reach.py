import numpy as np

import control
import control.sets as cset

import matplotlib.pyplot as plt

sys = control.ss(np.array([[0, 1], [0, 0]]), np.array([[0, 1]]).T, np.eye(2),
    np.atleast_2d(np.zeros(2)).T).sample(0.2)

X = cset.to_supportset(cset.Hyperrectangle(-np.ones(2), np.ones(2)))
U = cset.to_supportset(cset.Hyperrectangle(np.array([-0.1]), np.array([0.1])))
Y = np.linalg.inv(sys.A) @ (X + (-sys.B @ U))

Z = cset.ConvexWalkSampler(Y).sample(3000)
F = cset.ConvexWalkSampler(X).sample(3000)

plt.scatter(Z[0, :], Z[1, :], c='b')
plt.scatter(F[0, :], F[1, :], c='r')
plt.show()

