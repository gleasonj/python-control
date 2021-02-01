import control.sets as cset

import matplotlib.pyplot as plt

X = cset.UniformDSphereSampler(2).sample(100)
plt.scatter(X[0, :], X[1, :])
plt.show()