import control.sets as cset

import matplotlib.pyplot as plt

X = cset.UniformDBallSampler(2).sample(3000)
plt.scatter(X[0, :], X[1, :])
plt.show()