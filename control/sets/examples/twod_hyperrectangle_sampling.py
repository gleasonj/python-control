import control.sets as cset

import numpy as np

import matplotlib.pyplot as plt

X = cset.Hyperrectangle(-5*np.ones(2), 5*np.ones(2)).sample(3000)
plt.scatter(X[0, :], X[1, :])
plt.show()