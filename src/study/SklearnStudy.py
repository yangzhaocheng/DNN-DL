import numpy as np
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
import sklearn.linear_model

X, y = sklearn.datasets.make_moons(200, noise=0.2)
print(X)
print(len(X))
plt.scatter(X[:, 0], X[:, 1], s=100, c=y, cmap=plt.cm.Spectral)
plt.show()

