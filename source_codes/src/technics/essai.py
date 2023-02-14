import math
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def GaussianFunction(x, y, sig1=1):
    a = 1/(2*math.pi*(sig1**2))
    b = -(x**2 + y**2)/(2*(sig1**2))
    g = a*math.exp(b)
    return g


x, y, z = [], [], []
for (i, j) in zip(range(1, 201), range(1, 301)):
    z.append(GaussianFunction(i, j))
    x.append(i)
    y.append(j)

ax = plt.axes(projection="3d")

ax.scatter(x, y, z)

plt.show()

