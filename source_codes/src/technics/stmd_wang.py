import numpy as np
from scipy.integrate import quad
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import cv2

img = cv2.imread("insect.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# print(gray_img)


def gaussian_fct(x, y, sig1=68):
    """ Gaussian function """
    a = 1 / (2 * np.pi * (sig1 ** 2))
    b = -(x ** 2 + y ** 2) / (2 * (sig1 ** 2))
    g = a * np.exp(b)
    return g


class STMD_create:
    def __init__(self, frame):
        """ the is considered as a MxN matrice """
        self.frame = frame
        self.w = np.array([1/16, 1/8, 1/16, 1/8, 1/4, 1/8, 1/16, 1/8, 1/16]).reshape(3, 3)

    def get_m(self):
        """ Return M: rows' number """
        return self.frame.shape[0]

    def get_n(self):
        """ Return N: columns' number """
        return self.frame.shape[1]

    def d_integrate(self):
        m = self.get_m()
        frame_return = []
        for (x, y) in zip(range(self.get_m()), range(self.get_n())):
            def integrand(u, v):
                return self.frame[u, v] * gaussian_fct(x - u, y - v)
                # return self.GaussianFunction(self.get_x()-u, self.get_y() - v)

            def integrate_u(u):
                return quad(integrand, 0, 10, args=(u,))[0]

            result = quad(integrate_u, 0, 10)[0]

            return result

    def photoreceptor(self):
        result = np.zeros(self.frame.shape)
        for i in range(self.get_m()):
            for j in range(self.get_n()):
                l = 0
                for v in [-1, 0, 1]:
                    for u in [-1, 0, 1]:
                        try:
                            l += self.w[u, v] * self.frame[i+u, j+v]
                        except: pass
                result[i, j] = l #uint8
        # result = np.array(result).reshape(self.frame.shape)
        result = result.astype(np.uint8)
        return result

    def lipetz(self):
        result = np.zeros(self.frame.shape)
        for i in range(self.get_m()):
            for j in range(self.get_n()):
                pass



"""

cv2.imshow("real image", img)
cv2.imshow("gray image", gray_img)

rl = STMD_create(gray_img)
result = rl.photoreceptor()


cv2.imshow("Retina layer", result)

cv2.waitKey(0)
"""


# =================================== TEST =================================
x, y, z = [], [], []
for (i, j) in zip(range(-200, 201), range(-300, 301)):
    z.append(gaussian_fct(i, j))
    x.append(i)
    y.append(j)

"""
# Plotting with SCATTER
ax = plt.axes(projection="3d")
ax.scatter(x, y, z)
ax.set_title("Gaussian Function plotting with SCATTER")
"""

# Plotting with PLOT
ax2 = plt.axes(projection="3d")
# ax2.plot(x, y, z)
# ax2.set_title("Gaussian Function plotting with PLOT")


# Surface plotting
X, Y = np.meshgrid(x, y)
Z = gaussian_fct(X, Y)

ax3 = plt.axes(projection="3d")
ax3.plot_surface(X, Y, Z, cmap="plasma")
ax3.set_title("Gaussian Function SURFACE plotting")

plt.show()