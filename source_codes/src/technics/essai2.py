import numpy as np
import itertools
import cv2
from scipy.integrate import quad


def gaussian_fct(x, y, sig1=68):
    """ Gaussian function """
    a = 1 / (2 * np.pi * (sig1 ** 2))
    b = -(x ** 2 + y ** 2) / (2 * (sig1 ** 2))
    g = a * np.exp(b)
    return g


frame = cv2.imread("insect.jpg")


def d_integrate(frame):
    (m, n) = frame.shape
    result = np.zeros((m, n))
    for (x, y) in itertools.product(range(m), range(n)):

        def integrand(u, v):
            return frame[int(u), int(v)] * gaussian_fct(x - u, y - v)

        def integrate_u(u):
            return quad(integrand, -1, 1, args=(u,))[0]

        tp = quad(integrate_u, -1, 1)[0]

        result[x, y] = tp

    return result


img = cv2.imread("insect.jpg")
frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


result = d_integrate(frame)
cv2.imshow("real image", frame)
cv2.imshow("Result image", result)

print(result)

cv2.waitKey(0)

