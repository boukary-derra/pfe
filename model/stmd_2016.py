import cv2
import numpy as np
import itertools
import sympy as sp
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



#  tools
def low_pass(img, l, tau):
    (m, n) = img.shape
    for (i, j) in itertools.product(range(m), range(n)):
        x = sp.symbols('x')
        f = sp.Function('f')(x)
        diffeq = sp.Eq(f.diff(x)+(1/tau)*f, (1/tau)*l)
        display(diffeq)
        a = sp.dsolve(diffeq, f)
        return a


# =========Retine layer=======================================================
def photoreceptor(gray_img):
    w = np.array([1/16, 1/8, 1/16, 1/8, 1/4, 1/8, 1/16, 1/8, 1/16]).reshape(3, 3)
    result = np.zeros(gray_img.shape)
    (m, n) = gray_img.shape
    for (i, j) in itertools.product(range(m), range(n)):
        l = 0
        for v in [-1, 0, 1]:
            for u in [-1, 0, 1]:
                try:
                    l += w[u, v] * gray_img[i+u, j+v]
                except: pass
        result[i, j] = l #uint8
    # result = np.array(result).reshape(self.frame.shape)
    result = result.astype(np.uint8)
    return result

eq1 = low_pass(tau = tau1)


 #=====Lamina layer===============================================================
# low pass filter
eq2 = low_pass(tau = tau2)
# High Pass Filter (LMCs)
eq3 = low_pass(tau=tau3)

# =========Medulla layer=======================================================
