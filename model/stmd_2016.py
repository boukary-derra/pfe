import cv2
import numpy as np
import itertools
import sympy as sp
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



#  TOOLS
def low_pass(img, l, tau):
    (m, n) = img.shape
    for (i, j) in itertools.product(range(m), range(n)):
        x = sp.symbols('x')
        f = sp.Function('f')(x)
        diffeq = sp.Eq(f.diff(x)+(1/tau)*f, (1/tau)*l)
        display(diffeq)
        a = sp.dsolve(diffeq, f)
        return a


# =========================== Retine layer======================================
# photoreceptor
def photoreceptor(gray_img):
    """Input: Gray style image (matrix)
    Goal: Apply a blur effect to the input image"""

    w = np.array([1/16, 1/8, 1/16, 1/8, 1/4, 1/8, 1/16, 1/8, 1/16]).reshape(3, 3) # -> Gaussian convolution mask
    (m, n) = gray_img.shape # -> input matrix size (m: with, n: height)
    output = np.zeros((m, n)) # -> output initialization (null matrix with the same size as the input)

    for (i, j) in itertools.product(range(m), range(n)): # -> iterate through all the elements of the input matrix
        l = 0
        for v in [-1, 0, 1]: # -> sum of v from -1 to 1
            for u in [-1, 0, 1]: # -> sum of u from -1 to 1
                try:
                    l += w[u, v] * gray_img[i+u, j+v] # photoreceptor operation: doc [1] eq [1]
                except: pass
        output[i, j] = l # -> fill output matrix element by element

    output  = output .astype(np.uint8) # -> convert output to np.uint8
    return output  # -> return the output

# Lipetz transformation
    "pass"


# =========================== Lamina layer======================================
# low pass filter
    "pass"

# High Pass Filter (LMCs)
    "pass"


# =========================== Medulla layer=====================================
# FDSR (ON)
    "pass"

# FDSR (OFF)
    "pass"


# =========================== Lobula layer======================================
    "pass"




# =========================== Documentation ====================================
"""
[1] H.  Wang, J Peng and S.  Yue, “Bio-inspired Small Target Motion Detector with a new Lateral Inhibition Mechanism,”
Conference Paper - July 2016.
"""
