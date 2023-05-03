import cv2
import numpy as np
import itertools
import sympy as sp
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from IPython.display import display

#  TOOLS
def equadif_resoluion(l, t_val, tau=3, C1_val=1):
    t = sp.symbols('t')
    C1 = sp.symbols('C1')
    lc = sp.Function('lc')(t)
    diffeq = sp.Eq(lc.diff(t)+(1/tau)*lc, (1/tau)*l)
    display(diffeq)
    solution = sp.dsolve(diffeq, lc)
    solution = solution.rhs
    solution = solution.subs({t: t_val, C1: C1_val})
    return solution

"""def low_pass(l, t_val, tau=3, C1_val=1):
    return np.exp(-0.33*t_val) + l"""

"""re = low_pass(l=1, t_val=1)
print(re)
re = low_pass(l=7, t_val=1)
print(re)"""


# =========================== Retine layer======================================
# photoreceptor
def photoreceptor_2016(gray_img):
    """
        Input: Gray style image (matrix)
        Goal: Apply a blur effect to the input image
    """

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
def lipetz_trans(img):
    u = 0.7
    (m, n) = img.shape
    output = np.zeros((m, n))
    for (i, j) in itertools.product(range(m), range(n)):
        l = img[i, j]
        lc = equadif_resoluion(l=l, t_val=1)
        p = l**u/(l**u + lc**u)
        output[i, j] = l
    output  = output .astype(np.uint8)
    return output


# =========================== Lamina layer======================================
# Low Pass Filter
def low_pass_filter(img):
    (m, n) = img.shape
    output = np.zeros((m, n))
    for (i, j) in itertools.product(range(m), range(n)):
        p = img[i, j]
        x = equadif_resoluion(l=p, t_val=1)
        output[i, j] = l
    output  = output .astype(np.uint8)
    return output

# High Pass Filter (LMCs)
def high_pass_filter(img):
    (m, n) = img.shape
    output = np.zeros((m, n))
    for (i, j) in itertools.product(range(m), range(n)):
        x = img[i, j]
        x_lmc = equadif_resoluion(l=x, t_val=1)
        y_lmc = x - x_lmc
        output[i, j] = y_lmc
    output  = output .astype(np.uint8)
    return output


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
