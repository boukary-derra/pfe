import cv2
import stmd_2016
import sympy as sp
from IPython.display import display


img = cv2.imread("media/image_test.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ph_img = stmd_2016.photoreceptor(gray_img)

cv2.imshow("real image", img)
# cv2.imshow("gray image", gray_img)
cv2.imshow("photoreptors layer", ph_img)

cv2.waitKey(0)

"""
def low_pass():
    x = sp.symbols('x')
    f = sp.Function('f')(x)
    diffeq = sp.Eq(f.diff(x)-2*f, 3)
    display(diffeq)
    a = sp.dsolve(diffeq, f)
    return a


r = low_pass()
print(r)
"""
