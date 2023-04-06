import cv2
from estmd import ESTMD

img = cv2.imread("media/image2.jpg")

img = ESTMD(img)

img.high_pass_filter()

# Show Lipetz transformation result
#cv2.imshow("Lipets transformation", lipetz_img)

cv2.waitKey(0)
