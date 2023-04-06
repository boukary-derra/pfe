import cv2
import estmd

# import an image
img = cv2.imread("media/image2.jpg")

""" ======================= IMAGE PROCESSING ============================ """

# Apply gray scyle
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# use
ph_2016 = stmd.photoreceptor_2016(gray_img)

# lipetz_img = stmd_2016.lipetz_trans(gray_img)

""" ======================= RESULTS ============================ """


# Show the real image
cv2.imshow("Real image", img)

# Show the gray image
cv2.imshow("Gray image", gray_img)

cv2.waitKey(0)
