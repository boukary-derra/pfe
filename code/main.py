import cv2
import numpy as np
import stmd_2016


cap = cv2.VideoCapture("media/video1.avi")

# Iterate until the user presses the ESC key
while True:

    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.resize(gray_frame, (80, 80))

    ph_img = stmd_2016.photoreceptor(gray_frame)

    # cv2.imshow("Video", frame)
    # cv2.imshow("Gray style", gray_frame)

    ph_img = cv2.resize(ph_img, (400, 400))

    cv2.imshow("photoreceptor", ph_img)




    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
