import os
import cv2
import numpy as np

cap = cv2.VideoCapture('media/video.mp4')
if not cap.isOpened():
    print("Error opening video file")
    exit()

ret, frame1 = cap.read()
if not ret:
    print("Error reading first frame")
    exit()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

frame_num = 0
while(1):
    frame_num += 1
    frame_file = os.path.join("result/optical_flow", f'frame_{frame_num:04d}.jpg')
    ret, frame2 = cap.read()
    if not ret:
        break  # End of video, exit loop

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute magnitude and angle of 2D vectors
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Create mask based on the magnitude with a threshold
    mask = np.zeros_like(mag)
    mask[mag > 1.0] = 255  # Set your own threshold

    cv2.imwrite(frame_file, mask)

    prvs = next

    print(frame_num)

cap.release()
cv2.destroyAllWindows()
