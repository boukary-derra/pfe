import os
import cv2
import numpy as np

cap = cv2.VideoCapture('media/video.mp4')
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

frame_num = 0
while(1):
    frame_num += 1
    frame_file = os.path.join("result/optical_flow", f'frame_{frame_num:04d}.jpg')
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    draw = cv2.add(frame2, bgr)

    # cv2.imshow('frame2', draw)
    cv2.imwrite(frame_file, draw)
    if cv2.waitKey(1) == ord('q'): break

    prvs = next

    print(frame_num)

cap.release()
cv2.destroyAllWindows()
