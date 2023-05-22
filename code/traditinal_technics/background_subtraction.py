import cv2
import numpy as np
import os

# video
video = "../media/simple_frames/video.mp4"

cap = cv2.VideoCapture(video)

kernel_dil = np.ones((20, 20), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.createBackgroundSubtractorMOG2() # background subtraction function
# gg = cv2.calcOpticalFlowFarneback()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_path = 'result/background_subtraction.avi'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_num = 0

# step = 0
while True:
    frame_num += 1
    frame_file = os.path.join("result/background_subtraction", f'frame_{frame_num:04d}.jpg')

    print("--- frame {} / {} ---".format(frame_num, frame_count))

    ret, frame = cap.read()
    # print frame.shape
    if ret == True:
        fshape = frame.shape
        frame = frame[100:fshape[0] - 100, :fshape[1] - 100, :]  # cropping the video size
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        dilation = cv2.dilate(fgmask, kernel_dil, iterations=1)
        (contours, hierarchy) = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if(area > 3500): # if the blob size is greater than 3500 then only detect
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                roi_vehchile = frame[y:y-10+h+5, x:x-8+w+10]

                # cv2.imshow("original", frame)
                # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(frame_file, frame)
                # out.write(frame)
            else:
                break
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
# out.release()
cap.release()
cv2.destroyAllWindows()
