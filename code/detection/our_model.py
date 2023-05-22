
import sys
from pathlib import Path
import os
import datetime
import numpy as np
import cv2
# add the parent directory of "project" to the sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from stmd import STMD

# Open the video file
video_path = "media/video.mp4"
cap = cv2.VideoCapture(video_path)

frame = None
frame_num = 0
# Loop through the video frames
while True:
    frame_num += 1
    frame_file = os.path.join("result/our_model", f'frame_{frame_num:04d}.jpg')


    if frame is None:
        last_frame = None
    else:
        last_frame = frame

    # Read the next frame
    ret, frame = cap.read()

    # If the frame is not valid, break out of the loop
    if (not ret) or (cv2.waitKey(1) == ord('q')):
        break

    # cv2.imwrite(frame_normal, frame)

    # Write the frame to the output video
    if (last_frame is not None) and (frame is not None):
        model = STMD(frame, last_frame)
        output = model.get_final_output()

        """# Trouver les contours
        contours, _ = cv2.findContours(output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Dessiner les contours sur l'image originale
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) > 500:  # Ignorer les petits contours
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)"""


        cv2.imwrite(frame_file, output)
         #out.write(output)
        # cv2.imshow('Frame', stmd_frame)

        # cv2.imshow('Frame', std_frame)
    else:
        print("output is None")


    print(" ==========> " + str(frame_num) + " <==========")


# Release the video capture object and the video writer object
cv2.waitKey(0)
cv2.destroyAllWindows()
