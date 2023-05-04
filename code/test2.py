import cv2
import numpy as np
from stmd import STMD
import os
import datetime
import cv2

# Open the video file
video_path = "media/video2.mp4"
cap = cv2.VideoCapture(video_path)
print("=============================== PROGRAM START")


# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
channels = 3
lmc = None
limit_frame = 10

if (limit_frame is not None) and (limit_frame < frame_count):
    frame_count = limit_frame
frame_num = 0

print("=============================== OPERATION IN PROGRESS ...")

# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# std
output_path = 'result/stmd_output.avi'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


# Loop through the video frames
while True:
    frame_num += 1
    print("--- frame {} / {} ---".format(frame_num, frame_count))
    frame_file = os.path.join("result", f'frame_{frame_num:04d}.jpg')


    if lmc is None:
        pre_lmc = None
    else:
        pre_lmc = lmc

    # Read the next frame
    ret, frame = cap.read()

    # If the frame is not valid, break out of the loop
    if (not ret) or (frame_num == frame_count) or (cv2.waitKey(1) == ord('q')):
        break

    # cv2.imwrite(frame_normal, frame)

    # Write the frame to the output video
    model = STMD(frame)
    lmc = model.lmc_output
    if (pre_lmc is not None):
        output = model.get_stmd(lmc, pre_lmc)
    else:
        output = None

    if output is not None:
        output = model.convert_for_display(output)
        cv2.putText(output, "Date: " + str( datetime.datetime.now()), (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

        # cv2.imwrite(frame_file, frame_for_display)
        out.write(output)
        # cv2.imshow('Frame', stmd_frame)

        # cv2.imshow('Frame', std_frame)
    else:
        print("output is None")


print("=============================== PROGRAM END")


# Release the video capture object and the video writer object
out.release()
cv2.waitKey(0)
cv2.destroyAllWindows()








#            NOTES
"""

while True:



    ret, frame = cap.read()
    if ret:

        cv2.imwrite(frame_file, pre_frame)

    if cv2.waitKey(1) == ord('q') or frame_num == frame_count:
        break



"""

"""
# video
frame_num_tot = 100000
def get_frames(video):
    print("=============================== VIDEO UPLOADING ...")

    frames_list = []
    frame_num = 0

    while True:

        ret, frame = cap.read()
        if ret:
            frames_list.append(frame)

        if cv2.waitKey(1) == ord('q') or frame_num == frame_num_tot:
            break

        print("--- " + str((frame_num/frame_num_tot)*100) + " % ---")
    cap.release()
    return frames_list



frames_list = get_frames(video)
frame_num_tot = len(frames_list)
for i in range(len(frames_list)):
    frame = frames_list[i]

    # cv2.imshow('Output', estmd.output)
    frame_file = os.path.join("result", f'frame_{i:04d}.jpg')

    if i != 0:
        pre_frame = frames_list[i-1]
        stmd = ESTMD(frame, pre_frame)
        stmd.FDSR()
        output = stmd.testtt

        # Convert the image back to the original range (0-255)
        output = (output * 255).astype(np.uint8)
    else:
        output = frame


    # cv2.imshow("frame 0" + str(frame_num), frame)
    cv2.imwrite(frame_file, output)

    print(str(((i+1)/frame_num_tot)*100) + " %")

# Release the video capture object and close all windows
cv2.destroyAllWindows()

# cv2.waitKey(0)
# cv2.destroyAllWindows()


"""

"""
# Open the video file
video_path = 'media/video1.avi'
cap = cv2.VideoCapture(video_path)

# Get the video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a video writer object
output_path = 'result/output_video2.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through the video frames
n = 0
while True:
    n += 1
    # Read the next frame
    ret, frame = cap.read()

    # If the frame is not valid, break out of the loop
    print(n)
    if not ret or n == 12:
        break


    # Write the frame to the output video
    cv2.putText(frame, "Date: " + str( datetime.datetime.now()), (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

    out.write(frame)

# Release the video capture object and the video writer object
cap.release()
out.release()

cv2.waitKey(0)
cv2.destroyAllWindows()
"""
