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
frame = None
limit_frame = 10

if (limit_frame is not None) and (limit_frame < frame_count):
    frame_count = limit_frame
frame_num = 0

print("=============================== OPERATION IN PROGRESS ...")

# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# std
output_path = 'result/std_long.avi'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
# delta
delta_output_path = 'result/delta_long.avi'
delta_out = cv2.VideoWriter(delta_output_path, fourcc, fps, (width, height))
# fdsr
fdsr_output_path = 'result/fdsr_long.avi'
fdsr_out = cv2.VideoWriter(fdsr_output_path, fourcc, fps, (width, height))



# Loop through the video frames
while True:
    frame_num += 1
    print("--- frame {} / {} ---".format(frame_num, frame_count))
    frame_file = os.path.join("result", f'frame_{frame_num:04d}.jpg')
    delta_frame_file = os.path.join("result", f'frame_{frame_num:04d}_delta.jpg')
    fdsr_frame_file = os.path.join("result", f'frame_{frame_num:04d}_fdsr.jpg')
    frame_normal = os.path.join("result", f'frame_{frame_num:04d}_normal.jpg')


    if frame is None:
        pre_frame = None
    else:
        pre_frame = frame

    # Read the next frame
    ret, frame = cap.read()

    # If the frame is not valid, break out of the loop
    if (not ret) or (frame_num == frame_count) or (cv2.waitKey(1) == ord('q')):
        break

    cv2.imwrite(frame_normal, frame)
"""
    # Write the frame to the output video
    stmd = STMD(frame, None)
    frame = stmd.get_std()

    if pre_frame is None:
        delta = None
    else:
        delta = stmd.get_delta(frame, pre_frame)

    if delta is None:
        fdsr = None
    else:
        fdsr = stmd.get_fdsr(frame, delta)


    if frame is not None:
        frame_for_display = stmd.convert_for_display(frame)
        cv2.putText(frame_for_display, "Date: " + str( datetime.datetime.now()), (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

        # cv2.imwrite(frame_file, frame_for_display)
        out.write(frame_for_display)
        # cv2.imshow('Frame', stmd_frame)

    if delta is not None:
        delta_for_display = stmd.convert_for_display(delta)
        cv2.putText(delta_for_display, "Date: " + str( datetime.datetime.now()), (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

        # cv2.imwrite(delta_frame_file, delta_for_display)
        delta_out.write(delta_for_display)
        # cv2.imshow('Frame', std_frame)

    if fdsr is not None:
        fdsr_for_display = stmd.convert_for_display(delta)
        cv2.putText(fdsr_for_display, "Date: " + str( datetime.datetime.now()), (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)

        # cv2.imwrite(fdsr_frame_file, fdsr_for_display)
        fdsr_out.write(fdsr_for_display)
        # cv2.imshow('Frame', std_frame)"""

print("=============================== PROGRAM END")


# Release the video capture object and the video writer object
cap.release()
out.release()
delta_out.release()
fdsr_out.release()
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
