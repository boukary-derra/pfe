import os
import cv2

# Open the video file
video_path = "../media/complexe_frames/video.mp4"
cap = cv2.VideoCapture(video_path)
print(cap)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_num = 0

print("=============================== PROGRAM START")
# Loop through the video frames
while True:
    frame_num += 1
    frame_file = os.path.join("../media/complexe_frames", f'frame_{frame_num:04d}.jpg')

    # Read the next frame
    ret, frame = cap.read()

    # If the frame is not valid, break out of the loop
    if (not ret) or (cv2.waitKey(1) == ord('q')):
        break

    cv2.imwrite(frame_file, frame)
    print("--- frame {} / {} ---".format(frame_num, total_frames))


print("=============================== PROGRAM END")

# Release the video capture object and the video writer object
cv2.waitKey(0)
cv2.destroyAllWindows()
