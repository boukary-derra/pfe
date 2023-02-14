import cv2
import numpy as np

# Capture video from camera
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("test1.mp4")

# Get first frame and convert to grayscale
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Set up variables for optical flow
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Loop through video frames
while True:
    # Read next frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow between previous and current frame
    points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, **lk_params)

    # Extract good points
    good_new = points[status == 1]
    good_prev = points[status == 1]

    # Draw optical flow tracks on image
    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        """x_new, y_new = new.ravel()
        x_prev, y_prev = prev.ravel()"""
        """x_new, y_new = new[0]
        x_prev, y_prev = prev[0]"""
        """x_new, y_new = np.int32(new[0])
        x_prev, y_prev = np.int32(prev[0])"""
        x_new, y_new = int(new[0]), int(new[1])
        x_prev, y_prev = int(prev[0]), int(prev[1])
        cv2.line(frame, (x_new, y_new), (x_prev, y_prev), (0, 255, 0), 2)

    # Show image with optical flow tracks
    cv2.imshow('Optical Flow', frame)

    # Update previous frame and grayscale image
    prev_gray = gray
    prev_frame = frame

    # Break loop if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
