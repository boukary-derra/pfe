import cv2

# video
video = "../media/video1.avi"

# Initialize capture
cap = cv2.VideoCapture(video)

# Read the first frame as the background
_, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

while True:
    # Read current frame
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate absolute difference between current frame and background
    diff = cv2.absdiff(first_gray, gray)

    # Threshold the difference image
    threshold = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image
    dilated = cv2.dilate(threshold, None, iterations=2)

    # Display the resulting image with moving objects
    cv2.imshow("Motion Detection", dilated)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

"""
In this code, the cv2.VideoCapture(0) function is used to capture the video from the default camera.
The first frame is captured and converted to grayscale, which will be used as the background.
In the while loop, the current frame is captured, converted to grayscale, and subtracted from the background
to calculate the difference. The difference is thresholded, dilated, and displayed in a window.
The loop continues until the 'q' key is pressed, at which point the capture is released and the windows are closed.
"""
