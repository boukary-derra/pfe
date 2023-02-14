import cv2

# Load the video and set the parameters for the STMD
video = cv2.VideoCapture("../mediafiles/est1.mp4")
stmd_params = cv2.SimpleBlobDetector_Params()
stmd_params.filterByArea = True
stmd_params.minArea = 100

# Create the STMD detector
stmd = cv2.STMD_create(stmd_params)

# Initialize the background model
bg_model = cv2.createBackgroundSubtractorMOG2()

# Process the video frame by frame
while video.isOpened():
    # Read the frame
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Subtract the background from the frame
    fg_mask = bg_model.apply(gray_frame)

    # Detect the small moving objects using STMD
    keypoints = stmd.detect(fg_mask)

    # Draw the small moving objects on the frame
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 0, 255),
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show the frame with the small moving objects
    cv2.imshow("Frame", frame_with_keypoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video and destroy the windows
video.release()
cv2.destroyAllWindows()