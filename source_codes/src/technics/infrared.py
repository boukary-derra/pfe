import cv2


def infrared(video):
    # Initialize capture
    cap = cv2.VideoCapture(video)

    while True:
        # Read current frame
        _, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Threshold the image to isolate pixels with high infrared intensity
        _, threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

        # Dilate the thresholded image to fill in gaps
        dilated = cv2.dilate(threshold, None, iterations=2)

        # Detect contours in the image
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw rectangles around the detected contours
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display the resulting image
        cv2.imshow("Motion Detection", frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
