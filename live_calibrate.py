import numpy as np
import cv2

# Load intrinsic and distortion parameters from .npy files
cameraMatrix = np.load("intrinsic_params.npy")
distCoeffs = np.load("distortion_params.npy")

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Undistort the frame using the stored parameters
    undistorted_frame = cv2.undistort(frame.copy(), cameraMatrix, distCoeffs)

    # Show the undistorted frameq
    cv2.imshow("Undistorted frame", undistorted_frame)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and destroy the windows
cap.release()
cv2.destroyAllWindows()
