import cv2
import numpy as np

# Load calibration data
mtx = np.load('intrinsic_params.npy')
dist = np.load('distortion_params.npy')
# Set the size of the Aruco marker
markerLength = 0.05  # 5 cm

# Create a dictionary of markers
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)

# Start capturing webcam footage
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort the frame
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Detect the markers
    corners, ids, _ = cv2.aruco.detectMarkers(undistorted, aruco_dict)

    # Estimate the pose of the marker
    if ids is not None:
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)
        tvec = tvec[0].reshape(3, 1)
        rvec = rvec[0].reshape(3, 1)

        cv2.aruco.drawAxis(undistorted, mtx, dist, rvec, tvec, markerLength)

        # Compute the distance to the marker
        distance = np.linalg.norm(tvec[0][0])
        cv2.putText(undistorted, "Distance: {:.2f} m".format(distance), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2)

    # Display the image
    cv2.imshow("Undistorted", undistorted)

    # Exit if the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
