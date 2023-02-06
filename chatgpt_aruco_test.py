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
    corners1, ids1, _ = cv2.aruco.detectMarkers(undistorted, aruco_dict)
    corners2, ids2, _ = cv2.aruco.detectMarkers(undistorted, aruco_dict)

    # Estimate the pose of the marker
    if ids1 is not None:
        rvec1, tvec1, _ = cv2.aruco.estimatePoseSingleMarkers(corners1, markerLength, mtx, dist)
        tvec1 = tvec1[0].reshape(3, 1)
        rvec1 = rvec1[0].reshape(3, 1)

        cv2.aruco.drawAxis(undistorted, mtx, dist, rvec1, tvec1, markerLength)

        # Compute the distance to the marker
        distance_horizontal_1 = np.linalg.norm(tvec1[0][0])
        distance_vertical_1 = np.abs(tvec1[1][0])
        distance_xyz_1 = np.linalg.norm(tvec1[:, 0])


        rotate_1 = np.abs(rvec1[1][0])

        cv2.putText(undistorted, "Horizontal Distance: {:.2f} m".format(distance_horizontal_1), (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
        cv2.putText(undistorted, "Vertical Distance: {:.2f} m".format(distance_vertical_1), (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
        cv2.putText(undistorted, "Distance xyz: {:.2f} m".format(distance_xyz_1), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(undistorted, "rotate: {:.2f} deg".format(rotate_1 / np.pi * 180), (50, 250), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)

    if ids2 is not None:
        rvec2, tvec2, _ = cv2.aruco.estimatePoseSingleMarkers(corners2, markerLength, mtx, dist)
        tvec2 = tvec2[0].reshape(3, 1)
        rvec2 = rvec2[0].reshape(3, 1)

        cv2.aruco.drawAxis(undistorted, mtx, dist, rvec1, tvec1, markerLength)
        cv2.aruco.drawAxis(undistorted, mtx, dist, rvec2, tvec2, markerLength)


        distance_horizontal_2 = np.linalg.norm(tvec2[0][0])
        distance_vertical_2 = np.abs(tvec2[1][0])
        distance_xyz_2 = np.linalg.norm(tvec2[:, 0])

        rotate_2 = np.abs(rvec2[1][0])

        cv2.putText(undistorted, "Horizontal Distance 2: {:.2f} m".format(distance_horizontal_2), (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
        cv2.putText(undistorted, "Vertical Distance 2: {:.2f} m".format(distance_vertical_2), (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2)
        cv2.putText(undistorted, "Distance xyz 2: {:.2f} m".format(distance_xyz_2), (50, 800), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(undistorted, "rotate 2: {:.2f} deg".format(rotate_2 / np.pi * 180), (50, 1000), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0, 0), 2)

    # Display the image
    cv2.imshow("Undistorted", undistorted)

    # Exit if the "q" key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
