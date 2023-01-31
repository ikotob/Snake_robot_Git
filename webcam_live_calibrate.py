import numpy as np
import cv2 as cv


chessboardSize = (8, 6)
frameSize = (1920, 1080)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays to store object points and image points from all the images
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(frame, chessboardSize, corners2, ret)
        cv.imshow('frame', frame)
        key = cv.waitKey(1)
        if key == 27:
            break

    if len(objpoints) >= 10:
        break

cap.release()
cv.destroyAllWindows()

# CALIBRATION
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# UNDISTORTION
img = cv.imread('fixed_pic1.jpg')
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv.imwrite('caliResult1.jpg', dst)

# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]

new_cam_matrix = cameraMatrix
new_cam_matrix = np.array(new_cam_matrix)
np.save("new_cam_matrix.npy",new_cam_matrix)

#display the results
cv.imshow("Original Image", img)
cv.imshow("Undistorted Image", dst)
cv.waitKey(0)
cv.destroyAllWindows()
if dst.shape[0] > 0 and dst.shape[1] > 0:
    cv.imshow("Undistorted Image", dst)
    cv.waitKey(0)
else:
    print("Error: Invalid image size")

cv.destroyAllWindows()

