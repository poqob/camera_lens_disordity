import numpy as np
import cv2 as cv
import glob
import pickle

"""
This script is used to calibrate the camera and undistort the images.
"""


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9, 6)  # @param: chessboard size (count of black and white squares)
frameSize = (1440, 1080)  # @param: frame size (width, height)
size_of_chessboard_squares_mm = (
    20  # @param: size of chessboard squares in mm (real world size)
)
"""
the termination criteria is set to 30 iterations and epsilon = 0.001 (error threshold)
"""
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : chessboardSize[0], 0 : chessboardSize[1]].T.reshape(-1, 2)

objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


images = glob.glob(
    "./chess_board/*.png"
)  # @param: path to chessboard images (png format-supported): more images, better calibration
for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # print(ret, corners)
    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow("img", img)
        cv.waitKey(1000)


cv.destroyAllWindows()


############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, frameSize, None, None
)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
pickle.dump((cameraMatrix, dist), open("./calibration/calibration.pkl", "wb"))
pickle.dump(cameraMatrix, open("./calibration/cameraMatrix.pkl", "wb"))
pickle.dump(dist, open("./calibration/dist.pkl", "wb"))


############## UNDISTORTION #####################################################

img = cv.imread("./images/target.png")
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(
    cameraMatrix, dist, (w, h), 1, (w, h)
)

"""
The classical way to undistort the image is to use cv.undistort() function directly applies 
the parameters to the original image (one process)

In the other hand, initUndistortRectifyMap() creates a pixel map and re-build the image 
using remap() function (many process)

The second method is more accurate and recommended.
"""
# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y : y + h, x : x + w]
cv.imwrite("./images/result_undistort.png", dst)


# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(
    cameraMatrix, dist, None, newCameraMatrix, (w, h), 5
)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y : y + h, x : x + w]
cv.imwrite("./images/result_undistort_remapping.png", dst)


# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist
    )
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: {}".format(mean_error / len(objpoints)))
