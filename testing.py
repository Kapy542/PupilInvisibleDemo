import cv2
import numpy as np
import scipy.io

img = cv2.imread("./data/calibration_imgs/ezgif-frame-001.jpg")

data = scipy.io.loadmat('./data/intrinsicMatrix.mat')
K = data['K'].T

data = scipy.io.loadmat('./data/radialDistortion.mat')
radialDistortion = data['radialDistortion']

distCoeffs = np.zeros(4)
distCoeffs[:2] = radialDistortion

undistorted_img = cv2.undistort(img, K, distCoeffs) 

cv2.imshow("window1", img)
cv2.imshow("window2", undistorted_img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()