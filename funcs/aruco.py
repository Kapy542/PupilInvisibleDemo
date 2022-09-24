#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import imutils

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()

aruco_width = 200
pad = 20
resolution = (2560-100,1440-100)
# resolution = (1088,1080)
# resolution = (1920,1080)

def find_markers(frame):   
    # detect all ArUco markers in the input frame    
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
    
    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
        
            # If all necessary corners found 
            if all(elem in ids  for elem in [1,2,3,4]): 
                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):  
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                      
                    # Corners (x, y)
                    if markerID == 1:                      
                        tl = (int(bottomRight[0]), int(bottomRight[1]))
                    elif markerID == 2:
                        tr = (int(bottomLeft[0]), int(bottomLeft[1]))
                    elif markerID == 3:
                        bl = (int(topRight[0]), int(topRight[1]))
                    elif markerID == 4:
                        br = (int(topLeft[0]), int(topLeft[1]))
                    
                    # Debugging
                    # cv2.putText(frame, str(markerID),
                    #             (int(topLeft[0]), int(topLeft[1] )),
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             0.5, (0, 255, 0), 2)
                    
                area_corners = np.float32([tl, tr, bl, br])               
                return area_corners

def define_projective_matrix(src_points, dst_points):
    projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)   
    return projective_matrix

# Warp img from pupil to monitor. Ouput resolution is same as monitor and area outside monitor is not included
def get_warped(img, projective_matrix):
    warped = cv2.warpPerspective(img, projective_matrix, resolution)
    return warped

# TODO Check if it works (first need to resize overlay to match monitor resolution (resolution used in defining projective matrix))
# Morphs image to the area defined by arucos (background = pupil img)
# Add overlay img on the content area (This is world frame)
def add_overlay(background, overlay, projective_matrix):
    rows, cols = overlay.shape[:2]
    warped = cv2.warpPerspective(overlay, np.linalg.inv(projective_matrix), (cols,rows))
    both = cv2.addWeighted(background,0.4,warped,0.7,0)
    # both = background+warped
    return both

# Create frame with 4 aruco markers on the corners
def create_aruco_frame():
    aruco_img = np.ones((resolution[1], resolution[0], 3))
    aruco_img = np.uint8(aruco_img) * 255
    aruco_img = cv2.rectangle(aruco_img, (0,0), (resolution[0],resolution[1]), (255,0,0), 2)

    img1 = cv2.aruco.drawMarker(arucoDict,1, aruco_width)
    img2 = cv2.aruco.drawMarker(arucoDict,2, aruco_width)
    img3 = cv2.aruco.drawMarker(arucoDict,3, aruco_width)
    img4 = cv2.aruco.drawMarker(arucoDict,4, aruco_width)
    
    aruco_img[pad:aruco_width+pad, pad:aruco_width+pad] = cv2.merge([img1,img1,img1])
    aruco_img[pad:aruco_width+pad, resolution[0]-aruco_width-pad:resolution[0]-pad] = cv2.merge([img2,img2,img2])
    aruco_img[resolution[1]-aruco_width-pad:resolution[1]-pad, pad:aruco_width+pad] = cv2.merge([img3,img3,img3])
    aruco_img[resolution[1]-aruco_width-pad:resolution[1]-pad, resolution[0]-aruco_width-pad:resolution[0]-pad] = cv2.merge([img4,img4,img4])
    
    return aruco_img

# plots line between inner aruco corners
def draw_borders(img):
    tl = (aruco_width + pad,                  aruco_width + pad)
    tr = (resolution[0] - aruco_width - pad,  aruco_width + pad)
    bl = (aruco_width + pad,                  resolution[1] - aruco_width - pad)   
    br = (resolution[0] - aruco_width - pad,  resolution[1] - aruco_width - pad)
    img = cv2.line(img, tl, tr, (0,0,255), 2)
    img = cv2.line(img, bl, br, (0,0,255), 2)
    img = cv2.line(img, tl, bl, (0,0,255), 2)
    img = cv2.line(img, tr, br, (0,0,255), 2)
    return img

# Coords of inner aruco corners
def get_area_coords():
    tl = (aruco_width + pad,                  aruco_width + pad)
    tr = (resolution[0] - aruco_width - pad,  aruco_width + pad)
    bl = (aruco_width + pad,                  resolution[1] - aruco_width - pad)   
    br = (resolution[0] - aruco_width - pad,  resolution[1] - aruco_width - pad)
    area_corners = np.float32([tl, tr, bl, br])   
    return area_corners

# Adds image to the center of the aruco image   
def add_content(aruco_img, img):
    # Resize image to max size while keeping aspect ratio
    max_width = resolution[0] - 2*aruco_width - 4*pad
    max_height = resolution[1] - 2*aruco_width - 4*pad
    img2 = imutils.resize(img, width=max_width)
    if img2.shape[0] > max_height:
        img2 = imutils.resize(img, height=max_height)
        
    height, width = img2.shape[:2]    
    corner_y = round(resolution[1]/2 - height/2)
    corner_x = round(resolution[0]/2 - width/2)
    
    aruco_img[corner_y:corner_y+height, corner_x:corner_x+width] = img2
    return aruco_img

    
#%%
# aruco_img = create_aruco_frame()
# a = create_aruco_frame()

# cv2.imshow("Pupil Invisible - Live Preview", a)   
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
# cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
# cv2.imshow("window", a)
# key = cv2.waitKey(0)
# cv2.destroyAllWindows()
"""
#%%
img = cv2.imread("/home/kapyla/Documents/PupilDemo/imgs/civit_hero_banner_0_0.jpg")
#img = cv2.imread("/home/kapyla/Documents/PupilDemo/imgs/joker.jpg")

aruco_img = create_aruco_frame()

img2 = add_content(img, aruco_img)

cv2.imshow("window", img2)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
cv2.imshow("window", aruco_img)
key = cv2.waitKey(0)
cv2.destroyAllWindows()
"""