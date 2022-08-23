#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 15:46:20 2021

@author: kapyla
"""

import cv2
import numpy as np

img = cv2.imread('test_data/markers.png')
rows, cols = img.shape[:2]

#src_points = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
#dst_points = np.float32([[0,0], [cols-1,0], [int(0.33*cols),rows-1], [int(0.66*cols),rows-1]]) 
src_points = np.float32([[0,0], [0,rows-1], [cols-1,0], [cols-1,rows-1]])
dst_points = np.float32([[0,10], [50,100], [200,100], [200,150]])
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
img_output = cv2.warpPerspective(img, projective_matrix, (cols,rows))

cv2.imshow('Input', img)
cv2.imshow('Output', img_output)
cv2.waitKey()

cv2.destroyAllWindows()