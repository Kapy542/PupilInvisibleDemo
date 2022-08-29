#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
import cv2

class Blob:
  def __init__(self, x, y, size):
    self.x = x
    self.y = y
    self.size = size
    
def get_new_blob(area_coords):
        x = random.randint(area_coords[0,0]+50, area_coords[1,0]-50)
        y = random.randint(area_coords[0,1]+50, area_coords[2,1]-50)
        size = random.randint(50, 150)
        
        blob = Blob(x, y, size)
        return blob
    
def draw_blob(img, blob):
    img = cv2.circle(img, (blob.x, blob.y), blob.size, (255,0,0), -1)
    return img

def is_a_hit(blob, coords):
    point1 = np.array((coords[0], coords[1]))
    point2 = np.array((blob.x, blob.y))

    dist = np.linalg.norm(point1 - point2)
    if dist < blob.size:
        return True
    else:
        return False
    
def draw(img, coords, area_coords):
    if area_coords[0,0] < coords[0] < area_coords[0,0] and area_coords[0,1] < coords[1] < area_coords[2,1]:
        img = cv2.circle(img, (coords[0], coords[1]), 5, (255,0,0), -1)
    return img