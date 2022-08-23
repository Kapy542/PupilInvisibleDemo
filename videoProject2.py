#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 18:48:38 2021

@author: kapyla
"""

#import time

import cv2
import numpy as np

# https://github.com/pupil-labs/pyndsi/tree/v1.0
import ndsi  # Main requirement

#import imutils
from PIL import Image

SENSOR_TYPES = ["video", "gaze"]
SENSORS = {}  # Will store connected sensors


arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()
rows_world = 1088
cols_world = 1080


def main():
    # Start auto-discovery of Pupil Invisible Companion devices
    network = ndsi.Network(formats={ndsi.DataFormat.V4}, callbacks=(on_network_event,))
    network.start()

    try:
        #
        world_img = np.zeros((rows_world, cols_world, 3))
        gaze = (0, 0)

        # Event loop, runs until interrupted
        while network.running:
            # Check for recently connected/disconnected devices
            if network.has_events:
                network.handle_event()

            # Iterate over all connected devices
            for sensor in SENSORS.values():

                # We only consider gaze and video
                if sensor.type not in SENSOR_TYPES:
                    continue

                # Fetch recent sensor configuration changes,
                # required for pyndsi internals
                while sensor.has_notifications:
                    sensor.handle_notification()

                # Fetch recent gaze data
                for data in sensor.fetch_data():
                    if data is None:
                        continue
                    
                    if sensor.name == "PI world v1":
                        world_img = data.bgr

                    elif sensor.name == "Gaze":
                        # Draw gaze overlay onto world video frame
                        gaze = (int(data[0]), int(data[1]))
                        
            world_img = np.uint8(world_img)
            world_img = find_markers(world_img, arucoDict, arucoParams)
            
            # Show world video with gaze overlay
            cv2.circle(
                world_img,
                gaze,
                40, (0, 0, 255), 4
            )
            cv2.imshow("Pupil Invisible - Live Preview", world_img)
            key = cv2.waitKey(1)
            
            if key == ord("q"):
                network.stop()
                return world_img

    # Catch interruption and disconnect gracefully
    except (KeyboardInterrupt, SystemExit):
        network.stop()


def on_network_event(network, event):
    # Handle gaze sensor attachment
    if event["subject"] == "attach" and event["sensor_type"] in SENSOR_TYPES:
        # Create new sensor, start data streaming,
        # and request current configuration
        sensor = network.sensor(event["sensor_uuid"])
        sensor.set_control_value("streaming", True)
        sensor.refresh_controls()

        # Save sensor s.t. we can fetch data from it in main()
        SENSORS[event["sensor_uuid"]] = sensor
        print(f"Added sensor {sensor}...")

    # Handle gaze sensor detachment
    if event["subject"] == "detach" and event["sensor_uuid"] in SENSORS:
        # Known sensor has disconnected, remove from list
        SENSORS[event["sensor_uuid"]].unlink()
        del SENSORS[event["sensor_uuid"]]
        print(f"Removed sensor {event['sensor_uuid']}...")

def find_markers(frame, arucoDict, arucoParams):

    # detect ArUco markers in the input frame    
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, arucoDict, parameters=arucoParams)
    
    	# verify *at least* one ArUco marker was detected
    if len(corners) > 0:
            # flatten the ArUco IDs list
            ids = ids.flatten()
        
            # If all necessary corners found
            if all(elem in ids  for elem in [1,4,9,12]):          
                # loop over the detected ArUCo corners
                for (markerCorner, markerID) in zip(corners, ids):  
                  
                    if markerID == 1:
                        corners = markerCorner.reshape((4, 2))
                        (topLeft, topRight, bottomRight, bottomLeft) = corners
                        tl = (int(bottomRight[0]), int(bottomRight[1]))
                    elif markerID == 4:
                        corners = markerCorner.reshape((4, 2))
                        (topLeft, topRight, bottomRight, bottomLeft) = corners
                        tr = (int(bottomLeft[0]), int(bottomLeft[1]))
                    elif markerID == 9:
                        corners = markerCorner.reshape((4, 2))
                        (topLeft, topRight, bottomRight, bottomLeft) = corners
                        bl = (int(topRight[0]), int(topRight[1]))
                    elif markerID == 12:
                        corners = markerCorner.reshape((4, 2))
                        (topLeft, topRight, bottomRight, bottomLeft) = corners
                        br = (int(topLeft[0]), int(topLeft[1]))
                
                rows, cols = frame.shape[:2]
                src_points = np.float32([[0,0], [0,rows-1], [cols-1,0], [cols-1,rows-1]])
                dst_points = np.float32([tl, tr, bl, br])
                projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                warped = cv2.warpPerspective(frame, projective_matrix, (cols,rows))
                #frame = cv2.addWeighted(frame,0.4,warped,0.1,0)
                frame = frame+warped
                
    return frame

frame = main()  # Execute example
cv2.destroyAllWindows()
