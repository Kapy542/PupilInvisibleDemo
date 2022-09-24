#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pupil_labs.realtime_api.simple import discover_one_device
import cv2

# undistord gaze and image
def undistort(frame, gaze, K, distCoeffs):
    frame_undistorted = cv2.undistort(frame, K, distCoeffs) 
    gaze_undistorted = 	cv2.undistortImagePoints(gaze, K, distCoeffs)
    return frame_undistorted, gaze_undistorted
    

# Scan for pupil glasses in the same network
def discover_device():  
    print("Discovering device...")
    device = discover_one_device(max_search_duration_seconds=10.0)
    
    if device is None:
        print("No device found.")
        raise SystemExit(-1)
    
    # List all devices that could be found within 10 seconds
    print(f"Phone IP address: {device.phone_ip}")
    print(f"Phone name: {device.phone_name}")
    print(f"Battery level: {device.battery_level_percent}%")
    print(f"Free storage: {device.memory_num_free_bytes / 1024**3:.1f} GB")
    print(f"Serial number of connected glasses: {device.serial_number_glasses}")
    
    return device