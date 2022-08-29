# The two lines below are only needed to execute this code in a Jupyter Notebook (and spyder)
import nest_asyncio
nest_asyncio.apply()

import cv2
import numpy as np

from funcs.aruco import *
from funcs.pupil import discover_device
from funcs.click_game import *

# Needs to be matched with monitor resolution
cols_monitor = 2560-100
rows_monitor = 1440-100

# Resolution of pupil glasses
cols_world = 1088
rows_world = 1080

# Img with aruco markers on corners
background = create_aruco_frame()
background = cv2.resize(background, (cols_monitor,rows_monitor), interpolation = cv2.INTER_AREA)
area_coords = get_area_coords()

# overlay = cv2.imread("./heatmappy/assets/cat.jpg")
overlay = cv2.imread("c:Users/kapyla/Desktop/earth.jpg")
overlay = cv2.resize(overlay, (cols_world,rows_world), interpolation = cv2.INTER_AREA)

def main():

    # Scan for pupil glasses in the same network
    device = discover_device()    
    
    # blob = get_new_blob(area_coords)

    while True:
        image = background
        # image = add_content(image, overlay)
        image = draw_borders(image)
        # image = draw_blob(image, blob)
        
        cv2.imshow("Pupil Invisible - Live Preview", image)   
        k = cv2.waitKey(33)
        if k==27:    # Esc key to stop
            break
            
        # Fetch new matching img-gaze pair from the glasses
        scene_sample, gaze_sample = device.receive_matched_scene_video_frame_and_gaze()
        world_img = scene_sample.bgr_pixels
        gaze = np.array([int(gaze_sample.x), int(gaze_sample.y), 1])
                        
        # Find Aruco corners (Corners in pupil frame) None if not all 4 found
        aruco_corners = find_markers(world_img)
        
        # If Arucos are found
        if not aruco_corners is None:
            projective_matrix = define_projective_matrix(aruco_corners, area_coords)
            warped_world_img = get_warped(world_img, projective_matrix)
            # image = add_overlay(image, warped_world_img, projective_matrix)
            
            # Adjust gaze coordinates to monitor
            gaze_in_monitor = np.matmul( projective_matrix, gaze )
            gaze_in_monitor = gaze_in_monitor / gaze_in_monitor[2]
                    
            if k==32:
                image = draw(image, coords, area_coords)
            # # Show world video with gaze overlay
            # cv2.circle(
            #     image,
            #     (int(round(gaze_in_monitor[0])), int(round(gaze_in_monitor[1]))),
            #     10, (0, 0, 255), 2
            # )
            
            # image = cv2.addWeighted(image, 0.5, warped_world_img, 0.5, 0)
            
        # cv2.imshow("Pupil Invisible - Live Preview", image)   
        # k = cv2.waitKey(33)
        # if k==27:    # Esc key to stop
        #     break
        # elif k==32:
        #     draw = True
        #     # if  is_a_hit(blob, gaze_in_monitor):
        #     #     blob = get_new_blob(area_coords)
        
            
    return warped_world_img

if __name__=="__main__":
   ret_val = main()
   cv2.destroyAllWindows()
   
#%%
cv2.imshow("Pupil Invisible - Live Preview", ret_val)   
k = cv2.waitKey(0)
cv2.destroyAllWindows()
