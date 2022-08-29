# The two lines below are only needed to execute this code in a Jupyter Notebook (and spyder)
import nest_asyncio
nest_asyncio.apply()

import cv2
import numpy as np

from funcs.aruco import *
from funcs.pupil import discover_device
from funcs.click_game import *

from heatmappy.heatmap import Heatmapper
from PIL import Image

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
# overlay = cv2.resize(overlay, (cols_world,rows_world), interpolation = cv2.INTER_AREA)

def main():

    # Scan for pupil glasses in the same network
    device = discover_device()    
    gazes = []
    draw = False
    
    while True:
        # image = background
        image = draw_borders(background)
        image = add_content(image, overlay)
        # image = draw_blob(image, blob)
        
        if draw:
            image = cv2.circle(
                image,
                (int(round(gaze_in_monitor[0])), int(round(gaze_in_monitor[1]))),
                10, (0, 0, 255), 2
            )
            draw = False
        
        cv2.imshow("Pupil Invisible - Live Preview", image)   
        key = cv2.waitKey(33)
            
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
            
            # Keep trach of gaze locations (lower resolution)
            gazes.append( (gaze_in_monitor[0]/2, gaze_in_monitor[1]/2) ) # HOX
            
            if key==32:
                # Show world video with gaze overlay
                draw = True
        
        # Quit and save
        if key == 27:
            frame = cv2.resize(image, (0,0), fx=0.5, fy=0.5) # HOX
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame)

            heatmapper = Heatmapper()
            heatmap = heatmapper.heatmap_on_img(gazes, pil_img)
            heatmap = np.array(heatmap)
            
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

            cv2.imwrite("./outs/heatmap.jpg", heatmap)
            cv2.imshow("window", heatmap)
            key = cv2.waitKey(0)
            break
        
    return gazes

if __name__=="__main__":
   ret_val = main()
   cv2.destroyAllWindows()
   