# The two lines below are only needed to execute this code in a Jupyter Notebook (and spyder)
import nest_asyncio
nest_asyncio.apply()

from pupil_labs.realtime_api.simple import discover_one_device

from datetime import datetime
import matplotlib.pyplot as plt
import cv2

device = discover_one_device(max_search_duration_seconds=10.0)

# List all devices that could be found within 10 seconds
print(f"Phone IP address: {device.phone_ip}")
print(f"Phone name: {device.phone_name}")
print(f"Battery level: {device.battery_level_percent}%")
print(f"Free storage: {device.memory_num_free_bytes / 1024**3:.1f} GB")
print(f"Serial number of connected glasses: {device.serial_number_glasses}")

while True:
    scene_sample, gaze_sample = device.receive_matched_scene_video_frame_and_gaze()
    
    # dt_gaze = datetime.fromtimestamp(gaze_sample.timestamp_unix_seconds)
    # dt_scene = datetime.fromtimestamp(scene_sample.timestamp_unix_seconds)
    # print(f"This gaze sample was recorded at {dt_gaze}")
    # print(f"This scene video was recorded at {dt_scene}")
    # print(f"Temporal difference between both is {abs(gaze_sample.timestamp_unix_seconds - scene_sample.timestamp_unix_seconds) * 1000:.1f} ms")
    
    # scene_image_rgb = cv2.cvtColor(scene_sample.bgr_pixels, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(scene_image_rgb)
    # plt.scatter(gaze_sample.x, gaze_sample.y, s=200, facecolors='none', edgecolors='r')
    
    pupil_img = scene_sample.bgr_pixels
    pupil_img = cv2.circle(pupil_img, (int(gaze_sample.x), int(gaze_sample.y)), 20, (0,0,255), 2)
    
    cv2.imshow('gaze', pupil_img)
    # cv2.waitKey(1)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break
    
cv2.destroyAllWindows()