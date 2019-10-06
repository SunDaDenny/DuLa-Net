import math

camera_h = 1.6
pano_size = (512, 1024)
fp_size = 512
fp_fov = 160
fp_meter = camera_h / math.tan(math.pi *  (180 - fp_fov) / 360) * 2