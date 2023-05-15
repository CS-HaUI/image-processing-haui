from process.function import LBP
from visuallization.subplots import Plot
import cv2
import numpy as np

object = LBP(from_path= True, load_path="data/json/data.json")


lbp = object.get_image_lbp_from_id(ids = [11, 12])
color   = object.get_image_color_from_id(ids = [11, 12])

Plot(2, 2,
     images= [[color[0], lbp[0]],
              [color[1], lbp[1]]],
     figsize=(15, 15),
     show_type=[['RGB', 'GRAY'],
                ['RGB', 'GRAY']],
     show_axis=[['off', 'off'],
                ['off', 'off']],
     show_title=[['RGB Thật', 'LBP thật'],
                 ['RGB Giả', 'LBP Giả']])._save("visuallization/demo/result.png")