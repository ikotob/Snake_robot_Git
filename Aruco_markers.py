import cv2
from cv2 import cv2.aruco
import numpy as np

tag_size = 200
id = 23

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
img = np.zeros((tag_size, tag_size), dtype=np.uint8)
img = cv2.aruco.drawMarker(aruco_dict, id, tag_size, img, 1)

cv2.imwrite("marker_{}.png".format(id), img)
