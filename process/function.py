import cv2
from glob import glob
import numpy as np
from scipy.spatial.distance import cdist
from skimage.feature import local_binary_pattern
import json
from tqdm import tqdm
import shutil, threading, time
from threading import Thread
import matplotlib.pyplot as plt


class BaseImage(object):
    def __init__(self, path,
                       types= "*",
                       size= (128, 128), 
                       live= False):
        self.__path = path
        self.__types = types
        self.__size = size
        
        # BGR
        self.__image = path if live else self.__read()
        
        
    def __read(self):
        list_image = []
        for type in tqdm(self.__types, desc= "types"):
            root = glob(self.__path + type)
            for file in tqdm(root, desc= f"file: {type}"):
                tmp = cv2.resize(cv2.imread(file), self.__size)
                list_image.append(tmp)
        return np.asarray(list_image)
    
    
    def get_all_image(self):
        return self.__image

        
class LBP(BaseImage):
    def __init__(self, path, 
                       types= "*",
                       size = (128, 128),
                       P = 8,
                       R = 1,
                       method= 'default'):
        super().__init__(path, types, size, False)
        self.P = P
        self.R = R
        self.method = method
        self.__vector_feature = self.__feature()
        
    
    def __hist_lbp(self, image, _title=""):
        lbp = local_binary_pattern(image, self.P, self.R, method=self.method)
        lbp_hist, _ = np.histogram(lbp, bins= int(np.max(lbp) + 1), range=(0, int(np.max(lbp) + 1)))
        return np.asarray(lbp_hist)
    
    
    def __feature(self):
        bgrs = self.get_all_image().copy()
        features = []
        for index, bgr in tqdm(enumerate(bgrs), desc= "extract feature"):
            temp = bgr.copy()
            r, c, channels = temp.shape
            temp_hsv = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
            temp_gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            temp_Ycbcr = cv2.cvtColor(temp, cv2.COLOR_BGR2YCrCb)
            feature = []
            
            for channel in range(channels):
                feature.append(self.__hist_lbp(temp_hsv[:, :, channel]))
                feature.append(self.__hist_lbp(temp_Ycbcr[:, :, channel]))
            feature_gray = []
            feature.append(self.__hist_lbp(temp_gray, "gray"))
            
            feature = np.asarray(feature).reshape(-1)
            feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
            features.append( feature )
            
        return np.asarray(features)
    
    
    def get_feature(self):
        return self.__vector_feature
        
    



class Distance:
    """
        Hàm tính khoảng cách sẽ được dùng từ [version 2.0] trở đi
    """
    def __init__(self):
        self.one  = None
        self.two  = None
        self.two  = None
        self.k    = None
        
    
    def __distance_fromA2B(self):
        # (N, M) -- (1, M) -> (N, 1)
        distance = cdist(self.two, self.one)
        distance = distance.reshape(-1, 1)
        # range from 0 to N -> axis = 1, vertical
        indexs   = np.arange(len(distance)).reshape(-1, 1)
        distance = np.concatenate((distance, indexs), axis= 1)
        distance = distance[np.argsort(distance[:, 0])]
        return distance[:, 1].astype(int)
    
    
    def get_data(self):
        distance = self.__distance_fromA2B()
        
        return distance[:self.k]
    
    def reset(self, object_one, object_two, k):
        self.one = object_one
        self.two = object_two
        self.k   = k
    
    
# ----------------------------- CAMERA -------------------------------- #
class Camera(object):
    """
    Base Camera object
    """

    def __init__(self):
        self._cam = None
        self._frame = None
        self._frame_width = None
        self._frame_height = None
        self._ret = False

        self.auto_undistortion = False
        self._camera_matrix = None
        self._distortion_coefficients = None

        self._is_running = False

    def _init_camera(self):
        """
        This is the first for creating our camera
        We should override this!
        """

        pass

    def start_camera(self):
        """
        Start the running of the camera, without this we can't capture frames
        Camera runs on a separate thread so we can reach a higher FPS
        """

        self._init_camera()
        self._is_running = True
        threading.Thread(target=self._update_camera, args=()).start()

    def _read_from_camera(self):
        """
        This method is responsible for grabbing frames from the camera
        We should override this!
        """

        if self._cam is None:
            raise Exception("Camera is not started!")

    def _update_camera(self):
        """
        Grabs the frames from the camera
        """

        while True:
            if self._is_running:
                self._ret, self._frame = self._read_from_camera()
            else:
                break

    def get_frame_width_and_height(self):
        """
        Returns the width and height of the grabbed images
        :return (int int): width and height
        """

        return self._frame_width, self._frame_height

    def read(self):
        """
        With this you can grab the last frame from the camera
        :return (boolean, np.array): return value and frame
        """
        return self._ret, self._frame

    def release_camera(self):
        """
        Stop the camera
        """
        threading.Thread(target=self._update_camera, args=())._stop()
        self._is_running = False

    def is_running(self):
        return self._is_running

    def set_calibration_matrices(self, camera_matrix, distortion_coefficients):
        self._camera_matrix = camera_matrix
        self._distortion_coefficients = distortion_coefficients

    def activate_auto_undistortion(self):
        self.auto_undistortion = True

    def deactivate_auto_undistortion(self):
        self.auto_undistortion = False

    def _undistort_image(self, image):
        if self._camera_matrix is None or self._distortion_coefficients is None:
            import warnings
            warnings.warn("Undistortion has no effect because <camera_matrix>/<distortion_coefficients> is None!")
            return image

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self._camera_matrix,
                                                               self._distortion_coefficients, (w, h),
                                                               1,
                                                               (w, h))
        undistorted = cv2.undistort(image, self._camera_matrix, self._distortion_coefficients, None,
                                    new_camera_matrix)
        return undistorted


class WebCamera(Camera):
    """
    Simple Webcamera
    """

    def __init__(self, video_src=0):
        """
        :param video_src (int): camera source code (it should be 0 or 1, or the filename)
        """

        super().__init__()
        self._video_src = video_src

    def _init_camera(self):
        super()._init_camera()
        self._cam = cv2.VideoCapture(self._video_src)
        self._ret, self._frame = self._cam.read()
        if not self._ret:
            raise Exception("No camera feed")
        self._frame_height, self._frame_width, c = self._frame.shape
        return self._ret

    def _read_from_camera(self):
        super()._read_from_camera()
        self._ret, self._frame = self._cam.read()
        if self._ret:
            if self.auto_undistortion:
                self._frame = self._undistort_image(self._frame)
            return True, self._frame
        else:
            return False, None

    def release_camera(self):
        super().release_camera()
        self._cam.release()

# ----------------------------- CAMERA -------------------------------- #

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


"""
cam = WebCamera(video_src= 0)
cam.start_camera()
cam.release_camera()        
"""