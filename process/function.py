import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import shutil
from scipy.spatial.distance import cdist
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC


class BaseImage(object):
    def __init__(self, path = "", types= "*", size=(128, 128), lim= None, move=True):
        self.__path = path
        self.__types = types
        self.__size = size
        self.__lim = lim
        self.__move = move
        if len(self.__path):
            self.__images, self.__paths = self.__load()
        else:
            self.__images = None
            self.__paths = None
        

    def __load(self):
        __images = []
        __paths = []
        index = 0
        for _type in tqdm(self.__types, "TYPE OF IMAGE"):
            image_path = glob(self.__path + _type)
            for file in tqdm(image_path, "READ FILE OPENCV"):
                if index < self.__lim:
                    __images.append( 
                        cv2.resize(cv2.imread(file), self.__size) 
                    )
                    __paths.append(file)
                    index += 1
                    path_file = file.split('\\')
                    path_file[-2] += "_moved"
                    if self.__move:
                        shutil.move(file, '\\'.join(path_file))
        return np.asarray(__images), np.asarray(__paths)
    
    
    def get_images(self):
        return self.__images
    
    
    def get_paths(self):
        return self.__paths
    
    
    def get_size(self):
        return self.__size
    
    
    def set_size(self, size):
        self.__size = size
    
    def set_path(self, path):
        self.__paths = path
        
        
class LBP(BaseImage):
    def __init__(self, path = "", 
                 types= "*", 
                 size=(128, 128), 
                 is_json= False, 
                 json_data = "",
                 json_path = "",
                 save= False,
                 lim = None,
                 move= True,
                 live= False,
                 image_live = None):
        super().__init__(path, types, size, lim, move)



        self.size = size
        self.__live = live
        self.raw_image = image_live
        self._map8 = self.__genBIT(256)
        self._map4 = self.__genBIT(16)
        if is_json:
            self.__feature, self.__path = self.__loadJSON(json_data, 
                                                          json_path)
            self.set_path(self.__path)
            self.set_size(size)
        else:
            self.__feature = self.__get()
            
        if save:
            self.__save(json_data, json_path)
        
    



    def __genBIT(self, bins):
        _dict = {}
        index = 1
        for bits in range(bins):
            x= '{0:08b}'.format(bits)
            U = 0
            for bit in range(7):
                U += 1 if abs(int(x[bit]) - int(x[bit+1])) > 0 else 0
            if U <= 2:
                _dict[x] = index
                index+=1
        return _dict
            

    
    def __get(self):
        raw = self.get_images()
        if self.__live:
            raw = self.raw_image
            raw = np.expand_dims(raw, axis=0)
        # print(raw.shape)
    
        feature = []
        # for image in tqdm(raw, "extract feature"): 
        for image in raw:        
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_hist = []
            copy_hsv = hsv.copy()
            copy_hsv_hist = []
            for channel in range(3):
                hsv[:, :, channel] = local_binary_pattern(hsv[:, :, channel], 8, 1, method='uniform')
                hsv_hist.append(self.__histogram(hsv[:, :, channel]))

                copy_hsv[:, :, channel] = local_binary_pattern(copy_hsv[:, :, channel], 4, 1, method='uniform')
                copy_hsv_hist.append(self.__histogram(copy_hsv[:, :, channel]))

            hsv_hist = np.asarray(hsv_hist).reshape(-1)
            copy_hsv_hist = np.asarray(copy_hsv_hist).reshape(-1)
            feature.append(np.concatenate((hsv_hist, copy_hsv_hist)))

        return np.asarray(feature)
            

    def __lbp(self, img):
        rows, cols = img.shape
        expand = np.full((rows+2, cols+2), 0)
        expand[1:-1, 1:-1] = img.copy()
        output = expand.copy()
        for row in range(1, rows+1):
            for col in range(1, cols+1):
                kernel = expand[row-1:row+2, col-1:col+2].reshape(-1)
                kernel = np.delete(kernel,[3, 4])
                kernel[-3:] = kernel[-3:][::-1]
                kernel = np.concatenate((kernel, [expand[row, col-1]]))
                target = expand[row, col]
                result = (target > kernel).astype(int)
                U = 0
                for bit in range(7):
                    U += abs(result[bit] - result[bit+1])
                if U > 2:
                    output[row, col] = 0
                else:
                    output[row, col] = self._map8[''.join([str(i) for i in result])]
        return output[1:-1, 1:-1]
            
        
    def __histogram(self, img):
        h, _ = np.histogram(img, bins= 59, range=(0, 59))
        h = (h - np.min(h)) / (np.max(h) - np.min(h))
        return h
        
        
    def __save(self, path_image, path_source):
        path = self.get_paths().copy().tolist()
        save_feature = self.__feature.copy().astype(str).tolist()
        
        with open(path_image, "w") as fa:
            fa.write(json.dumps(save_feature))
            
        with open(path_source, "w") as fb:
            fb.write(json.dumps(path))
            
    
    def __loadJSON(self, path_image, path_source):
        data = None
        path_str = None
        with open(path_image, "r") as f:
            data = np.asarray(json.load(f)).astype(float)
        with open(path_source, "r") as f:
            path_str = np.asarray(json.load(f))
        return data, path_str
    
    
    def search(self, ids):
        __images = []
        for id in ids:
            __images.append(
                cv2.resize(
                    cv2.imread(self.get_paths()[id]), 
                    self.get_size()
                )
            )
        return np.asarray(__images)
    
    
    def feature(self):
        return self.__feature
    

