import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import shutil
from scipy.spatial.distance import cdist


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
                 move= True):
        super().__init__(path, types, size, lim, move)



        self.size = size
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
        feature = []
        for image in tqdm(raw, "extract feature"):         
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_hist = []

            for channel in range(3):
                hsv[:, :, channel] = self.__lbp(hsv[:, :, channel])
                hsv_hist.append(self.__histogram(hsv[:, :, channel]))

            hsv_hist = np.asarray(hsv_hist).reshape(-1)
            feature.append(hsv_hist)

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
    

class FaceRecognition:
    def __init__(self,
                 config = "data/JSON/config.json",
                 data_json= "data/JSON/data.json",
                 path_json= "data/JSON/path.json",
                 labels_json= "data/JSON/labels.json",
                 KNN = 13) -> None:
        self.obj = json.load(open(config))
        self.vector_feature = np.asarray(json.load(open(data_json))).astype(float)
        self.vector_path    = np.asarray(json.load(open(path_json))).astype(str)
        self.vector_labels    = np.asarray(json.load(open(labels_json))).astype(int)
        self.data_json = data_json
        self.path_json = path_json
        self.labels_json = labels_json
        self.knn = KNN
        
    def fit(self):

        batch = self.obj['batch']

        # FACE
        face = self.obj['train']['face']
        number_face = self.obj['train']['img_face']

        for image in tqdm(range(0, number_face, batch), "Extract Face"):
            lbp_face = LBP(path= face, size= (112, 112), lim= batch, move= True)
            if lbp_face.feature().shape[0] == 0:
                continue
            if self.vector_feature.shape[0] == 0:
                self.vector_feature = lbp_face.feature()
            else:
                self.vector_feature = np.concatenate((self.vector_feature, lbp_face.feature()))
            if self.vector_path.shape[0] == 0:
                self.vector_path = lbp_face.get_paths()
            else:
                self.vector_path = np.concatenate((self.vector_path, lbp_face.get_paths()))
            if self.vector_labels.shape[0] == 0:
                self.vector_labels = np.full((lbp_face.get_paths().shape[0]), 1)
            else:
                self.vector_labels = np.concatenate((self.vector_labels, np.full((lbp_face.get_paths().shape[0]), 1)))

            
        # NONFACE
        nonface = self.obj['train']['nonface']
        number_nonface = self.obj['train']['img_nonface']

        for image in tqdm(range(0, number_nonface, batch), "Extract Nonface"):
            lbp_nonface = LBP(path= nonface, size= (112, 112), lim= batch, move= True)
            if lbp_nonface.feature().shape[0] == 0:
                continue
            if self.vector_feature.shape[0] == 0:
                self.vector_feature = lbp_nonface.feature()
            else:
                self.vector_feature = np.concatenate((self.vector_feature, lbp_nonface.feature()))
            if self.vector_path.shape[0] == 0:
                self.vector_path = lbp_nonface.get_paths()
            else:
                self.vector_path = np.concatenate((self.vector_path, lbp_nonface.get_paths()))
            if self.vector_labels.shape[0] == 0:
                self.vector_labels = np.full((lbp_face.get_paths().shape[0]), 0)
            else:
                self.vector_labels = np.concatenate((self.vector_labels, np.full((lbp_face.get_paths().shape[0]), 0)))
            
            

        
        vector_feature_temp = self.vector_feature.copy().astype(str).tolist()
        vector_labels_temp = self.vector_labels.copy().astype(str).tolist()
        vector_path_temp = self.vector_path.copy().tolist()

        with open(self.data_json, "w") as f:
            f.write(json.dumps(vector_feature_temp))
            
        with open(self.path_json, "w") as f:
            f.write(json.dumps(vector_path_temp))

        with open(self.labels_json, "w") as f:
            f.write(json.dumps(vector_labels_temp))


    def val(self):
        vector_feature = None
        vector_path = None
        vector_labels = None
        acc = 0
        wra = 0
        total = 0
        batch = self.obj['batch']


        # FACE
        face = self.obj['val']['face']
        number_face = self.obj['val']['img_face']

        for image in tqdm(range(0, number_face, batch), "Extract Face"):
            lbp_face = LBP(path= face, size= (112, 112), lim= batch, move= False)
            if lbp_face.feature().shape[0] == 0:
                continue
            if vector_feature is None:
                vector_feature = lbp_face.feature()
            else:
                vector_feature = np.concatenate((vector_feature, lbp_face.feature()))
            if vector_path is None:
                vector_path = lbp_face.get_paths()
            else:
                vector_path = np.concatenate((vector_path, lbp_face.get_paths()))
            if vector_labels is None:
                vector_labels = np.full((lbp_face.get_paths().shape[0]), 1)
            else:
                vector_labels = np.concatenate((vector_labels, np.full((lbp_face.get_paths().shape[0]), 1)))


            
        # NONFACE
        nonface = self.obj['val']['nonface']
        number_nonface = self.obj['val']['img_nonface']

        for image in tqdm(range(0, number_nonface, batch), "Extract Nonface"):
            lbp_nonface = LBP(path= nonface, size= (112, 112), lim= batch, move= False)
            if lbp_nonface.feature().shape[0] == 0:
                continue
            if vector_feature is None:
                vector_feature = lbp_nonface.feature()
            else:
                vector_feature = np.concatenate((vector_feature, lbp_nonface.feature()))
            if vector_path is None:
                vector_path = lbp_nonface.get_paths()
            else:
                vector_path = np.concatenate((vector_path, lbp_nonface.get_paths()))
            if vector_labels is None:
                vector_labels = np.full((lbp_face.get_paths().shape[0]), 0)
            else:
                vector_labels = np.concatenate((vector_labels, np.full((lbp_face.get_paths().shape[0]), 0)))
        
        for item, value in tqdm(enumerate(vector_feature), "evaluate"):

            feature = value.copy().reshape(1, -1)
            labels = int(vector_labels[item])
            labels_predict = None

            distance = cdist(self.vector_feature, feature)
            indexs = np.arange(len(distance)).reshape(-1, 1)
            result = np.concatenate((distance, indexs), axis=1)
            indexs = result[np.argsort(result[:, 0])][:self.knn, 1].astype(int)

            print(indexs)
            founded_labels = self.vector_labels[indexs]
            
            ones = len(founded_labels[founded_labels == 1])
            zeros = len(founded_labels[founded_labels == 0])  
            if ones > zeros:
                labels_predict = 1
            else:
                labels_predict = 0
            if labels == labels_predict:
                acc+=1
            else:
                wra+=1
            total += 1
        print(f"ACC: {acc}/{total}: {acc/total}%")