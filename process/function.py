import cv2
from glob import glob
import numpy as np
from scipy.spatial.distance import cdist
from skimage.feature import local_binary_pattern
import json
from tqdm import tqdm


class BaseImage:
    """
        Args:
            path :   path to folder image
            types:   end of file, "*" or ['jpg', 'png']
            size :   size of image, default = (128, 128)
    """
    def __init__(self, 
                 path, 
                 types = "*", 
                 size=(128, 128)):
        
        self.__path = path
        self.__types = types
        self.__size = size
        self.__image, self.__pathImage = self.__getImageFromPath()
        # (N, 128, 128, 3)
        
    # Read image from self.__path
    def __getImageFromPath(self):
        """_summary_
        Args:
            None
        Returns:
            __image (nparray): list image
            __path  (nparray): list path for each image
        """
        print("LOADING FILE!!!")
        __image = []
        __path  = []
        for type_file in tqdm(iterable= self.__types):
            print(f"FILE: {type_file}")
            for file in tqdm(glob(self.__path + type_file), desc= f"Read file  .{type_file}"):
                __image.append( cv2.resize(cv2.imread(file), self.__size))
                __path.append( file )
        __image = np.asarray(__image)
        __path = np.asarray(__path)
        return __image, __path

    
    def setImage(self, image):
        self.__image = image

    
    def getImage(self):
        return self.__image.copy()
    
    
    def getSize(self):
        return self.__size
    
    
    def setSize(self, size):
        self.__size = size
        
        
    def getPath(self):
        return self.__path
    
    
    def setPath(self, path):
        self.__path = path


    def getPathImage(self):
        return self.__pathImage
    
    
    def setPathImage(self, path):
        self.__pathImage = path
        

class LBP(BaseImage):
    """_summary_
        Args:
            path     :   path to folder image
            types    :   end of file, "*" or ['jpg', 'png']
            size     :   size of image, default = (128, 128)
            from_path:   Boolean
                         If True, data read from file json else data read from path
            load_path:   path to file json
    """
    def __init__(self, path = "", 
                       types = "*",
                       size= (128, 128), 
                       from_path= False, 
                       load_path = ""):
        self.__imageColor = None
        self.__imageLBP   = None
        if from_path:
            self.__vector_feature, self.path, self.size = self.__load(load_path)
            self.setSize(self.size)
            self.setPathImage(self.path)
        else:
            super().__init__(path, types, size)
            self.__imageColor = self.getImage()
            # convert BGR -> HSV -> LBP
            self.__imageLBP   = self.__change2LBP()
            self.__vector_feature = self.__extract_feature()
            
        
    def __change2LBP(self):
        """_summary_
        Args:
            None

        Returns:
            __lbp (nparray): shape=(N, 128, 128 3) [LBP image]
                             convert rgb, bgr to lbp
        """
        # start from (0, 0) --> (0, 1) --> (0, 2)
        #            --> (1, 2) --> (2, 2) --> (2, 1)
        #            --> (2, 0) --> (1, 0)
        __lbp = []
        for index, image in enumerate(self.__imageColor):
            temp = image.copy()
            # convert BGR -> HSV
            r, c, channels = temp.shape
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2HSV)
            
            # convert HSV -> LBP
            for channel in range(channels):
                temp[:, :, channel] = local_binary_pattern(temp[:, :, channel], 8, 1, 'nri_uniform')
            __lbp.append(temp)
        return np.asarray(__lbp)
            
                
    
    def __extract_feature(self):
        """_summary_
        Args:
            None
        Returns:
            __histogram (nparray): vector feature from image -> (N, 177)
        """
        __histogram = []
        for index, value in enumerate(self.__imageLBP):
            image = value.copy()
            r, c, channels = image.shape
            hist_array = np.asarray([])
            for channel in range(channels):
                max_bins = np.max(image[:, :, channel]) + 1
                hist, _ = np.histogram(image[:, :, channel], 
                                       bins= max_bins, 
                                       range=(0, max_bins))
                hist_array = np.concatenate((hist_array, hist))
            __histogram.append(hist_array)
        return np.asarray(__histogram)
    
    
    def save(self, path_file):
        """_summary_
        Args:
            path_file (str): path file json to save data
            
        Returns:
            None
        """
        _dict = dict()
        _dict["size_image"] = self.getSize()
        for index, value in tqdm(enumerate(self.__vector_feature), desc= "Save file json"):
            vector_feature = " ".join(value.astype(str))
            rows, cols = self.getSize()
            size_image = f"{rows} {cols}"
            dict_image = {
                "vector_feature" : vector_feature,
                "path"           : f"{self.getPath()}{index+1}.jpg"
            }
            _dict[f"{index+1}"] = dict_image
        with open(path_file, "w") as outfile:
            json.dump(_dict, outfile, indent=self.__vector_feature.shape[0] + 1)
            
    
    def __load(self, path_file):
        """_summary_

        Args:
            path_file (str): path file json to load data
            
        Returns:
            vector_feature (nparray)
            path           (nparray)
            size_image     (tuple)
        """
        data = json.load(open(path_file))
        size_image = tuple(data['size_image'])
        del data["size_image"]
        vector_feature   = []
        path = []
        for index, value in tqdm(enumerate(data), desc= "Read data from json"):
            feature = np.asarray(data[value]['vector_feature'].split())
            vector_feature.append(feature)
            path.append(data[value]['path'])

        vector_feature = np.asarray(vector_feature)
        path = np.asarray(path)
        
        return vector_feature, path, size_image
    
    
    def get_vector_feature(self):
        return self.__vector_feature
    
    
    def get_image_color(self):
        return self.__imageColor
    
    
    def get_image_LBP(self):
        return self.__imageLBP
    
    
    def get_image_color_from_path(self, path: list):
        """_summary_

        Args:
            path (list): list path to folder image

        Returns:
            result (nparray): list image
        """
        if self.__imageColor is not None:
            # situation load from folder
            __path = self.getPathImage()
            result = []
            for file in path:
                indexs = np.where(__path == file)[0]
                if len(indexs) > 0:
                    image = self.__imageColor[indexs[0]]
                    result.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return np.asarray(result)
        else:
            # situation load from JSON
            result = []
            for file in path:
                result.append( cv2.resize(
                        cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB), 
                        self.getSize()) )
            return np.asarray(result)
        
        
    def get_image_color_from_id(self, ids: list):
        """_summary_

        Args:
            id (list): list id to image

        Returns:
            result (nparray): list image
        """
        if self.__imageColor is not None:
            # situation load from folder
            result = []
            for id in ids:
                assert 1 <= id <= self.__imageColor.shape[0]
                image = cv2.cvtColor(self.__imageColor[id-1], cv2.COLOR_BGR2RGB)
                result.append(image)
            return np.asarray(result)
        else:
            # situation load from JSON
            list_path = self.getPathImage()
            result = []
            for id in ids:
                assert 1 <= id <= self.__vector_feature.shape[0]
                image = cv2.resize( 
                                   cv2.cvtColor(cv2.imread(list_path[id-1]), cv2.COLOR_BGR2RGB), 
                                   self.getSize()
                                   )
                result.append(image)
            return np.asarray(result)
      

    def get_image_lbp_from_path(self, path: list):
        """_summary_

        Args:
            path (list): list path to folder image

        Returns:
            result (nparray): list image
        """
        if self.__imageLBP is not None:
            # situation load from folder
            __path = self.getPathImage()
            result = []
            for file in path:
                indexs = np.where(__path == file)[0]
                if len(indexs) > 0:
                    result.append(self.__imageLBP[indexs[0]])
            return np.asarray(result)
        else:
            # situation load from JSON
            result = []
            for file in path:
                # temp = cv2.resize(
                #         cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2HSV), 
                #         self.getSize())
                # for channel in range(3):
                #     temp[:, :, channel] = local_binary_pattern(temp[:, :, channel], 8, 1, 'nri_uniform')
                temp = cv2.resize(
                        cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY), 
                        self.getSize())
                temp = local_binary_pattern(temp, 8, 1, 'nri_uniform')
                result.append(temp)
            return np.asarray(result)
        
        
    def get_image_lbp_from_id(self, ids: list):
        """_summary_

        Args:
            id (list): list id to image

        Returns:
            result (nparray): list image
        """
        if self.__imageLBP is not None:
            # situation load from folder
            result = []
            for id in ids:
                assert 1 <= id <= self.__imageLBP.shape[0]
                result.append(self.__imageLBP[id-1])
            return np.asarray(result)
        else:
            # situation load from JSON
            list_path = self.getPathImage()
            result = []
            for id in ids:
                assert 1 <= id <= self.__vector_feature.shape[0]
                # temp = cv2.resize( 
                #                    cv2.cvtColor(cv2.imread(list_path[id-1]), cv2.COLOR_BGR2HSV), 
                #                    self.getSize()
                #                    )
                # for channel in range(3):
                #     temp[:, :, channel] = local_binary_pattern(temp[:, :, channel], 8, 1, 'nri_uniform')
                temp = cv2.resize( 
                                   cv2.cvtColor(cv2.imread(list_path[id-1]), cv2.COLOR_BGR2GRAY), 
                                   self.getSize()
                                   )
                temp = local_binary_pattern(temp, 8, 1, 'nri_uniform')
                result.append(temp)
            return np.asarray(result)

class Distance:
    def __init__(self, object_one, 
                       object_two, 
                       database,
                       k= 3):
        self.one  = object_one
        self.two  = object_two
        self.data = database
        self.k = k
        
    
    def __distance_fromA2B(self):
        # (N, M) -- (1, M) -> (N, 1)
        distance = cdist(self.two, self.one)
        # range from 0 to N -> axis = 1, vertical
        indexs   = np.arange(len(distance)).reshape(-1, 1)
        distance = np.concatenate((distance, indexs), axis= 1)
        distance = distance[np.argsort(distance[:, 0])]
        return distance[:, 1].astype(int)
    
    
    def get_data(self):
        distance = self.__distance_fromA2B()
        print(distance)
        return self.data[distance[:self.k], :]