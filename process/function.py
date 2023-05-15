import cv2
from glob import glob
import numpy as np
from scipy.spatial.distance import cdist
from skimage.feature import local_binary_pattern
import json
from tqdm import tqdm


class BaseImage:
    """
        Lớp Cơ Sở
        các tham số
            path : đường dẫn đến folder chứa ảnh 
                    trong dự án này là path = "data/raw/"
                    
            types: đuôi file ảnh
                    nếu types= "*" lớp BaseImage sẽ đọc tất cả các ảnh trong folder
                    hoặc nếu không muốn đọc tất cả thì truyền vào 1 list đuôi file
                                                                    ["png", "jpg", ...]
            size : kích cỡ của ảnh mặc định là (128, 128), tất cả các file sẽ được sử lý bằng với kích cỡ này
            
        NOTE: ảnh được đọc vào là ảnh BGR    
        
        Args:
            path :   path to folder image
            types:   end of file, "*" or ['jpg', 'png']
            size :   size of image, default = (128, 128)
    """
    def __init__(self, 
                 path, 
                 types = "*", 
                 size=(128, 128)):
        # đường dẫn ảnh
        self.__path = path
        # đuôi file
        self.__types = types
        # kích cỡ ảnh
        self.__size = size
        # đọc ảnh và đường dẫn của mỗi ảnh
        self.__image, self.__pathImage = self.__getImageFromPath()
        # (N, 128, 128, 3)
        
    # Read image from self.__path
    def __getImageFromPath(self):
        print("LOADING FILE!!!")
        __image = []
        __path  = []
        # duyệt qua tất cả các kiểu đuôi file : "*" hoặc ["png", "jpg", ..., ]
        for type_file in self.__types:
            print(f"FILE: {type_file}")
            for file in tqdm(glob(self.__path + type_file), desc= f"Read file  .{type_file}"):
                # đường dẫn file ảnh
                
                # đọc ảnh bằng opencv, chuyển lại kích cơ về self.__size ví dụ: (128, 128) hoặc (512, 512)...
                __image.append( cv2.resize(cv2.imread(file), self.__size) )
                
                # lưu lại đường dẫn file ảnh
                __path.append( file )
                
        # chuyển về dạng numpy array
        __image = np.asarray(__image)
        __path = np.asarray(__path)
        
        return __image, __path

    # đặt lại file ảnh gốc
    def setImage(self, image):
        self.__image = image

    # lấy file ảnh gốc
    def getImage(self):
        return self.__image.copy()
    
    # lấy kích cỡ ảnh
    def getSize(self):
        return self.__size
    
    # đặt lại kích cỡ ảnh
    def setSize(self, size):
        self.__size = size
        
    # lấy đường đẫn ảnh root
    # vd: "data/raw/"
    def getPath(self):
        return self.__path
    
    # đặt đường dẫn ảnh root
    # vd: "data/raw/" -> "data/image/"
    def setPath(self, path):
        self.__path = path

    # lấy đường dẫn của từng file ảnh
    """
        ví dụ: 
            ["data/raw/1.jpg", "data/raw/2.jpg", ..., "data/raw/N.jpg"]
    """
    def getPathImage(self):
        return self.__pathImage
    
    # đặt đường dẫn của từng file ảnh
    """
    NOTE: chú ý khi đặt lại các tên file vẫn phải được đặt theo thứ từ 1.jpg->N.jpg
        ví dụ:
            ["data/raw/1.jpg", "data/raw/2.jpg", ..., "data/raw/N.jpg"]
            -->
            ["data/image/1.jpg", "data/image/2.jpg", ..., "data/image/N.jpg"]
    """
    def setPathImage(self, path):
        self.__pathImage = path
        

class LBP(BaseImage):
    """_summary_
    
        Lớp LBP kế thừa lớp BaseImage sẽ kế thừa các hàm, biến không phải là hàm, biến đặc biệt
                                                                        hàm, biến đặc biệt là có __ ở đầu
                                                                        ví dụ: self.__path
                                                                               def __load()
                                                                        Nếu muốn truy cập thì phải thông qua
                                                                        hàm get, set của lớp cha
                                                                        ví dụ:
                                                                                def get()
                                                                                def set()
        BaseImage là lớp cha, LBP là lớp con
        
        NOTE:
            Có 2 cách lựa chọn của lớp LBP là đọc từ file raw ví dụ: "data/raw/"
                                           hoặc đọc từ file json ví dụ: "data/json/data.json"
            
            Mô tả file json:
                {
                    "size_image" : [128, 128],
                    "0" : {
                        "vector_feature" : "Đặc điểm của ảnh",
                        "path"           : "đường dẫn ảnh"
                    },
                    "1" : {
                        ....
                    }
                }
            self.__imageColor = None nếu đọc file từ json ngược lại bằng 1 danh sách các ảnh (N, 128, 128, 3) nếu   
                                                                                    đọc từ file raw
            self.__imageLBP   = None nếu đọc file từ json ngược lại bằng 1 danh sách các ảnh (N, 128, 128, 3) được
                                                                                    chuyển từ self.__imageColor
            
            Nếu đọc dữ liệu từ file json thì self.__imageColor và self.__imageLBP là None
    """
    def __init__(self, path = "", 
                       types = "*",
                       size= (128, 128), 
                       from_path= False, 
                       load_path = ""):
        self.__imageColor = None
        self.__imageLBP   = None
        

        if from_path:
            # đọc dữ liệu từ file json
            # vector đặc trưng
            # đường dẫn từng file ảnh
            # kích cỡ file ảnh
            self.__vector_feature, self.path, self.size = self.__load(load_path)
            # đặt lại kích cỡ ảnh cho lớp Cha
            self.setSize(self.size)
            # đặt lại đường dẫn cho mỗi file ảnh cho lớp Cha
            self.setPathImage(self.path)
        else:
            # đọc dữ liệu từ file raw
            super().__init__(path, types, size)
            self.__imageColor = self.getImage()
            # convert BGR -> HSV -> LBP
            self.__imageLBP   = self.__change2LBP()
            # trích xuất đặc trưng từ file raw
            self.__vector_feature = self.__extract_feature()
            
        
    def __change2LBP(self):
        """_summary_
        NOTE:
        [version 1.0] Hiện đang dùng
            Chuyển ảnh từ BGR sang GRAYSCALE 
            chuyển GRAYSCALE sang LBP
            
            
        [version 1.1] Dự kiến
            Chuyển ảnh từ BGR sang HSV 
            chuyển HSV sang LBP
        """
        __lbp = []
        for index, image in enumerate(self.__imageColor):
            temp = image.copy()
            # convert BGR -> GRAY
            r, c, channels = temp.shape
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
            
            # convert GRAY -> LBP
            temp = local_binary_pattern(temp, 8, 1, 'nri_uniform')
            __lbp.append(temp)
            
        return np.asarray(__lbp)
            
                
    
    def __extract_feature(self):
        """_summary_
        [version 1.0] : Trích xuất đặc trưng từ ảnh LBP sang HISTOGRAM 
                        được gọi khi đọc từ file raw
        [version 1.1] : Cập nhật sau
        """
        __histogram = []
        for index, value in tqdm(enumerate(self.__imageLBP), desc="Extract Feature"):
            image = value.copy()
            r, c = image.shape
                
            # tính toán histogram từ lbp
            max_bins = int(np.max(image) + 1)
            hist, _ = np.histogram(image, 
                                bins= max_bins, 
                                range=(0, max_bins))              
            __histogram.append(hist)
            
        return np.asarray(__histogram)
    
    
    def save(self, path_file):
        """_summary_
        lưu đặc trưng của ảnh vào file json
        các đặc trưng được lưu như sau:
            "size_image"     : Kích cỡ của tất cả các ảnh
            "vector_feature" : Đặc trưng của mỗi ảnh
            "path"           : Đường dẫn của mỗi ảnh
        """
        _dict = dict()
        _dict["size_image"] = self.getSize()
        __path_image = self.getPathImage()
        for index, value in tqdm(enumerate(self.__vector_feature), desc= "Save file json"):
            vector_feature = " ".join(value.astype(str))
            rows, cols = self.getSize()
            size_image = f"{rows} {cols}"
            end_file = __path_image[index].split('.')[-1]
            dict_image = {
                "vector_feature" : vector_feature,
                "path"           : f"{self.getPath()}{index+1}.{end_file}"
            }
            _dict[f"{index+1}"] = dict_image
        with open(path_file, "w") as outfile:
            json.dump(_dict, outfile, indent=self.__vector_feature.shape[0] + 1)
            
    
    def __load(self, path_file):
        """_summary_
        Đọc dữ liệu từ file json khiw đã được lưu
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
        Lấy ra các ảnh dạng RGB bằng một danh sách các dường dẫn
        Ví dụ: path = ["data/raw/1.jpg", "data/raw/2.jpg", ...]
        """
        if self.__imageColor is not None:
            # situation load from folder
            # Trường hợp đọc từ file raw
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
            # Trường hợp đọc từ file JSON
            result = []
            for file in path:
                result.append( cv2.resize(
                        cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB), 
                        self.getSize()) )
            return np.asarray(result)
        
        
    def get_image_color_from_id(self, ids: list):
        """_summary_
        Lấy ra các ảnh dạng RGB bằng một danh sách các chỉ số ảnh 
        NOTE: các chỉ số bắt đầu từ 1 cho đến N : số lượng ảnh có trong cơ sở dữ liệu
        Ví dụ: ids = [1, 2, ..., N]
        """
        if self.__imageColor is not None:
            # situation load from folder
            # Trường hợp đọc từ file raw
            result = []
            for id in ids:
                assert 1 <= id <= self.__imageColor.shape[0]
                image = cv2.cvtColor(self.__imageColor[id-1], cv2.COLOR_BGR2RGB)
                result.append(image)
            return np.asarray(result)
        else:
            # situation load from JSON
            # Trường hợp đọc từ file JSON
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
      
    # giống với hàm get_image_color_from_path ở trên nhưng được đọc ở dạng lbp
    def get_image_lbp_from_path(self, path: list):
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
        
    # giống với hàm get_image_color_from_id ở trên nhưng được đọc ở dạng lbp
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
    """
        Hàm tính khoảng cách sẽ được dùng từ [version 2.0] trở đi
    """
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