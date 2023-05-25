
from process.function import LBP, Distance
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


class AntiSpoofing:
    def __init__(self, 
                 root,
                 type,
                 size,
                 P,
                 R,
                 method,
                 test,
                 file_json= ""):
        self.__root = root
        self.__type = type
        self.__size = size
        self.__p    = P
        self.__r    = R
        self.__method = method
        self.__test   = test
        self.__json = file_json
        self.__distance = Distance()
        

        self.__Xtrain = None
        self.__ytrain = None
        self.__Xtest  = None
        self.__ytest  = None
        self.__save   = None
        if len(self.__json):
            self.__Xtrain, self.__ytrain = self.__load()
        else:
            self.__Xtrain, self.__ytrain, self.__save = self.__feature(self.__root)
    
    
    def __load(self):
        with open(self.__json, "r") as f:
            data = json.load(f)
            data = np.asarray(data)
            return data[:, :-1], data[:, -1]
    
    
    def __feature(self, path):
        obj_live = LBP(path= path + "live\\", types= self.__type, size= self.__size, P= self.__p, R= self.__r, method= self.__method)
        obj_spoof = LBP(path= path + "spoof\\", types= self.__type, size= self.__size, P= self.__p, R= self.__r, method= self.__method)
        
        feature_live = obj_live.get_feature()
        labels_live = np.full((feature_live.shape[0]), 0)
        
        feature_spoof = obj_spoof.get_feature()
        labels_spoof = np.full((feature_spoof.shape[0]), 1)
        
        labels = np.concatenate((labels_live, labels_spoof)).reshape(-1, 1)
        
        feature = np.concatenate((feature_live, feature_spoof), axis= 0)
        vector = np.concatenate((feature, labels), axis= 1)
        
        return feature, labels, vector
    
    
    def get_save_feature(self):
        return self.__save
    
    
    def get_X_train(self):
        return self.__Xtrain
    
    
    def get_y_train(self):
        return self.__ytrain
        
        
    def predict(self):
        total = 0
        acc   = []
        true = 0
        feature, labels, _ = self.__feature(self.__test)
        labels = labels.reshape(-1)
        for index, value in tqdm(enumerate(feature), desc= "predict value"):
            self.__distance.reset(self.__Xtrain, value.reshape(1, -1), 101)
            index_labels = self.__distance.get_data()
            labels_predict = self.__ytrain[index_labels]
            
            labels_one = labels_predict[labels_predict == 0].shape[0]
            labels_two = labels_predict[labels_predict == 1].shape[0]
            labels_predict = None
            if labels_one > labels_two:
                labels_predict = 0
            else:
                labels_predict = 1
            labels_actual = labels[index]
            total += 1
            if labels_actual == labels_predict:
                true += 1
            acc.append(true / total)
        plt.plot(acc)
        plt.show()
        

        
    
        
"""
1, 2, 3, 4

1/4, 2/4, 3/4, 4/4

1/4 * 255 = 63.75

1/3

"""