from glob import glob
import requests, os, json, cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from skimage.feature import local_binary_pattern
import numpy as np
from tqdm import tqdm


class LBP(object):
    def __init__(self, root, slat) -> None:
        self.root = root
        self.slat = slat
        self.labels = []
        self.__createlabels__()


    def __createlabels__(self):
        data = np.asarray(json.load(open("annotations\\train.json")))
        labels = list(set(data[:, -1].tolist()))
        
        for paths in glob(self.root + "*"):
            path = paths.split(self.slat)[-1]
            if path not in labels:
                self.labels.append(path)


    def __histogram(self, img, bins):
        h, _ = np.histogram(img, bins= bins, range=(0, bins))
        h = (h - np.min(h)) / (np.max(h) - np.min(h))
        return h


    def process(self):
        for labels in tqdm(self.labels, "process image by labels"):
            images = []
            y = []
            for file in tqdm(glob(self.root + labels + self.slat + "*"), f"extract feature of {labels}"):
                if file.split(".")[-1] == "json":
                    continue
                image = cv2.imread(file)
                try:
                    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
                except:
                    continue

                lbp_h1 = self.__histogram(local_binary_pattern(h, 8, 1, 'uniform'), 59)
                lbp_s1 = self.__histogram(local_binary_pattern(s, 8, 1, 'uniform'), 59)
                lbp_v1 = self.__histogram(local_binary_pattern(v, 8, 1, 'uniform'), 59)
                lbp_h2 = self.__histogram(local_binary_pattern(h, 4, 1, 'uniform'), 16)
                lbp_s2 = self.__histogram(local_binary_pattern(s, 4, 1, 'uniform'), 16)
                lbp_v2 = self.__histogram(local_binary_pattern(v, 4, 1, 'uniform'), 16)

                feature = np.concatenate((lbp_h1, lbp_s1, lbp_v1, lbp_h2, lbp_s2, lbp_v2))
                images.append(feature)
                y.append(labels)
            if len(images) == 0:
                continue
            images = np.asarray(images).astype(str)
            y = np.asarray(y).reshape(-1, 1)
            print(f"CONCAT A:[{images.shape}] B[{y.shape}]")
            images = np.concatenate((images, y), axis=1)
            self.save(images)

    def save(self, feature):
        obj = np.asarray(json.load(open("annotations\\train.json")))
        if obj.shape[0] == 0:
            obj = feature
        else:
            obj = np.concatenate((obj, feature))
        print(obj.shape)

        with open("annotations\\train.json", "w") as fa:
            fa.write(json.dumps(obj.tolist()))

obj = LBP("data\\", "\\")
obj.process()