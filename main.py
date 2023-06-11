# import shutil
# from glob import glob
# path = "data\\couch\\*\\*\\*"

# move_path = "data\\couch\\"

# for file in glob(path):
#     target = file.split("\\")[-1]
#     print(target)
#     shutil.move(file, move_path + f"{target}")
# import numpy as np

# a = [[1], [2], [3], [4]]

# a = np.asarray(a).reshape(-1)

# a = a.reshape(1, -1)

# print(a.shape)
# import json
# import numpy as np
# import cv2


# database = np.asarray(json.load(open("annotations\\train.json")))

# labels = database[:, -1]
# database = database[:, :-1].astype(str)
# print(labels.shape)
# print(database)