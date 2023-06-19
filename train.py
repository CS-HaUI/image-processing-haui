import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing


def load_images(path):

    images = []
    filenames = glob(path)
    for file in tqdm(filenames):
        image = cv2.resize(cv2.imread(file), (200, 200))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    
    return np.array(images)


def PLOT(data, classname):
    fig, ax = plt.subplots(nrows=1, ncols=5)
    fig.suptitle(classname, fontsize=15)
    ax[0].imshow(data[0])
    ax[1].imshow(data[1])
    ax[2].imshow(data[2])
    ax[3].imshow(data[3])
    ax[4].imshow(data[4])
    plt.show()



def extract_lbp(images, method):
    lbps = []
    for image in tqdm(images):
        lbp = local_binary_pattern(image, P=8, R=1, method=method)
        lbps.append(lbp)
    
    return np.array(lbps)



def show_Example_lbp(images, labels, classnames):
    index = 0
    fig, ax = plt.subplots(nrows=1, ncols=4)
    ax[0].set_title(classnames[labels[index]])
    ax[0].imshow(images[index], cmap='gray')
    index+=432

    ax[1].set_title(classnames[labels[index]])
    ax[1].imshow(images[index], cmap='gray')
    index+=432

    ax[2].set_title(classnames[labels[index]])
    ax[2].imshow(images[index], cmap='gray')
    index+=432

    ax[3].set_title(classnames[labels[index]])
    ax[3].imshow(images[index], cmap='gray')
    index+=432    
    plt.show()



def create_histogram(images, bins):
    all_histogram = []
    for image in tqdm(images):
        hist = np.histogram(image, bins=bins)[0]
        all_histogram.append(hist.flatten())
    return np.asarray(all_histogram)



def create_histogram2(images, div, bins):
    all_histograms = []
    for image in tqdm(images):
        grid = np.arange(0, image.shape[1]+1, image.shape[1]//div)

        temp = []

        for i in range(1, len(grid)):
            for j in range(1, len(grid)):
                sub_image = image[grid[i-1]:grid[i], grid[j-1]:grid[j]]

                temp_histogram = np.histogram(sub_image, bins=bins)[0]
                temp.append(temp_histogram)

        histogram = np.array(temp).flatten()
        all_histograms.append(histogram)
        
    return np.array(all_histograms)


main_path = "data\\"
classnames = ['aluminium_foil', 'brown_bread', 'corduroy', 
              'cork', 'cotton', 'cracker', 'lettuce_leaf', 
              'linen', 'white_bread', 'wood', 'wool']

class_images = [load_images(main_path + i + "\\*\\*") for i in classnames]


# PLOT(class_images[0], classnames[0])
# PLOT(class_images[1], classnames[1])
# PLOT(class_images[2], classnames[2])



all_images = []
for index, value in enumerate(class_images):
    tmp = []
    for jndex, image in enumerate(class_images[index]):
        tmp.append(cv2.cvtColor(class_images[index][jndex], cv2.COLOR_RGB2GRAY))
    all_images.append(np.asarray(tmp))

all_images = np.vstack(tuple(all_images))

print(all_images.shape)  # (4752, 200, 200)


samples = 432
labels = np.array([0]*samples + [1]*samples + \
                  [2]*samples + [3]*samples + \
                  [4]*samples + [5]*samples + \
                  [6]*samples + [7]*samples + \
                  [8]*samples + [9]*samples + \
                  [10]*samples)


X_train, X_test, y_train, y_test = train_test_split(all_images, 
                                                    labels, 
                                                    test_size=0.3)

print('X_train.shape\t', X_train.shape)
print('X_test.shape\t', X_test.shape)
print('y_train.shape\t', y_train.shape)
print('y_test.shape\t', y_test.shape)

X_train_lbp = extract_lbp(X_train, 'default')
X_test_lbp = extract_lbp(X_test, 'default')

bin = 80
X_train_hist = create_histogram2(X_train_lbp, 4, bin).copy()
X_test_hist = create_histogram2(X_test_lbp, 4, bin).copy()


scaler_xtrain = preprocessing.StandardScaler().fit(X_train_hist)
scaler_xtest = preprocessing.StandardScaler().fit(X_test_hist)

X_train_hist = scaler_xtrain.transform(X_train_hist)
X_test_hist = scaler_xtest.transform(X_test_hist)

# KNN
model_knn = KNeighborsClassifier(n_neighbors=1)
model_knn.fit(X_train_hist, y_train)

print('KNN train acc:', model_knn.score(X_train_hist, y_train))
print('KNN test acc:', model_knn.score(X_test_hist, y_test))


acc = []
title_acc = []


wr = []
title_wr = []
title_actual = []

predictions = model_knn.predict(X_test_hist)
for index, value in enumerate(predictions):
    if value == y_test[index] and len(acc) < 3 and classnames[value] not in title_acc:
        acc.append(X_test[index])
        title_acc.append(classnames[value])
    if value != y_test[index] and len(wr) < 3 and classnames[value] not in title_wr:
        wr.append(X_test[index])
        title_wr.append(classnames[value])
        title_actual.append(classnames[y_test[index]])

fig, ax = plt.subplots(nrows=2, ncols=3)
ax[0][0].imshow(cv2.cvtColor(acc[0], cv2.COLOR_GRAY2RGB))
ax[0][0].set_title(f"actual: {title_acc[0]}\npred: {title_acc[0]}")
ax[0][0].axis('off')  

ax[0][1].imshow(cv2.cvtColor(acc[1], cv2.COLOR_GRAY2RGB))
ax[0][1].set_title(f"actual: {title_acc[1]}\npred: {title_acc[1]}")
ax[0][1].axis('off')  

ax[0][2].imshow(cv2.cvtColor(acc[2], cv2.COLOR_GRAY2RGB))
ax[0][2].set_title(f"actual: {title_acc[2]}\npred: {title_acc[2]}")
ax[0][2].axis('off')  


ax[1][0].imshow(cv2.cvtColor(wr[0], cv2.COLOR_GRAY2RGB))
ax[1][0].set_title(f"actual: {title_actual[0]}\npred: {title_wr[0]}")
ax[1][0].axis('off')  

ax[1][1].imshow(cv2.cvtColor(wr[1], cv2.COLOR_GRAY2RGB))
ax[1][1].set_title(f"actual: {title_actual[1]}\npred: {title_wr[1]}")
ax[1][1].axis('off')  

ax[1][2].imshow(cv2.cvtColor(wr[1], cv2.COLOR_GRAY2RGB))
ax[1][2].set_title(f"actual: {title_actual[2]}\npred: {title_wr[2]}")
ax[1][2].axis('off')  
plt.show()