from PIL import Image
import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage.feature import local_binary_pattern
from django.shortcuts import render, redirect
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import base64
def histogram(img, bins):
    h, _ = np.histogram(img, bins= bins, range=(0, bins))
    h = (h - np.min(h)) / (np.max(h) - np.min(h))
    return h

def webcam(request):
    if (request.method == 'POST'):
        image = Image.open(request.FILES['upload_image']).convert('RGB')
        open_cv_image = np.array(image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 


        h, s, v = cv2.split(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2HSV))
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        Y, cr, cb = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2YCrCb)

        lbp_h1 = histogram(local_binary_pattern(h, 8, 1, 'uniform'), 59)
        lbp_s1 = histogram(local_binary_pattern(s, 8, 1, 'uniform'), 59)
        lbp_v1 = histogram(local_binary_pattern(v, 8, 1, 'uniform'), 59)
        lbp_h2 = histogram(local_binary_pattern(h, 4, 1, 'uniform'), 16)
        lbp_s2 = histogram(local_binary_pattern(s, 4, 1, 'uniform'), 16)
        lbp_v2 = histogram(local_binary_pattern(v, 4, 1, 'uniform'), 16)
        gray = histogram(local_binary_pattern(gray, 8, 1, 'uniform'), 59)
        Y = histogram(local_binary_pattern(Y, 8, 1, 'uniform'), 59)
        cr = histogram(local_binary_pattern(cr, 8, 1, 'uniform'), 59)
        cb = histogram(local_binary_pattern(cb, 8, 1, 'uniform'), 59)

        feature = np.concatenate((lbp_h1, lbp_s1, lbp_v1, lbp_h2, lbp_s2, lbp_v2)).reshape(1, -1)



    template = loader.get_template("home/base.html")
    context = {
        
    }
      
    return HttpResponse(template.render(context, request))