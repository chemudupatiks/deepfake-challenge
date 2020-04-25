# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 16:39:33 2020

@author: ckris
@reference: https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/
"""

# plot photo with detected faces using opencv cascade classifier
from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
import os
from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


"""
@params: pixels: matrix of pixel values of the image
@params: path  : path to the pre-trained haarcascade models
"""
def bounds(pixels):
    classifier = MTCNN(select_largest=False, post_process=False, keep_all=True, device='cuda:0')
    img = Image.fromarray(pixels)
    bboxes = classifier(pixels)
    faces = []
    if (len(list(bboxes.shape)) > 3):
        for bbox in bboxes:
            # print(bbox.shape)
            face = bbox.permute(1, 2, 0).int().numpy()
            if len(faces)>0:
                faces = np.append(faces, [face], axis=0)
            else:
                faces = np.array([face])
    else:        
        bbox = bboxes.permute(1, 2, 0).int().numpy()
        faces = np.array([face])
   
    return faces

    

