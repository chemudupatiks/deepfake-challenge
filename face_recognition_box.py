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
# from mtcnn.mtcnn import MTCNN


"""
@params: pixels: matrix of pixel values of the image
@params: path  : path to the pre-trained haarcascade models
"""
def bounds(pixels, path = "C:\\Users\\ckris\\Desktop\\DeepFakeProject\\haar-cascades-for-face-detection\\"):
    # load the pre-trained model
    # classifier = CascadeClassifier(os.path.join(path, 'haarcascade_frontalface_default.xml'))
    classifier = MTCNN(select_largest=False, post_process=False, keep_all=True, device='cuda:0')
    # perform face detection
    # bboxes = classifier.detectMultiScale(pixels, scaleFactor = 1.1, minNeighbors=8)#, minSize=(10,10)
    img = Image.fromarray(pixels)
    # print(img)
    # plt.figure()
    # plt.imshow(img)
    # plt.axis('off')
    bboxes = classifier(pixels)
    # print("bboxes shape: " + str(bboxes.shape))
    faces = []
    if (len(list(bboxes.shape)) > 3):
        for bbox in bboxes:
            # print(bbox.shape)
            face = bbox.permute(1, 2, 0).int().numpy()
            if len(faces)>0:
                faces = np.append(faces, [face], axis=0)
            else:
                faces = np.array([face])
            # print(bbox)
            # plt.figure()
            # plt.axis('off')
            # plt.imshow(face)
    else:        
        bbox = bboxes.permute(1, 2, 0).int().numpy()
        faces = np.array([face])
        # plt.figure()
        # plt.axis('off')
        # plt.imshow(bbox)
    
    return faces

   
    
    
    
    # print(bboxes.shape)
    # print(bboxes.permute(1,2,0))
    # print(bboxes.permute(1,2,0).int())
    # array to hold vertices info 
    # vert = []
    # for box in bboxes:
    #    	# extract
    #    	x, y, width, height = box['box']
    #    	x2 = x + width
    #     y2 = y + height
    #     vert.append([x, y, x2, y2])
    #     #return only the first box
    #     break
    # return vert
    # return bboxes
    

    

