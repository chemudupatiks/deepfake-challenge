# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 01:14:28 2020

@author: ckris
"""

import numpy as np
import pandas as pd 
import os
import matplotlib
import matplotlib.pyplot as plt 
# import seaborn as sb
# from tqdm import tqdm_notebook
import torch
import tensorflow as tf
from face_recognition_box import bounds
import cv2 as cv

#Loading data
data_dir = "deepfake-detection-challenge/"
training_data="dfdc_train_part_0"
sample_training_data = "train_sample_videos"
test_data = "test_videos"
haar_cascade = "haar-cascades-for-face-detection"

metadata= pd.read_json(os.path.join(os.path.join(data_dir,sample_training_data), "metadata.json"))

metadata = metadata.T
metadata.head()

def missing_data(data):
    total = data.isnull().sum()
    # print(total)
    percent = (data.isnull().sum()/data.isnull().count()*100)
    # print(percent)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    # print(tt)
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    # print(tt)
    return(np.transpose(tt))

# 1. Make an array with 5 random frames from each video with the label
# 2. Crop the photos so that the only the face is seen
# 3. Make dicriminator model
# 4. Train the model, and test it using the test set using the same process. 


def all_frames_frm_video(video_path):
    cap = cv.VideoCapture(video_path)

    frames = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frames.append(frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    return np.array(frames)

def training_ds(video_path, video_name, num_imgs):
    print(video_path)
    frames = all_frames_frm_video(video_path)
    # print(frames.shape)
    indx = np.linspace(0, len(frames)-1, num_imgs, dtype=int)
    # print(indx)
    temp_x = frames[indx,:,:,:]
    all_faces = []
    idx = 0
    for i in range(temp_x.shape[0]):
        frame = cv.cvtColor(temp_x[i], cv.COLOR_BGR2RGB)
        faces = bounds(frame)
        all_faces += list(faces)
    temp_y = np.empty(len(all_faces), dtype=object)
    temp_y[:] = metadata.label[metadata.index == video_name]

    return np.array(all_faces), temp_y


temp = missing_data(metadata)
fake_sample_videos = list(metadata.loc[metadata.label=='FAKE'].sample(4).index)

training_x = []
training_y = []
count = 0
for video_path in metadata.index:
    temp_x, temp_y = training_ds(os.path.join(data_dir, sample_training_data, video_path), video_path, 5)
    print(temp_x.shape)
    if(count == 0):
        training_x = temp_x
        training_y = temp_y
        count=1
    else:        
        training_x = np.append(training_x, temp_x, axis=0)
        training_y = np.append(training_y, temp_y, axis=0)
        count += 1
        if count > 4:
            break
    
    





