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
data_dir = "C:/Users/ckris/Desktop/DeepFakeProject/deepfake-detection-challenge/"
training_data="C:/Users/ckris/Desktop/DeepFakeProject/dfdc_train_part_0"
sample_training_data = "train_sample_videos"
test_data = "test_videos"
haar_cascade = "C:/Users/ckris/Desktop/DeepFakeProject/haar-cascades-for-face-detection"

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
            # print(frame.shape)
            # box = bounds(frame)[0]
            # print(box)
            # frame = frame[box[1]:box[3], box[0]:box[2]]
            # print(frame)
            # print(frame.shape)
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # ax.imshow(frame)
            frames.append(frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    return np.array(frames)

    # print(img)
    # # print(ret)
    # # print(frame)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # print(frame.shape)
    # ax.imshow(frame)
    # img.release()

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
        frame = temp_x[i]
        # print(frame)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # ax.imshow(frame)
        # print(bounds(temp_x[i]))
        # gray = cv.cvtColor(temp_x[i], cv.COLOR_BGR2GRAY)
        faces = bounds(frame)
        all_faces += list(faces)
        # plt.imshow(face)
        # print(faces.shape)
        # print(faces.dtype)
        # # print(box)
        # # frame = temp_x[i][box[1]:box[3], box[0]:box[2]]
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # # frame = cv.cvtColor(face, cv.COLOR_BGR2RGB)
        # frame = face
        # ax.imshow(frame)
        # break
    # return 
        # temp_x[i] = temp_x[i][box[1]:box[3], box[0]:box[2]]
    temp_y = np.empty(len(all_faces), dtype=object)
    temp_y[:] = metadata.label[metadata.index == video_name]
    # return temp_x, temp_y
    return np.array(all_faces), temp_y


temp = missing_data(metadata)
fake_sample_videos = list(metadata.loc[metadata.label=='FAKE'].sample(4).index)

# for video_path in fake_sample_videos:
#     img_frm_video(os.path.join(data_dir, sample_training_data, video_path))

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
    
    





