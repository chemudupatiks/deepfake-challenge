# -*- coding: utf-8 -*-
"""
Created on Sat May  2 17:04:11 2020

@author: ckris
"""
import os
import glob
import json
import torch
import cv2 as cv
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from facenet_pytorch import MTCNN
import time

'''
functions
'''
def ExtractFaces(video_dir_path, output_dir, n_frames):
    metadata = pd.read_json(os.path.join(video_dir_path, 'metadata.json'))
    metadata = metadata.T
    
    video_files = glob.glob(os.path.join(video_dir_path, '*.mp4'))
    
    model = MTCNN(margin=14, factor=0.5, keep_all=True, device='cuda:0').eval()
    
    with torch.no_grad():
        for video_path in video_files:
            dir_name = video_path.split('\\')[-1].split('.')[0]
            save_dir = os.path.join(output_dir, dir_name)
            v_cap = cv.VideoCapture(video_path)
            v_len = int(v_cap.get(cv.CAP_PROP_FRAME_COUNT))
    
            # Pick 'n_frames' evenly spaced frames to sample
            if n_frames is None:
                sample = np.arange(0, v_len)
            else:
                sample = np.linspace(0, v_len - 1, n_frames).astype(int)
    
            # Loop through frames
            for j in range(v_len):
                success = v_cap.grab()
                if j in sample:
                    # Load frame
                    success, frame = v_cap.retrieve()
                    if not success:
                        continue
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    
                    # print(j)
                    save_path = os.path.join(save_dir, f'{j}.jpg')
    
                    model([frame], save_path=save_path)
    
            v_cap.release()
            break
    
'''
Main Script 
'''
# video_dir_path = 'deepfake-detection-challenge\\train_sample_videos\\'
video_dir_path = 'deepfake-detection-challenge/train_sample_videos/'
output_dir = 'Faces/'
s = time.time()
ExtractFaces(video_dir_path, output_dir, 50)
print(time.time()-s)