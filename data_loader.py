# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:01:35 2020

@author: ckris
"""


from __future__ import print_function, division
import os
import pandas as pd
from PIL import Image
import numpy as np

def data(video_dir_path, output_dir):
    video_dir_path = video_dir_path
    output_dir = output_dir
    metadata = pd.read_json(video_dir_path+'metadata.json').T
    diff_frames = []
    labels = []
    count = 0
    for subdir in os.listdir(output_dir):
        frames = []
        frame_nums = []
        label = metadata[metadata.index == str(subdir)+'.mp4']['label'][0]
        
        for file in os.listdir(os.path.join(output_dir,subdir)):
            img = Image.open(os.path.join(output_dir+subdir, file))
            data = np.array(img)
            frames.append(data)
            frame_num = str(file).split('.')[0].split('_')
            frame_nums.append(int(''.join(frame_num)))
        
        frames = np.array(frames)[np.argsort(frame_nums)]
        frame_nums = np.sort(frame_nums)
        for i in range(len(frame_nums)):
            if(i%2 == 1):
                diff_frames.append(frames[i] - frames[i-1])
                labels.append(label)
            
        # print(count)
        count +=1
        # print(frames)
        # print(frame_nums)
    return pd.DataFrame({'frame': diff_frames, 'label': labels})
    
# video_dir_path = 'deepfake-detection-challenge/train_sample_videos/'
# output_dir = 'Faces/'  
# df = data(video_dir_path, output_dir)