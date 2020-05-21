# -*- coding: utf-8 -*-
"""
Created on Sat May  2 21:11:01 2020

@author: ckris
"""


import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np 
import os
import pathlib

AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_label(video_path):
    dir_name = tf.strings.split(video_path, os.path.sep)
    print(dir_name)
    label = labels['label'][labels['name']==dir_name]
    print(label)
    return label

def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    # # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    # img = tf.image.convert_image_dtype(img, tf.float32)
    # # resize the image to the desired size.
    # return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])
    return img

@tf.autograph.experimental.do_not_convert
def process_path(file_path):
    print(file_path)
    label = get_label(file_path)
    print(label)
    image = tf.io.read_file(file_path)
    image = decode_img(image)
    return image, label
    
project_dir = 'C:/Users/ckris/Desktop/DeepFakeProject/'
video_dir_path = project_dir +'deepfake-detection-challenge/train_sample_videos/'
faces = project_dir+'Faces/'


metadata = pd.read_json(os.path.join(video_dir_path, 'metadata.json'))
metadata = metadata.T

labels = pd.DataFrame({'name': [i.split('.')[0] for i in metadata.index],
                      'label': metadata['label']})

list_ds = tf.data.Dataset.list_files(faces+'*/*.jpg')
print(list_ds)

labeled_ds = list_ds.map(process_path)#, num_parallel_calls=AUTOTUNE)

for image, label in labeled_ds.take(1):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())


# for f in list_ds.take(5):
#     video_dir = tf.strings.split(f, os.path.sep)[-2]
#     print(video_dir)