import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from utils.visualize_tool import get_test_img, GradCam, ScoreCam, superimpose, img_resize, get_heatmap_imgs
from utils.preprocessing import loadImg4Classification


with open('./information/visualization_info.json') as info:
    vis_info = json.load(info)


gpu_number = vis_info['gpu_number']
isize      = vis_info['imgSize']
test_dir   = vis_info['test_dir']
model_path = vis_info['model_path']
imageSavePath = vis_info['imageSavePath']
intermediate_layername = vis_info['intermediate_layername']


if gpu_number != None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)


# image number information
img_number = 0
for classidx, imgclass in enumerate(os.listdir(test_dir)):
    img_number += (len(os.listdir(test_dir+imgclass)))
    
print('there have %d images belong to %d classes'%(img_number, classidx+1))


# load the prediction model
prediction_model = load_model(model_path)


# load images
test = loadImg4Classification.test(isize, img_number, test_dir)
test_img = test[0][0]


# make sure the input images have been normalized
print('make sure the input images have been normalized\nthe maximum is:', test_img.max())
assert test_img.max() <= 1


# plot and save the heatmap images for all images in the test folder

for number in range(img_number-685):
    print('saving heatmap image...', number+1)
    fig = get_heatmap_imgs(number, test_img, prediction_model, intermediate_layername)
    fig.savefig(imageSavePath+'heatmap_%s'%(test.filenames[number].split('/')[1]))

