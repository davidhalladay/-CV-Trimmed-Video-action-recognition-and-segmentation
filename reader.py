import numpy as np
import skvideo
import skvideo.io
import skimage.transform
import torch
from torchvision import transforms
import csv
import collections
import os
import matplotlib
import matplotlib.pyplot as plt

import cv2

def normalize(img):
    transform = transforms.Compose([
        # input = (3,240,320)
        transforms.ToPILImage(),
        transforms.Pad((0,40)),
        # input = (3,320,320)
        transforms.Resize(224),
        # reference come from:
        # https://discuss.pytorch.org/t/whats-the-range-of-the-input-value-desired-to-use-pretrained-resnet152-and-vgg19/1683
        transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

def readShortVideo(video_path, video_category, video_name, downsample_factor=12, rescale_factor=1):
    '''
    @param video_path: video directory
    @param video_category: video category (see csv files)
    @param video_name: video name (unique, see csv files)
    @param downsample_factor: number of frames between each sampled frame (e.g., downsample_factor = 12 equals 2fps)
    @param rescale_factor: float of scale factor (rescale the image if you want to reduce computations)

    @return: (T, H, W, 3) ndarray, T indicates total sampled frames, H and W is heights and widths
    '''

    filepath = video_path + '/' + video_category
    filename = [file for file in os.listdir(filepath) if file.startswith(video_name)]
    video = os.path.join(filepath,filename[0])

    videogen = skvideo.io.vreader(video)
    frames = []
    for frameIdx, frame in enumerate(videogen):
        if frameIdx % downsample_factor == 0:
            frame = skimage.transform.rescale(frame, rescale_factor, mode='constant', preserve_range=True, multichannel=True, anti_aliasing=True).astype(np.uint8)
            #frame = skimage.transform.resize(frame, (224, 224))
            frames.append(normalize(frame).numpy())
        else:
            continue

    return np.array(frames).astype(np.float32)

def readFullVideo(video_path, video_category,downsample_factor=1,rescale_factor=1):

    filepath = video_path + '/' + video_category
    video_name_list = sorted(os.listdir(os.path.join(video_path,video_category)))

    frames = []
    for i, frames_name in enumerate(video_name_list):
        print("\r%d/%d" %(i,len(video_name_list)),end="")
        frame_path = os.path.join(filepath,video_name_list[i])
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frames.append(normalize(frame))
    frames = torch.stack(frames, 0)
    return frames

def getVideoList(data_path):
    '''
    @param data_path: ground-truth file path (csv files)

    @return: ordered dictionary of videos and labels {'Action_labels', 'Nouns', 'End_times', 'Start_times', 'Video_category', 'Video_index', 'Video_name'}
    '''
    result = {}

    with open (data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for column, value in row.items():
                result.setdefault(column,[]).append(value)

    od = collections.OrderedDict(sorted(result.items()))
    return od
