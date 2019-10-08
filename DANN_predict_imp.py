import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import skimage.io
import skimage
import cv2
import os
import time
import pandas as pd
import random
import pickle
import sys
import DANN_model_imp
import DANN_Dataset

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path,map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def predict_img(file,F_model,L_model,image_name,root_path,pred_file):
    img = cv2.imread(os.path.join(root_path, image_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.
    img = img.transpose(2, 0, 1)
    img = torch.FloatTensor(img)
    with torch.no_grad():
        img = Variable(img[:,:,:])
    img = img.cuda()
    img = torch.unsqueeze(img,0)
    tmp_output = F_model(img)
    output = L_model(tmp_output)
    _, output = torch.max(output,1)
    output = output.cpu()
    image_name = image_name.replace("png","png")
    text = "%s,%d" %(image_name,output[0])
    # print(text)
    file.write(text+"\n")
    return True

def mkdir(path):

    path=path.strip()
    path=path.rstrip("\\")

    isExists=os.path.exists(path)

    if not isExists:
        print (path,"successful")
        os.makedirs(path)
        return True
    else:
        print("-"*10)
        print ("Directory already exists.Please remove it.")
        print("-"*10)
        return False

def main():
    F_model = DANN_model_imp.Feature_Extractor()
    L_model = DANN_model_imp.Label_Predictor()

    print('load model...')
    F_model_path = sys.argv[3]
    L_model_path = sys.argv[4]
    load_checkpoint(F_model_path,F_model)
    load_checkpoint(L_model_path,L_model)
    F_model.eval()
    F_model.cuda()
    L_model.eval()
    L_model.cuda()

    root_path = sys.argv[1]
    filenames = os.listdir(root_path)
    filenames = sorted(filenames)

    pred_file = sys.argv[2]
    total = len(filenames)
    iter = 1
    #print("creat the log file...")
    #mkdir(pred_file)
    print('predicting...')

    file = open(pred_file,"w")
    file.write("image_name,label\n")
    for i in filenames:
        if(predict_img(file,F_model,L_model,i,root_path,pred_file)):

            if (iter % 1000 == 0):
                print("[%d/%d] completed" %(iter,total))
            iter += 1

if __name__ == '__main__':
    main()
